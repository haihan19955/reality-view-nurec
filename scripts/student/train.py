"""Train the video student with flow matching (rectified flow, Wan's native objective). MeanFlow distillation comes after.

    python scripts/student/train.py [--steps 4000 --batch 4] [--out $RV_WORK/checkpoints/student]
    python scripts/student/train.py --geo --init $RV_WORK/checkpoints/student/last.pt --out $RV_WORK/checkpoints/student_geo
"""
import argparse
import glob
import json
import math
import os
import random
import time

import numpy as np
import torch

from realityview import paths
from realityview.student import CAM_FRAMES, EGO_DIM, EGO_DIM_CAM, FEAT_TIMES, Student, ego_vector

VAL_SCENES = ("e8cf41e8", "ff28f961")

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=os.path.join(paths.WORK, "student_cache"))
ap.add_argument("--out", default=os.path.join(paths.CHECKPOINTS, "student"))
ap.add_argument("--wan", default=paths.model("wan2.1-1.3b"))
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--accum", type=int, default=1)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=64)
ap.add_argument("--shift", type=float, default=3.0)
ap.add_argument("--ctx_noise", type=float, default=0.3, help="max noise level of the context frame")
ap.add_argument("--p_drop_ctx", type=float, default=0.1, help="train without a previous frame (rollout start)")
ap.add_argument("--p_drop_feat", type=float, default=0.1, help="drop camera features (enables CFG)")
ap.add_argument("--val_every", type=int, default=500)
ap.add_argument("--save_every", type=int, default=1000)
ap.add_argument("--geo", action="store_true", help="pixel-aligned conditioning on the warped point cloud (prepare_warp.py)")
ap.add_argument("--geo_init", type=float, default=0.5, help="init the warp channels as this x the noisy-latent weights")
ap.add_argument("--geo_lr", type=float, default=10.0, help="lr multiplier of the patch embedding (new geo inputs)")
ap.add_argument("--p_corrupt", type=float, default=0.0,
                help="replace the context frame inside the observed mask by a wrong one (other time / shifted), "
                     "so the warp wins over the previous frame where they disagree")
ap.add_argument("--obs_w", type=float, default=0.0, help="extra loss weight on observed (warp-covered) pixels")
ap.add_argument("--p_drop_geo", type=float, default=0.0, help="drop the warp condition (enables guidance on it)")
ap.add_argument("--init", default="", help="start from a previous student checkpoint")
ap.add_argument("--free_view", action="store_true", help="train on chase + orbit tracks, camera pose in the ego tokens")
ap.add_argument("--feat", default="dino", choices=["dino", "dino3"], help="camera feature cache (DINOv2 / DINOv3)")
ap.add_argument("--val_kind", default="", help="validate only on chunks of this camera kind (chase / orbit)")
ap.add_argument("--stop_at", type=int, default=0, help="save and stop at this step (short A/B runs on the full schedule)")
ap.add_argument("--start_step", type=int, default=0,
                help="continue a run stopped at this step (--init its last.pt): same lr schedule, fresh optimizer")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()


class Data:
    def __init__(self, cache, val):
        self.items, self.scenes = [], {}
        for d in sorted(os.path.dirname(p) for p in glob.glob(os.path.join(cache, "*", "index.json"))):
            if any(v in d for v in VAL_SCENES) != val or (args.geo and not os.path.exists(os.path.join(d, "warp_done"))):
                continue
            idx = json.load(open(os.path.join(d, "index.json")))
            self.scenes[d] = dict(idx=idx, dino=np.load(os.path.join(d, args.feat + ".npy"), mmap_mode="r"), ego=np.load(os.path.join(d, "ego.npy")),
                                  cams=np.load(os.path.join(d, "cams.npy")) if args.free_view else None)
            chunks = [(ci, s) for ci, s in idx["chunks"] if args.free_view or idx["chase"][ci].get("kind", "chase") == "chase"]
            self.items += [(d, ci, s) for ci, s in chunks]
            self.kind = getattr(self, "kind", {})
            for ci, s in chunks:
                self.kind[(d, ci, s)] = idx["chase"][ci].get("kind", "chase")
        self.have = set(self.items)

    def get(self, i, aug=False):
        d, ci, s = self.items[i]
        sc = self.scenes[d]
        z = torch.load(os.path.join(d, "lat", f"c{ci}_{s:04d}.pt")).float()
        f = torch.from_numpy(np.stack([sc["dino"][s + o] for o in FEAT_TIMES]).astype(np.float32))
        if args.free_view:
            e = torch.from_numpy(ego_vector(sc["ego"], s, cams=[sc["cams"][ci, s + o] for o in CAM_FRAMES]))
        else:
            e = torch.from_numpy(ego_vector(sc["ego"], s, sc["idx"]["chase"][ci]))
        g, w = torch.zeros(0), torch.ones(2, z.shape[-2], z.shape[-1])
        if args.geo:
            g = torch.cat([torch.load(os.path.join(d, "lat", f"{p}{ci}_{s:04d}.pt")).float() for p in ("m", "w")])
            w = 1 + args.obs_w * g[:4, 1:].mean(0)                       # observed fraction of latent frames 1, 2
            if aug and random.random() < args.p_corrupt:
                offs = [o for o in (-12, -8, -6, -4, 4, 6, 8, 12) if (d, ci, s + o) in self.have]
                if offs and random.random() < 0.6:
                    o = random.choice(offs)
                    alt = torch.load(os.path.join(d, "lat", f"c{ci}_{s + o:04d}.pt")).float()[:, 0]
                else:
                    alt = torch.roll(z[:, 0], (random.randint(-6, 6), random.randint(-8, 8)), (1, 2))
                obs = (g[0, 0] > 0.5)[None]                                 # observed mask of the context frame
                z[:, 0] = torch.where(obs, alt, z[:, 0])
        return z, f, e, g, w


def sample_t(n, shift):
    u = torch.sigmoid(torch.randn(n))                    # logit-normal
    return shift * u / (1 + (shift - 1) * u)


def loss_fn(model, z, f, e, g, t, tc, drop_feat, w=None):
    """z [B,16,3,H,W]; g [B,20,3,H,W] or empty; t, tc [B]; w [B,2,H,W] loss weight or None; frame 0 is the context."""
    B = z.shape[0]
    tf = torch.stack([tc, t, t], 1).cuda()
    eps = torch.randn_like(z)
    xt = (1 - tf.view(B, 1, 3, 1, 1)) * z + tf.view(B, 1, 3, 1, 1) * eps
    f = f * (~drop_feat).float().view(B, 1, 1, 1, 1).to(f)
    v = model(xt, tf, f, e, g if args.geo else None)
    err = (v - (eps - z))[:, :, 1:] ** 2
    if w is None:
        return err.mean()
    return (err * w[:, None]).mean() / w.mean()


def main():
    torch.manual_seed(0); random.seed(0)
    train, val = Data(args.cache, False), Data(args.cache, True)
    print(f"train {len(train.items)} chunks / {len(train.scenes)} scenes, val {len(val.items)} chunks", flush=True)
    model = Student(args.wan, rank=args.rank, geo=args.geo, geo_init=args.geo_init,
                    ego_dim=EGO_DIM_CAM if args.free_view else EGO_DIM).cuda()
    if args.init:
        own = model.state_dict()        # skip tensors whose shape changed (v1 patch embedding without geo inputs)
        src = torch.load(args.init, map_location="cuda")
        sd = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
        r = model.load_state_dict(sd, strict=False)
        w = src.get("ego.0.weight")
        if w is not None and w.shape != own["ego.0.weight"].shape:    # chase model -> free view: keep the motion inputs
            with torch.no_grad():
                model.ego[0].weight.zero_(); model.ego[0].weight[:, :32] = w[:, :32]
            print("ego input layer: copied the 32 motion columns, camera columns start at zero", flush=True)
        print("init from", args.init, "unexpected", r.unexpected_keys, flush=True)
    if args.feat != "dino":
        print("camera features:", args.feat, flush=True)
    model.dit.enable_gradient_checkpointing()
    params = model.trainable()
    print("trainable %.1fM" % (sum(p.numel() for p in params) / 1e6), flush=True)
    if args.geo:      # keep the pretrained noisy-latent input channels frozen, only the new geo channels learn
        model.dit.patch_embedding.weight.register_hook(lambda g: torch.cat([torch.zeros_like(g[:, :16]), g[:, 16:]], 1))
    pe = [p for n, p in model.named_parameters() if p.requires_grad and "patch_embedding" in n]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and "patch_embedding" not in n]
    opt = torch.optim.AdamW([dict(params=rest), dict(params=pe, lr=args.lr * args.geo_lr)],
                            lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    s0 = args.start_step       # on a continued run the step count starts at s0, with a short re-warmup for the new optimizer
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / 200) * (0.5 * (1 + math.cos(math.pi * (s0 + i) / args.steps)) * 0.9 + 0.1))
    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, "feat.txt"), "w").write(args.feat)     # read by distill.py / the viewer
    rng = random.Random(1)
    pool = [i for i, it in enumerate(val.items) if not args.val_kind or val.kind[it] == args.val_kind]
    val_ids = rng.sample(pool, min(16, len(pool)))
    if args.free_view and not args.val_kind:     # half chase, half orbit, so both are tracked
        ch = [i for i, it in enumerate(val.items) if val.kind[it] == "chase"]
        ob = [i for i, it in enumerate(val.items) if val.kind[it] == "orbit"]
        val_ids = rng.sample(ch, min(8, len(ch))) + rng.sample(ob, min(16, len(ob)))
    log, t0, run = open(os.path.join(args.out, "log.txt"), "a"), time.time(), []
    for step in range(s0 + 1, args.steps + 1):
        model.train()
        for _ in range(args.accum):
            B = args.batch
            z, f, e, g, w = (torch.stack(x).cuda() for x in zip(*[train.get(random.randrange(len(train.items)), aug=True) for _ in range(B)]))
            t = sample_t(B, args.shift)
            tc = torch.rand(B) * args.ctx_noise
            tc[torch.rand(B) < args.p_drop_ctx] = 1.0
            drop = torch.rand(B) < args.p_drop_feat
            if args.geo:
                g = g * (torch.rand(B) >= args.p_drop_geo).float().view(B, 1, 1, 1, 1).cuda()
            loss = loss_fn(model, z, f, e, g, t, tc, drop, w if args.obs_w > 0 else None)
            (loss / args.accum).backward()
            run.append(loss.item())
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % 20 == 0 or args.smoke:
            msg = f"step {step} loss {np.mean(run):.4f} gn {gn:.2f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / (step - s0):.2f}s/it mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
            print(msg, flush=True); log.write(msg + "\n"); log.flush(); run = []
        if step % args.val_every == 0 or (args.smoke and step == 2):
            model.eval(); vl = {}
            with torch.no_grad():
                for tv in (0.2, 0.5, 0.8):
                    for i in val_ids:
                        z, f, e, g, _ = (x[None].cuda() for x in val.get(i))
                        torch.manual_seed(i)
                        kind = val.kind[val.items[i]]
                        vl.setdefault((kind, tv), []).append(
                            loss_fn(model, z, f, e, g, torch.tensor([tv]), torch.tensor([0.1]), torch.tensor([False])).item())
            msg = f"VAL step {step} " + " ".join(f"{k}{tv}:{np.mean(v):.4f}" for (k, tv), v in sorted(vl.items()))
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
        stop = step == args.stop_at or (args.smoke and step == 3)
        if step % args.save_every == 0 or step == args.steps or stop:
            torch.save(model.trainable_state(), os.path.join(args.out, "last.pt"))
        if stop:
            break


if __name__ == "__main__":
    main()
