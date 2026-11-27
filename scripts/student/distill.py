"""MeanFlow distillation of the geo student into a 1-2 step model.

The student predicts the average velocity u(x_t, r, t) = (x_t - x_r) / (t - r) over an interval of the probability-flow
ODE. Instead of the MeanFlow JVP identity, the target is built from its integral definition: the frozen teacher
(the 30-step geo student) integrates the ODE from t to r with a few Heun steps. With r = t the target is the teacher's
instantaneous velocity, so the student stays a valid flow model at every interval length.

    python scripts/student/distill.py --teacher $RV_WORK/checkpoints/student_geo3/last.pt --out $RV_WORK/checkpoints/student_mf
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
from realityview.student import CAM_FRAMES, EGO_DIM_CAM, FEAT_TIMES, Student, ego_vector

VAL_SCENES = ("e8cf41e8", "ff28f961")

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", required=True)
ap.add_argument("--cache", default=os.path.join(paths.WORK, "student_cache"))
ap.add_argument("--out", default=os.path.join(paths.CHECKPOINTS, "student_mf"))
ap.add_argument("--wan", default=paths.model("wan2.1-1.3b"))
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--rank", type=int, default=64)
ap.add_argument("--shift", type=float, default=3.0)
ap.add_argument("--heun_per_unit", type=float, default=6.0, help="teacher Heun steps per unit of t (interval 1 -> 6 steps)")
ap.add_argument("--ctx_noise", type=float, default=0.3)
ap.add_argument("--p_drop_ctx", type=float, default=0.1)
ap.add_argument("--p_corrupt", type=float, default=0.3)
ap.add_argument("--val_every", type=int, default=500)
ap.add_argument("--save_every", type=int, default=500)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
# the teacher checkpoint decides: free-view (camera pose in the ego tokens) or chase-only, and its camera features
_sd = torch.load(args.teacher, map_location="cpu")
EGO_IN = _sd["ego.0.weight"].shape[1]
FREE = EGO_IN == EGO_DIM_CAM
_ft = os.path.join(os.path.dirname(args.teacher), "feat.txt")
FEAT = open(_ft).read().strip() if os.path.exists(_ft) else "dino"
del _sd


class Data:
    def __init__(self, cache, val):
        self.items, self.scenes = [], {}
        for d in sorted(os.path.dirname(p) for p in glob.glob(os.path.join(cache, "*", "index.json"))):
            if any(v in d for v in VAL_SCENES) != val or not os.path.exists(os.path.join(d, "warp_done")):
                continue
            idx = json.load(open(os.path.join(d, "index.json")))
            self.scenes[d] = dict(idx=idx, dino=np.load(os.path.join(d, FEAT + ".npy"), mmap_mode="r"), ego=np.load(os.path.join(d, "ego.npy")),
                                  cams=np.load(os.path.join(d, "cams.npy")) if FREE else None)
            self.items += [(d, ci, s) for ci, s in idx["chunks"] if FREE or idx["chase"][ci].get("kind", "chase") == "chase"]
        self.have = set(self.items)

    def get(self, i, aug=False):
        d, ci, s = self.items[i]
        sc = self.scenes[d]
        lat = lambda p, k: torch.load(os.path.join(d, "lat", f"{p}{ci}_{k:04d}.pt")).float()   # noqa: E731
        z = lat("c", s)
        f = torch.from_numpy(np.stack([sc["dino"][s + o] for o in FEAT_TIMES]).astype(np.float32))
        if FREE:
            e = torch.from_numpy(ego_vector(sc["ego"], s, cams=[sc["cams"][ci, s + o] for o in CAM_FRAMES]))
        else:
            e = torch.from_numpy(ego_vector(sc["ego"], s, sc["idx"]["chase"][ci]))
        g = torch.cat([lat("m", s), lat("w", s)])
        if aug and random.random() < args.p_corrupt:          # same context corruption the teacher was trained with
            offs = [o for o in (-12, -8, -6, -4, 4, 6, 8, 12) if (d, ci, s + o) in self.have]
            alt = lat("c", s + random.choice(offs))[:, 0] if offs and random.random() < 0.6 else \
                torch.roll(z[:, 0], (random.randint(-6, 6), random.randint(-8, 8)), (1, 2))
            z[:, 0] = torch.where((g[0, 0] > 0.5)[None], alt, z[:, 0])
        return z, f, e, g


def shifted(u):
    return args.shift * u / (1 + (args.shift - 1) * u)


def sample_tr(B):
    """t: Wan's shifted logit-normal, sometimes exactly 1 (first sampling step); r: = t (flow), = 0 (one step),
    or anywhere in [0, t]."""
    t = shifted(torch.sigmoid(torch.randn(B)))
    t[torch.rand(B) < 0.15] = 1.0
    u = torch.rand(B)
    r = torch.where(u < 0.25, t, torch.where(u < 0.5, torch.zeros(B), t * torch.rand(B)))
    return t, r


@torch.no_grad()
def teacher_target(teacher, xt, tc, t, r, f, e, g):
    """Average velocity (x_t - x_r) / (t - r) along the teacher ODE (Heun); the teacher velocity where r = t."""
    B = xt.shape[0]
    tfr = lambda tau: torch.stack([tc, tau, tau], 1).cuda()                 # noqa: E731
    v_t = teacher(xt, tfr(t), f, e, g)
    k = max(1, math.ceil(args.heun_per_unit * float((t - r).max())))
    x, v_a = xt.clone(), v_t
    for j in range(k):
        ta, tb = t + (r - t) * j / k, t + (r - t) * (j + 1) / k
        if j > 0:
            v_a = teacher(x, tfr(ta), f, e, g)
        d = (tb - ta).cuda().view(B, 1, 1, 1, 1)
        x_e = x.clone(); x_e[:, :, 1:] = x[:, :, 1:] + d * v_a[:, :, 1:]
        v_b = teacher(x_e, tfr(tb), f, e, g)
        x[:, :, 1:] = x[:, :, 1:] + d * 0.5 * (v_a + v_b)[:, :, 1:]
    dt = (t - r).cuda().view(B, 1, 1, 1, 1)
    u = (xt - x) / dt.clamp(min=1e-4)
    return torch.where(dt > 1e-4, u, v_t)


def noised(z, tc, t):
    B = z.shape[0]
    tf = torch.stack([tc, t, t], 1).cuda().view(B, 1, 3, 1, 1)
    return (1 - tf) * z + tf * torch.randn_like(z)


@torch.no_grad()
def val_x0(student, teacher, val, ids):
    """x0 error of 1-step and 2-step student vs 8-step-Heun teacher from pure noise (fixed seeds), frames 1-2."""
    out = {1: [], 2: []}
    for i in ids:
        z, f, e, g = (x[None].cuda() for x in val.get(i))
        torch.manual_seed(i)
        x1 = noised(z, torch.tensor([0.1]), torch.tensor([1.0]))
        one, zero = torch.tensor([1.0]), torch.tensor([0.0])
        ref = x1 - teacher_target(teacher, x1, torch.tensor([0.1]), one, zero, f, e, g)
        for n in (1, 2):
            grid = shifted(torch.linspace(1, 0, n + 1))
            x = x1.clone()
            for a, b in zip(grid[:-1], grid[1:]):
                a, b = float(a), float(b)
                u = student(x, torch.tensor([[0.1, a, a]]).cuda(), f, e, g, torch.tensor([[0.0, a - b, a - b]]).cuda())
                x[:, :, 1:] = x[:, :, 1:] - (a - b) * u[:, :, 1:]
            out[n].append(((x - ref)[:, :, 1:] ** 2).mean().item())
    return {n: float(np.mean(v)) for n, v in out.items()}


def main():
    torch.manual_seed(0); random.seed(0)
    train, val = Data(args.cache, False), Data(args.cache, True)
    print(f"train {len(train.items)} chunks, val {len(val.items)} chunks", flush=True)
    sd = torch.load(args.teacher, map_location="cuda")
    print("teacher:", "free view" if FREE else "chase only", "features", FEAT, flush=True)
    teacher = Student(args.wan, rank=args.rank, geo=True, ego_dim=EGO_IN).cuda().eval()
    teacher.load_state_dict(sd, strict=False); teacher.requires_grad_(False)
    student = Student(args.wan, rank=args.rank, geo=True, meanflow=True, ego_dim=EGO_IN).cuda()
    r = student.load_state_dict(sd, strict=False)
    print("student missing (new):", [k for k in r.missing_keys if "h_emb" in k][:4], flush=True)
    student.dit.enable_gradient_checkpointing()
    params = student.trainable()
    print("trainable %.1fM" % (sum(p.numel() for p in params) / 1e6), flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * (0.5 * (1 + math.cos(math.pi * s / args.steps)) * 0.9 + 0.1))
    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, "feat.txt"), "w").write(FEAT)
    val_ids = random.Random(1).sample(range(len(val.items)), 8 if not args.smoke else 1)
    log, t0, run = open(os.path.join(args.out, "log.txt"), "a"), time.time(), []
    for step in range(1, args.steps + 1):
        student.train()
        B = args.batch
        z, f, e, g = (torch.stack(x).cuda() for x in zip(*[train.get(random.randrange(len(train.items)), aug=True) for _ in range(B)]))
        t, r = sample_tr(B)
        tc = torch.rand(B) * args.ctx_noise
        tc[torch.rand(B) < args.p_drop_ctx] = 1.0
        xt = noised(z, tc, t)
        target = teacher_target(teacher, xt, tc, t, r, f, e, g)
        h = torch.stack([torch.zeros(B), t - r, t - r], 1).cuda()
        u = student(xt, torch.stack([tc, t, t], 1).cuda(), f, e, g, h)
        loss = ((u - target)[:, :, 1:] ** 2).mean()
        loss.backward()
        run.append(loss.item())
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % 20 == 0 or args.smoke:
            msg = f"step {step} loss {np.mean(run):.4f} gn {gn:.2f} lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / step:.2f}s/it mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
            print(msg, flush=True); log.write(msg + "\n"); log.flush(); run = []
        if step % args.val_every == 0 or (args.smoke and step == 2) or step == 1:
            student.eval()
            vl = val_x0(student, teacher, val, val_ids)
            msg = f"VAL step {step} x0-MSE vs teacher: 1-step {vl[1]:.4f}  2-step {vl[2]:.4f}"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if step % args.save_every == 0 or step == args.steps or (args.smoke and step == 3):
            torch.save(student.trainable_state(), os.path.join(args.out, "last.pt"))
        if args.smoke and step == 3:
            break


if __name__ == "__main__":
    main()
