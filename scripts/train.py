"""Train the MeanFlow DiT on the preprocessed cache.

    python scripts/train.py --steps 8000 [--init carla_pretrained.pt]

MeanFlow objective with a JVP target, adaptive loss weighting, loss masked inside the ego zone (unreliable NuRec
near field, covered by the CG car), geometry-condition dropout, EMA 0.999. The last --n_val scenes are held out.
Checkpoints {ema, step, val_sc, hl, wl} every --ck_every steps.
"""
import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
from torch.func import jvp

from realityview import paths
from realityview.geometry import LH, LW, plucker
from realityview.model import MVDiT

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=8000)
ap.add_argument("--bs", type=int, default=24)
ap.add_argument("--cache", default=paths.CACHE)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--geo_drop", type=float, default=0.3, help="probability of dropping the geometry condition")
ap.add_argument("--init", default="", help="optional checkpoint to start from (EMA weights; shape-mismatched tensors are skipped)")
ap.add_argument("--out", default=paths.CHECKPOINTS)
ap.add_argument("--n_val", type=int, default=2)
ap.add_argument("--ck_every", type=int, default=2500)
ap.add_argument("--smoke", action="store_true", help="200 steps")
args = ap.parse_args()
C = args.cache
os.makedirs(args.out, exist_ok=True)
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True

meta = json.load(open(f"{C}/index.json"))
index = meta["index"]
n_sc = len(meta["episodes"])
val_sc = list(range(n_sc - args.n_val, n_sc))
V = np.load(f"{C}/rel.npy", mmap_mode="r").shape[1]
pairs = torch.tensor([(k, v) for k, (s, _) in enumerate(index) if s not in val_sc for v in range(V)], device=dev)
print("train pairs", len(pairs), "| val scenes", [os.path.basename(meta["episodes"][s]) for s in val_sc], flush=True)

tok = torch.from_numpy(np.load(f"{C}/tok.npy", mmap_mode="r")[:]).pin_memory()
tgt = torch.from_numpy(np.load(f"{C}/tgt.npy", mmap_mode="r")[:]).to(dev)
geo = torch.from_numpy(np.load(f"{C}/geo.npy", mmap_mode="r")[:]).to(dev)
gm = torch.from_numpy(np.load(f"{C}/gmask.npy")).to(dev)
lm = torch.from_numpy(np.load(f"{C}/lmask.npy")).to(dev)
rel = torch.from_numpy(np.load(f"{C}/rel.npy")).to(dev)

model = MVDiT(hl=LH, wl=LW).to(dev)
if args.init:
    sd = torch.load(os.path.expanduser(args.init), map_location=dev)["ema"]
    own = model.state_dict()
    ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    own.update(ok)
    model.load_state_dict(own)
    print(f"init from {args.init}: {len(ok)}/{len(own)} tensors (skipped {[k for k in own if k not in ok]})", flush=True)
ema = copy.deepcopy(model).eval().requires_grad_(False)
print("params %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
steps_total = 200 if args.smoke else args.steps
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, steps_total) / steps_total)))
hist_l, hist_g, skipped = [], [], 0


def sample_tr(B):
    """(r, t) with t >= r; 75% of the batch uses r = t (plain flow matching)."""
    a = torch.sigmoid(torch.randn(B, device=dev) - 0.4)
    b = torch.sigmoid(torch.randn(B, device=dev) - 0.4)
    t, r = torch.maximum(a, b), torch.minimum(a, b)
    return torch.where(torch.rand(B, device=dev) < 0.75, t, r), t


t0 = time.time()
for step in range(1, steps_total + 1):
    i = pairs[torch.randint(len(pairs), (args.bs,), device=dev)]
    k, v = i[:, 0], i[:, 1]
    x = tgt[k, v].float()
    rays = plucker(rel[k, v])
    g = geo[k, v].float()
    m = gm[k, v].float()[:, None]
    w_pix = lm[k, v].float()[:, None]
    drop = (torch.rand(args.bs, device=dev) < args.geo_drop)[:, None, None, None]
    g = torch.where(drop, torch.zeros_like(g), g)
    m = torch.where(drop, torch.zeros_like(m), m)
    tk = tok[k.cpu()].to(dev, non_blocking=True).float() * (~(torch.rand(args.bs, 6, device=dev) < 0.05))[:, :, None, None]   # camera dropout
    r, t = sample_tr(args.bs)
    e = torch.randn_like(x)
    T = t[:, None, None, None]
    z = (1 - T) * x + T * e
    vel = e - x
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ctx = model.context(tk)
        u, dudt = jvp(lambda z_, r_, t_: model(z_, r_, t_, ctx, rays, g, m), (z, r, t), (vel, torch.zeros_like(r), torch.ones_like(t)))
    u_tgt = (vel - (t - r)[:, None, None, None] * dudt.float()).detach()
    err = (u.float() - u_tgt) ** 2 * w_pix
    l = err.flatten(1).sum(1) / (w_pix.flatten(1).sum(1) * 4).clamp(min=1)
    loss = (l / (l.detach() + 1e-3)).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    lmv = l.mean().item()
    bad = not (math.isfinite(gn) and math.isfinite(lmv))
    if len(hist_l) >= 100:   # spike guard
        bad = bad or lmv > 8 * float(np.median(hist_l)) or gn > 8 * float(np.median(hist_g))
    if bad:
        skipped += 1
        opt.zero_grad(set_to_none=True)
    else:
        opt.step()
        hist_l = (hist_l + [lmv])[-500:]
        hist_g = (hist_g + [gn])[-500:]
    sched.step()
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.lerp_(pm, 1 - 0.999)
    if step % 200 == 0:
        print(f"step {step} mse {lmv:.4f} med {np.median(hist_l) if hist_l else 0:.3f} gn {gn:.2f} lr {sched.get_last_lr()[0]:.2e} "
              f"skipped {skipped} {(time.time() - t0) / step * 1000:.0f}ms/it", flush=True)
    if step % args.ck_every == 0 or step == steps_total:
        ck = {"ema": ema.state_dict(), "step": step, "val_sc": val_sc, "hl": LH, "wl": LW}
        torch.save(ck, f"{args.out}/last.pt")
        torch.save(ck, f"{args.out}/step{step / 1000:04.1f}k.pt")
print("done", flush=True)
