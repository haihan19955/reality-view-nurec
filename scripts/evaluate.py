"""Evaluate checkpoints on the held-out scenes of the cache.

    python scripts/evaluate.py work/checkpoints/step*.pt

PSNR (outside the ego zone, 384x208) of:
  geo_only   DA3 point splat, holes black
  mf_geo     MeanFlow 1-step with the geometry condition
  mf_nogeo   MeanFlow 1-step without geometry
  composite  real (splat) pixels where covered, MeanFlow in the holes
Writes <outdir>/metrics.json and a contact sheet per checkpoint (target | geometry | MeanFlow | composite).
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from realityview import paths
from realityview.geometry import TH, TW, plucker
from realityview.latents import decode, load_vae
from realityview.model import load_checkpoint, sample

ap = argparse.ArgumentParser()
ap.add_argument("ckpts", nargs="+")
ap.add_argument("--cache", default=paths.CACHE)
ap.add_argument("--out", default=os.path.join(paths.OUTPUTS, "eval"))
ap.add_argument("--max_views", type=int, default=240)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev, C = "cuda", args.cache

index = json.load(open(f"{C}/index.json"))["index"]
vae = load_vae(dev)
tok = np.load(f"{C}/tok.npy", mmap_mode="r")
tgt = np.load(f"{C}/tgt.npy", mmap_mode="r")
geo = np.load(f"{C}/geo.npy", mmap_mode="r")
gm, lm, rel = np.load(f"{C}/gmask.npy"), np.load(f"{C}/lmask.npy"), np.load(f"{C}/rel.npy")


def psnr(a, b, w):
    """a, b [B,3,H,W] 0..255; w [B,1,H,W] pixel weights."""
    mse = (((a - b) ** 2) * w).flatten(1).sum(1) / (w.flatten(1).sum(1) * 3).clamp(min=1)
    return (10 * torch.log10(255 ** 2 / mse.clamp(min=1e-6))).cpu().numpy()


results = {}
for ck_path in args.ckpts:
    model, ck = load_checkpoint(ck_path, dev)
    ks = [k for k, (sc, _) in enumerate(index) if sc in ck["val_sc"]]
    ks = ks[::max(1, len(ks) * 6 // args.max_views)]
    acc = {"geo_only": [], "mf_geo": [], "mf_nogeo": [], "composite": []}
    gen = torch.Generator(device=dev).manual_seed(0)
    sheet = []
    for bi in range(0, len(ks), 8):
        kb = ks[bi:bi + 8]
        kk, vv = np.repeat(kb, 6), np.tile(np.arange(6), len(kb))
        x = torch.tensor(tgt[kk, vv], device=dev).float()
        g = torch.tensor(geo[kk, vv], device=dev).float()
        m = torch.tensor(gm[kk, vv], device=dev).float()[:, None]
        w = torch.tensor(lm[kk, vv], device=dev).float()[:, None]
        rays = plucker(torch.tensor(rel[kk, vv], device=dev))
        noise = torch.randn(x.shape, device=dev, generator=gen)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ctx = model.context(torch.tensor(tok[kk], device=dev).float())
            z1 = sample(model, ctx, rays, g, m, noise=noise)
            z0 = sample(model, ctx, rays, torch.zeros_like(g), torch.zeros_like(m), noise=noise)
        X, G, Y1, Y0 = decode(vae, x), decode(vae, g), decode(vae, z1.float()), decode(vae, z0.float())
        W = F.interpolate(w, size=(TH, TW), mode="nearest")
        Mv = F.interpolate(m, size=(TH, TW), mode="nearest")
        soft = F.avg_pool2d(Mv, 5, 1, 2) * Mv           # latent-res coverage, slightly feathered
        comp = G * soft + Y1 * (1 - soft)
        acc["geo_only"] += list(psnr(G * Mv, X, W))
        acc["mf_geo"] += list(psnr(Y1, X, W))
        acc["mf_nogeo"] += list(psnr(Y0, X, W))
        acc["composite"] += list(psnr(comp, X, W))
        if bi == 0:
            for j in range(0, 24, 4):
                sheet.append(np.hstack([t[j].permute(1, 2, 0).byte().cpu().numpy()[..., ::-1] for t in (X, G * Mv, Y1, comp)]))
    name = os.path.splitext(os.path.basename(ck_path))[0]
    results[name] = {k: float(np.mean(v)) for k, v in acc.items()}
    print(name, " ".join(f"{k} {np.mean(v):.2f}" for k, v in acc.items()), f"(n={len(acc['mf_geo'])})", flush=True)
    cv2.imwrite(f"{args.out}/{name}_sheet.jpg", np.vstack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 88])
json.dump(results, open(f"{args.out}/metrics.json", "w"), indent=1)
