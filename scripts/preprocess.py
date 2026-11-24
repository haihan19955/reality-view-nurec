"""Build the MeanFlow training cache from the rendered NuRec dataset, with DA3-metric geometry.

    python scripts/preprocess.py [--data $NUREC_DATA] [--out $RV_CACHE]

Output:
  tok.npy    [N,6,112,256] fp16  DINOv2-L patch tokens of the 6 inputs (8x14 pooled), PCA -> 256 (basis in pca.pt)
  tgt.npy    [N,V,4,26,48] fp16  VAE latents of the target views (384x208)
  geo.npy    [N,V,4,26,48] fp16  VAE latents of the DA3 point splat into the target views
  gmask.npy  [N,V,26,48]  uint8  splat coverage
  lmask.npy  [N,V,26,48]  uint8  loss mask (0 = ego zone)
  rel.npy    [N,V,4,4]    fp32   target camera -> rig
  index.json {"index": [(scene, frame)], "episodes": [scene_dir]}
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from realityview import paths
from realityview.dataset import list_scenes, read_rgb, split_views
from realityview.depth import MetricDepth
from realityview.features import dino_tokens, load_dinov2
from realityview.geometry import LH, LW, TH, TW, ego_zone_mask, geo_render, lift
from realityview.latents import encode, load_vae

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=paths.NUREC_DATA)
ap.add_argument("--out", default=paths.CACHE)
ap.add_argument("--views", type=int, default=6, help="target views per frame")
ap.add_argument("--max_frames", type=int, default=0, help="limit for smoke tests")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev, V = "cuda", args.views

scenes = list_scenes(args.data)
index = [(si, fi) for si, (_, m) in enumerate(scenes) for fi in range(len(m["frames"]))]
if args.max_frames:
    index = index[:args.max_frames]
N = len(index)
print("scenes", len(scenes), "frames", N, flush=True)

dino = load_dinov2(dev)
vae = load_vae(dev)
depth = MetricDepth(paths.model("DA3METRIC-LARGE"), dev)


def load(k):
    si, fi = index[k]
    d, m = scenes[si]
    ins, tg = split_views(m["frames"][fi])
    tg = tg[:V]
    return [read_rgb(d, v) for v in ins], ins, [read_rgb(d, v) for v in tg], tg


# PCA basis on a subset of frames
sub = np.random.RandomState(0).choice(N, min(400, N), replace=False)
X = torch.cat([dino_tokens(dino, load(k)[0]).reshape(-1, 1024) for k in sub])
mu = X.mean(0)
_, S, Vt = torch.linalg.svd((X - mu)[torch.randperm(len(X))[:60000]], full_matrices=False)
basis = Vt[:256].T.contiguous()
print("PCA 256 explained variance %.3f" % ((S[:256] ** 2).sum() / (S ** 2).sum()).item(), flush=True)
torch.save({"mu": mu.cpu(), "basis": basis.cpu()}, f"{args.out}/pca.pt")

tok = np.lib.format.open_memmap(f"{args.out}/tok.npy", "w+", np.float16, (N, 6, 112, 256))
tgt = np.lib.format.open_memmap(f"{args.out}/tgt.npy", "w+", np.float16, (N, V, 4, LH, LW))
geo = np.lib.format.open_memmap(f"{args.out}/geo.npy", "w+", np.float16, (N, V, 4, LH, LW))
gmask = np.zeros((N, V, LH, LW), np.uint8)
lmask = np.zeros((N, V, LH, LW), np.uint8)
rel = np.zeros((N, V, 4, 4), np.float32)
for k in range(N):
    imgs, ins, timgs, tg = load(k)
    fs = [v["f"] for v in ins]
    with torch.no_grad():
        tok[k] = ((dino_tokens(dino, imgs) - mu) @ basis).view(6, 112, 256).half().cpu().numpy()
        deps, _ = depth(imgs, fs)
        pts, col = lift(torch.tensor(np.stack(deps), device=dev), torch.tensor(np.stack(imgs), device=dev),
                        torch.tensor(np.stack([v["c2rig"] for v in ins]), dtype=torch.float32, device=dev), fs)
        renders, masks = [], []
        for i, v in enumerate(tg):
            rel[k, i] = np.array(v["c2rig"], np.float32)
            img, msk = geo_render(pts, col, torch.tensor(rel[k, i], device=dev))
            renders.append(img)
            masks.append(msk)
        small = np.stack([cv2.resize(t, (TW, TH), interpolation=cv2.INTER_AREA) for t in timgs])
        tgt[k] = encode(vae, torch.tensor(small, device=dev)).cpu().numpy()
        geo[k] = encode(vae, torch.stack(renders)).cpu().numpy()
        gmask[k] = (F.avg_pool2d(torch.stack(masks).float()[:, None], 8)[:, 0] > 0.5).cpu().numpy()
        lmask[k] = (~ego_zone_mask(torch.tensor(rel[k], device=dev))).cpu().numpy()
    if k % 200 == 0:
        print(k, N, "coverage %.2f  loss-mask %.2f" % (gmask[:k + 1].mean(), lmask[:k + 1].mean()), flush=True)

np.save(f"{args.out}/gmask.npy", gmask)
np.save(f"{args.out}/lmask.npy", lmask)
np.save(f"{args.out}/rel.npy", rel)
json.dump({"index": index, "episodes": [d for d, _ in scenes]}, open(f"{args.out}/index.json", "w"))
tok.flush()
tgt.flush()
geo.flush()
print("done", flush=True)
