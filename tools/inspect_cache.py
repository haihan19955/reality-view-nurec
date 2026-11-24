"""Decode a few cached (target, geometry) latents and tint the loss-masked ego zone red, to sanity-check preprocessing.

    python tools/inspect_cache.py [--cache $RV_CACHE]
"""
import argparse
import os

import cv2
import numpy as np
import torch

from realityview import paths
from realityview.latents import decode, load_vae

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=paths.CACHE)
args = ap.parse_args()
C = args.cache
vae = load_vae()
tgt, geo = np.load(f"{C}/tgt.npy", mmap_mode="r"), np.load(f"{C}/geo.npy", mmap_mode="r")
lm, gm = np.load(f"{C}/lmask.npy"), np.load(f"{C}/gmask.npy")


def dec(z):
    return decode(vae, torch.tensor(np.array(z), device="cuda")).permute(0, 2, 3, 1).byte().cpu().numpy()[..., ::-1]


rows = []
for k in [0, 10, 20]:
    T, G = dec(tgt[k]), dec(geo[k])
    for v in range(0, 6, 2):
        m = cv2.resize(lm[k, v].astype(np.uint8), (T.shape[2], T.shape[1]), interpolation=cv2.INTER_NEAREST)
        t = T[v].copy()
        t[m == 0] = (t[m == 0] * 0.4 + np.array([0, 0, 150])).astype(np.uint8)
        rows.append(np.hstack([t, G[v]]))
os.makedirs(paths.OUTPUTS, exist_ok=True)
out = os.path.join(paths.OUTPUTS, "cache_check.jpg")
cv2.imwrite(out, np.vstack([np.hstack(rows[i:i + 3]) for i in range(0, len(rows), 3)]))
print("geometry coverage %.2f  loss mask kept %.2f  ->" % (gm.mean(), lm.mean()), out)
