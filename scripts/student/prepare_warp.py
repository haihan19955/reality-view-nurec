"""Pixel-aligned geometric conditioning for the video student: an accumulated point cloud of the 6 rig cameras,
rendered into the chase camera.

    python scripts/student/prepare_warp.py [--scenes id,...] [--no_latents]

Per frame k and chase camera i:
  1. dense metric depth of every rig image = DA3METRIC, scaled to the sparse lidar depth (median ratio), with depth
     edges removed (they produce flying pixels)
  2. points of the current frame k are splatted into the chase view first (z-buffer); points of the previous frames
     k-2 .. k-20 only fill pixels the current frame leaves empty, so moving objects do not leave ghost trails
  3. writes teacher/<scene>/chase<i>/<k>_warp.jpg and <k>_wmask.png (255 = observed)
Then, for every training chunk in student_cache/<scene>/index.json, the warped 9-frame clip is VAE-encoded
(lat/w<i>_<k>.pt, [16,3,60,104]) and the mask is packed the way Wan I2V packs it (lat/m<i>_<k>.pt, [4,3,60,104]).
At inference the depth would come from DA3 alone (camera-only car); the lidar only fixes the scale here.
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import torch

from realityview import paths
from realityview.warp import PAST, Splatter, dense_depth, pack_mask, read_inputs

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", default=os.path.join(paths.WORK, "teacher"))
ap.add_argument("--cache", default=os.path.join(paths.WORK, "student_cache"))
ap.add_argument("--wan", default=paths.model("wan2.1-1.3b"))
ap.add_argument("--da3", default=paths.model("DA3METRIC-LARGE"))
ap.add_argument("--scenes", default="")
ap.add_argument("--no_latents", action="store_true")
args = ap.parse_args()
dev = "cuda"


def main():
    from realityview.depth import MetricDepth
    da3 = MetricDepth(args.da3)
    vae = None
    for d in sorted(os.path.dirname(p) for p in glob.glob(os.path.join(args.teacher, "*", "meta.json"))):
        if args.scenes and not any(s in d for s in args.scenes.split(",")):
            continue
        meta = json.load(open(os.path.join(d, "meta.json")))
        cdir = os.path.join(args.cache, meta["scene"])
        idx = json.load(open(os.path.join(cdir, "index.json")))
        todo = [(ci, s) for ci, s in idx["chunks"] if not os.path.exists(os.path.join(cdir, "lat", f"w{ci}_{s:04d}.pt"))]
        if not todo and os.path.exists(os.path.join(cdir, "warp_done")):
            print("skip", meta["scene"]); continue
        t0 = time.time()
        # incremental: only cameras whose warps are not rendered yet (new orbit tracks)
        cis = [ci for ci in range(len(meta["chase"])) if not os.path.exists(os.path.join(d, meta["frames"][-1]["chase"][ci]["file"] + "_wmask.png"))]
        depth = torch.from_numpy(dense_depth(da3, d, meta)).to(dev) if cis else None
        t1 = time.time()
        sp = Splatter(meta)
        frames = meta["frames"]
        E = [torch.tensor(f["ego2world"], device=dev, dtype=torch.float32) for f in frames]
        cache = {}

        def pts(j):
            if j not in cache:
                rgb = torch.from_numpy(read_inputs(d, frames[j]["k"])).to(dev)
                cache[j] = sp.points(depth[j], rgb, E[j])
                for old in [x for x in cache if x < j - max(PAST)]:
                    del cache[old]
            return cache[j]

        cover = []
        for j, fr in enumerate(frames if cis else []):
            layers = [pts(j)] + [pts(j - p) for p in PAST if j - p >= 0]
            for ci in cis:
                ch = fr["chase"][ci]
                c2w = E[j] @ torch.tensor(ch["c2rig"], device=dev, dtype=torch.float32)
                img, m = sp.render(layers, c2w)
                cv2.imwrite(os.path.join(d, ch["file"] + "_warp.jpg"), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
                cv2.imwrite(os.path.join(d, ch["file"] + "_wmask.png"), m.astype(np.uint8) * 255)
                cover.append(m.mean())
        t2 = time.time()
        depth = cache = None                        # free the GPU memory before the VAE
        n = 0
        if not args.no_latents:
            if vae is None:
                from diffusers import AutoencoderKLWan
                vae = AutoencoderKLWan.from_pretrained(args.wan, subfolder="vae", torch_dtype=torch.bfloat16).to(dev).eval()
                lm = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
                ls = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
            for ci, s in todo:
                files = [frames[s + j]["chase"][ci]["file"] for j in range(idx["chunk"])]
                fr9 = [cv2.imread(os.path.join(d, f + "_warp.jpg"))[..., ::-1] for f in files]
                mk = [cv2.imread(os.path.join(d, f + "_wmask.png"), cv2.IMREAD_GRAYSCALE) > 127 for f in files]
                x = torch.from_numpy(np.stack(fr9)).to(dev).permute(3, 0, 1, 2)[None].float().div(127.5).sub(1)
                with torch.no_grad():
                    z = (vae.encode(x.bfloat16()).latent_dist.mode().float() - lm) / ls
                torch.save(z[0].bfloat16().cpu().clone(), os.path.join(cdir, "lat", f"w{ci}_{s:04d}.pt"))
                torch.save(pack_mask(mk).bfloat16(), os.path.join(cdir, "lat", f"m{ci}_{s:04d}.pt"))
                n += 1
            open(os.path.join(cdir, "warp_done"), "w").write("ok")
        print(f"done {meta['scene']} depth {t1 - t0:.0f}s warp {t2 - t1:.0f}s latents {n} {time.time() - t2:.0f}s  coverage {np.mean(cover):.2f}", flush=True)


if __name__ == "__main__":
    main()
