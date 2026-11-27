"""Compare depth sources on NuRec frames against rendered-lidar ground truth, and as geometry for novel views.

Methods:
  dav2      : old pipeline, Depth-Anything-V2 metric (vkitti, vitl), per image
  da3_pose6 : DA3-LARGE-1.1, the 6 surround views of one frame + known intrinsics/extrinsics (metric via pose alignment)
  da3_pose12: same + the 6 views of the previous frame (0.2 s earlier) for wider baselines
  da3_free6 : DA3 without poses, depth median-scaled to lidar per frame (scale-oracle; shows shape quality only)
  da3metric : DA3METRIC-LARGE per image (what the pipeline uses)
  df_splat / df_gs : DrivingForward (nuScenes weights, zero-shot) depth splat / its own Gaussian rendering
Metrics on input views vs lidar (<=80 m, sky removed): AbsRel, delta<1.25.  Novel view: z-buffer splat the 6 input views'
coloured points into each target view, PSNR on covered pixels + coverage. Also timing.

    python benchmarks/depth/compare_depth.py [--scenes 6] [--frames 4]
Needs Depth-Anything-V2 (metric_depth, vkitti vitl weights), DA3-LARGE-1.1, DA3METRIC-LARGE and DrivingForward in $RV_MODELS.
"""
import argparse
import os
import sys
import json
import time
import numpy as np
import cv2
import torch
from realityview import paths
from realityview.dataset import list_scenes, read_lidar_depth
from realityview.depth import MetricDepth
sys.path.insert(0, paths.model("Depth-Anything-V2", "metric_depth"))
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_3.api import DepthAnything3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drivingforward import DFRunner

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=paths.NUREC_DATA)
ap.add_argument("--out", default=os.path.join(paths.OUTPUTS, "depth_benchmark"))
ap.add_argument("--scenes", type=int, default=6)
ap.add_argument("--frames", type=int, default=4, help="frames per scene")
args = ap.parse_args()
OUT = args.out; os.makedirs(OUT, exist_ok=True)
NSC, NFR = args.scenes, args.frames
dev = "cuda"
da2 = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024], max_depth=80)
da2.load_state_dict(torch.load(paths.model("depth_anything_v2_metric_vkitti_vitl.pth"), map_location="cpu")); da2 = da2.to(dev).eval()
da3 = DepthAnything3.from_pretrained(paths.model("DA3-LARGE-1.1")).to(dev).eval()
dfr = DFRunner()
da3m = MetricDepth(paths.model("DA3METRIC-LARGE"), dev)

def run_da3m(views):
    """DA3 metric mono. Also returns the sky mask used to clean the lidar GT for every method."""
    return da3m([cv2.cvtColor(v["img"], cv2.COLOR_BGR2RGB) for v in views], [v["f"] for v in views], zero_sky=False)

def K(f, W, H): return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float32)
def load_frame(d, meta, fi):
    fr = meta["frames"][fi]; E = np.array(fr["ego2world"]); out = []
    for v in fr["views"]:
        img = cv2.imread(f"{d}/{v['file']}.jpg")
        out.append(dict(v, img=img, lidar=read_lidar_depth(d, v), c2w=E @ np.array(v["c2rig"])))
    return out

def run_da3(views, posed):
    imgs = [cv2.cvtColor(v["img"], cv2.COLOR_BGR2RGB) for v in views]
    H, W = imgs[0].shape[:2]
    kw = {}
    if posed:
        c0 = views[0]["c2w"]   # express in first camera's frame for numerical sanity
        kw = dict(extrinsics=np.stack([np.linalg.inv(np.linalg.inv(c0) @ v["c2w"]) for v in views]).astype(np.float32),
                  intrinsics=np.stack([K(v["f"], W, H) for v in views]))
    p = da3.inference(imgs, process_res=504, **kw)
    return [cv2.resize(p.depth[i], (W, H), interpolation=cv2.INTER_LINEAR) for i in range(len(views))]

def run_dav2(views):
    with torch.no_grad(): return [da2.infer_image(v["img"], 518) for v in views]

def depth_metrics(pred, gt):
    m = (gt > 1) & (gt < 80); p, g = pred[m], gt[m]
    return np.mean(np.abs(p - g) / g), np.mean(np.maximum(p / g, g / p) < 1.25)

def splat(views, depths, tgt):
    H, W = tgt["img"].shape[:2]; f = tgt["f"]; zb = np.full((H, W), np.inf, np.float32); out = np.zeros((H, W, 3), np.uint8)
    w2t = np.linalg.inv(tgt["c2w"])
    for v, D in zip(views, depths):
        ys, xs = np.mgrid[0:H, 0:W]; z = D.reshape(-1); ok = (z > 0.5) & (z < 120)
        P = np.stack([(xs.reshape(-1) - W / 2) / v["f"] * z, (ys.reshape(-1) - H / 2) / v["f"] * z, z, np.ones_like(z)])[:, ok]
        Q = (w2t @ v["c2w"] @ P)[:3]; col = v["img"].reshape(-1, 3)[ok]
        fr = Q[2] > 0.3; Q, col = Q[:, fr], col[fr]
        u = np.round(Q[0] / Q[2] * f + W / 2).astype(int); w = np.round(Q[1] / Q[2] * f + H / 2).astype(int)
        m = (u >= 0) & (u < W) & (w >= 0) & (w < H); u, w, zz, col = u[m], w[m], Q[2][m], col[m]
        o = np.argsort(-zz); u, w, zz, col = u[o], w[o], zz[o], col[o]
        closer = zz < zb[w, u]; u, w, zz, col = u[closer], w[closer], zz[closer], col[closer]
        zb[w, u] = zz; out[w, u] = col
    cov = np.isfinite(zb)
    out = cv2.dilate(out, np.ones((2, 2), np.uint8)) * (~cov[..., None]) + out * cov[..., None]   # tiny hole fill
    cov2 = cv2.dilate(cov.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    mse = np.mean((out[cov2].astype(np.float32) - tgt["img"][cov2].astype(np.float32)) ** 2)
    return out, 10 * np.log10(255 ** 2 / max(mse, 1e-6)), cov2.mean()

scenes = list_scenes(args.data)[:NSC]          # scenes without rear cameras are skipped
R = {k: dict(absrel=[], d1=[], psnr=[], cov=[], t=[]) for k in ["dav2", "da3metric", "da3_pose6", "da3_pose12", "da3_free6", "df_splat", "df_gs"]}
for si, (d, meta) in enumerate(scenes):
    nfr = len(meta["frames"])
    for fi in np.linspace(5, nfr - 1, NFR).astype(int):
        views = load_frame(d, meta, fi); ins = [v for v in views if v["kind"] == "in"]; tg = [v for v in views if v["kind"] != "in"]
        prev = [v for v in load_frame(d, meta, fi - 1) if v["kind"] == "in"]
        dm, sky = run_da3m(ins)
        for v, s in zip(ins, sky): v["lidar"][s] = 0          # rendered lidar has phantom returns in the sky
        for k in R:
            torch.cuda.synchronize(); t0 = time.time()
            if k == "dav2": dep = run_dav2(ins)
            elif k == "da3metric": dep = run_da3m(ins)[0]
            elif k in ("df_splat", "df_gs"):
                G, dep = dfr.reconstruct([v["img"] for v in ins], [v["f"] for v in ins], [np.array(v["c2rig"]) for v in ins])
            elif k == "da3_pose6": dep = run_da3(ins, True)
            elif k == "da3_pose12": dep = run_da3(ins + prev, True)[:6]
            else:
                dep = run_da3(ins, False)
                s = np.median(np.concatenate([v["lidar"][(v["lidar"] > 1) & (v["lidar"] < 80)] / D[(v["lidar"] > 1) & (v["lidar"] < 80)] for v, D in zip(ins, dep)]))
                dep = [D * s for D in dep]
            torch.cuda.synchronize(); R[k]["t"].append(time.time() - t0)
            dep = [np.where(s, 0, D) for D, s in zip(dep, sky)]   # same sky removal for all methods (sky = background, not geometry)
            for v, D in zip(ins, dep):
                a, b = depth_metrics(D, v["lidar"]); R[k]["absrel"].append(a); R[k]["d1"].append(b)
            for j, t in enumerate(tg):
                if k == "df_gs":   # DrivingForward's own Gaussian rendering (its intended output)
                    img, cov = dfr.render_bgr(G, np.array(t["c2rig"]), t["f"])
                    mse = np.mean((img[cov].astype(np.float32) - t["img"][cov].astype(np.float32)) ** 2) if cov.any() else 1e9
                    ps, cv = 10 * np.log10(255 ** 2 / max(mse, 1e-6)), cov.mean()
                else:
                    img, ps, cv = splat(ins, dep, t)
                R[k]["psnr"].append(ps); R[k]["cov"].append(cv)
                if si < 2 and fi == np.linspace(5, nfr - 1, NFR).astype(int)[1] and j < 3:
                    cv2.imwrite(f"{OUT}/s{si}_{t['name']}_{k}.jpg", img)
                    cv2.imwrite(f"{OUT}/s{si}_{t['name']}_gt.jpg", t["img"])
    print(f"scene {si + 1}/{len(scenes)} done", flush=True)
summ = {k: {m: float(np.mean(v[m][1:] if m == "t" else v[m])) for m in v} for k, v in R.items()}
for k, s in summ.items():
    print(f"{k:11s} n={len(R[k]['psnr'])} AbsRel {s['absrel']:.3f}  d<1.25 {s['d1']:.3f}  novel PSNR {s['psnr']:.2f}  cov {s['cov']:.2f}  time {s['t'] * 1000:.0f} ms")
json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)
