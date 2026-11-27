"""Evaluate student rollouts (videos written by scripts/student/sample.py: row 0 = student, row 1 = teacher).

    python benchmarks/student/evaluate.py [--out $RV_WORK/outputs/eval]

Per video, over all generated frames (frame 0 is the teacher seed and is skipped):
  PSNR / SSIM          pixel agreement with the teacher (NuRec + Difix) - favours blur, read with the others
  DINO                 mean cosine similarity of DINOv2-L patch tokens with the teacher (semantic / layout agreement)
  sharpness            Laplacian variance relative to the teacher (1 = as sharp, < 1 = blurrier)
  flicker              mean abs error after warping frame t+1 onto t with its own optical flow, relative to the teacher
                       computed the same way (1 = as stable as the teacher)
  drift                PSNR of the first 16 generated frames minus the last 16
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch

from realityview import paths
from realityview.features import MEAN, STD, load_dinov2

RUNS = {   # name -> (folder, file suffix, seconds per 8-frame chunk on an RTX 4090)
    "v1 (no geometry), 30 steps": ("student", "_teacher_s30_cfg1.0.mp4", 8.3),
    "geo v3, 30 steps": ("student_geo3", "_teacher_s30_cfg1.0_geo.mp4", 8.3),
    "geo v3, 2 steps (not distilled)": ("student_mf", "_teacher_s2_cfg1.0_geo.mp4", 0.6),
    "MeanFlow, 2 steps": ("student_mf", "_teacher_s2_cfg1.0_geo_mf.mp4", 1.2),
    "MeanFlow, 1 step": ("student_mf", "_teacher_s1_cfg1.0_geo_mf.mp4", 0.9),
}
CLIPS = {"e8cf41e8_c0": "held out", "e8cf41e8_c1": "held out", "ff28f961_c0": "held out", "166fe701_c0": "train"}

ap = argparse.ArgumentParser()
ap.add_argument("--outputs", default=paths.OUTPUTS)
ap.add_argument("--out", default=os.path.join(paths.OUTPUTS, "eval"))
args = ap.parse_args()


def read(p):
    cap, gen, tea = cv2.VideoCapture(p), [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        gen.append(f[:480]); tea.append(f[480:960])
    return gen[1:], tea[1:]


def psnr(a, b):
    return 10 * np.log10(255 ** 2 / max(np.mean((a.astype(np.float32) - b) ** 2), 1e-6))


def ssim(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64); b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)            # noqa: E731
    ma, mb = g(a), g(b)
    va, vb, cab = g(a * a) - ma ** 2, g(b * b) - mb ** 2, g(a * b) - ma * mb
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return float(np.mean((2 * ma * mb + c1) * (2 * cab + c2) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))))


def sharp(x):
    return cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def flicker(frames):
    e = []
    for a, b in zip(frames[:-1], frames[1:]):
        ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        fl = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 4, 21, 3, 5, 1.1, 0)
        h, w = ga.shape
        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        mx, my = (gx + fl[..., 0]).astype(np.float32), (gy + fl[..., 1]).astype(np.float32)
        bw = cv2.remap(b, mx, my, cv2.INTER_LINEAR)
        ok = (mx > 1) & (mx < w - 2) & (my > 1) & (my < h - 2)
        e.append(np.abs(a.astype(np.float32) - bw).mean(-1)[ok].mean())
    return float(np.mean(e))


@torch.no_grad()
def dino_sim(net, gen, tea):
    out = []
    for i in range(0, len(gen), 16):
        ims = [cv2.resize(x[..., ::-1], (448, 252), interpolation=cv2.INTER_AREA)[:252 // 14 * 14] for x in gen[i:i + 16] + tea[i:i + 16]]
        x = torch.from_numpy(np.stack(ims)).cuda().permute(0, 3, 1, 2).float() / 255
        x = (x - torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)) / torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
        t = net.forward_features(x.half())["x_norm_patchtokens"].float()
        n = len(t) // 2
        out += torch.nn.functional.cosine_similarity(t[:n], t[n:], dim=-1).mean(1).tolist()
    return float(np.mean(out))


def main():
    os.makedirs(args.out, exist_ok=True)
    net = load_dinov2()
    rows = []
    for name, (folder, suffix, spc) in RUNS.items():
        for clip, split in CLIPS.items():
            p = os.path.join(args.outputs, folder, clip + suffix)
            if not os.path.exists(p):
                continue
            gen, tea = read(p)
            ps = [psnr(a, b) for a, b in zip(gen, tea)]
            r = dict(run=name, clip=clip, split=split, frames=len(gen), sec_per_chunk=spc,
                     psnr=float(np.mean(ps)), ssim=float(np.mean([ssim(a, b) for a, b in zip(gen[::4], tea[::4])])),
                     dino=dino_sim(net, gen, tea),
                     sharpness=float(np.mean([sharp(x) for x in gen]) / np.mean([sharp(x) for x in tea])),
                     flicker=flicker(gen[::2]) / flicker(tea[::2]),
                     drift=float(np.mean(ps[:16]) - np.mean(ps[-16:])))
            rows.append(r)
            print(json.dumps(r), flush=True)
    json.dump(rows, open(os.path.join(args.out, "student_eval.json"), "w"), indent=1)

    keys = ["psnr", "ssim", "dino", "sharpness", "flicker", "drift"]
    lines = ["| run | clips | s / 8 frames | PSNR ↑ | SSIM ↑ | DINO ↑ | sharpness (1 = teacher) | flicker (1 = teacher) ↓ | drift dB ↓ |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in RUNS:
        rs = [r for r in rows if r["run"] == name and r["split"] == "held out"]
        if not rs:
            continue
        m = {k: np.mean([r[k] for r in rs]) for k in keys}
        lines.append(f"| {name} | {len(rs)} | {rs[0]['sec_per_chunk']} | {m['psnr']:.2f} | {m['ssim']:.3f} | {m['dino']:.3f} | "
                     f"{m['sharpness']:.2f} | {m['flicker']:.2f} | {m['drift']:.2f} |")
    md = "Held-out clips only (e8cf41e8 chase 0/1, ff28f961 chase 0); v1 has no ff28f961 rollout.\n\n" + "\n".join(lines) + "\n"
    open(os.path.join(args.out, "student_eval.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
