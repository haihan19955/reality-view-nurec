"""Autoregressive rollout of the video student on a scene: 8 new chase frames per chunk, each chunk conditioned on the
last generated frame (+ noise), the 6 rig cameras and ego motion.

    python scripts/student/sample.py --ckpt $RV_WORK/checkpoints/student/last.pt --scene e8cf41e8 [--chase 0] [--steps 30]

Writes an mp4 (top: student, bottom: Difix-cleaned teacher) and prints per-chunk PSNR vs the teacher, to see drift.
--start teacher seeds the first context frame from the teacher; --start none generates the first chunk from scratch.
"""
import argparse
import glob
import json
import os
import subprocess
import time

import cv2
import numpy as np
import torch

from realityview import paths
from realityview.student import CAM_FRAMES, EGO_DIM_CAM, FEAT_TIMES, Student, ego_vector

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--cache", default=os.path.join(paths.WORK, "student_cache"))
ap.add_argument("--teacher", default=os.path.join(paths.WORK, "teacher"))
ap.add_argument("--wan", default=paths.model("wan2.1-1.3b"))
ap.add_argument("--scene", default="e8cf41e8")
ap.add_argument("--chase", type=int, default=0)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--shift", type=float, default=3.0)
ap.add_argument("--cfg", type=float, default=1.0)
ap.add_argument("--gcfg", type=float, default=1.0, help="guidance on the point-cloud condition (unconditional = no warp)")
ap.add_argument("--ctx_noise", type=float, default=0.1)
ap.add_argument("--chunks", type=int, default=12)
ap.add_argument("--start", default="teacher", choices=["teacher", "none"])
ap.add_argument("--rank", type=int, default=64)
ap.add_argument("--geo", action="store_true")
ap.add_argument("--meanflow", action="store_true", help="distilled average-velocity model (distill.py); use --steps 1 or 2")
ap.add_argument("--out", default=os.path.join(paths.OUTPUTS, "student"))
args = ap.parse_args()


def ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def main():
    from diffusers import AutoencoderKLWan
    cdir = [d for d in glob.glob(os.path.join(args.cache, "*")) if args.scene in d][0]
    idx = json.load(open(os.path.join(cdir, "index.json")))
    tdir = os.path.join(args.teacher, idx["scene"])
    meta = json.load(open(os.path.join(tdir, "meta.json")))
    sd = torch.load(args.ckpt, map_location="cuda")
    ego_in = sd["ego.0.weight"].shape[1]
    free = ego_in == EGO_DIM_CAM            # free-view checkpoint: camera pose per chunk in the ego tokens
    ft = os.path.join(os.path.dirname(args.ckpt), "feat.txt")
    feat = open(ft).read().strip() if os.path.exists(ft) else "dino"
    dino, ego = np.load(os.path.join(cdir, feat + ".npy"), mmap_mode="r"), np.load(os.path.join(cdir, "ego.npy"))
    cams = np.load(os.path.join(cdir, "cams.npy")) if free else None
    starts = sorted(s for ci, s in idx["chunks"] if ci == args.chase)
    k0 = starts[0]
    n_chunks = min(args.chunks, (idx["n"] - 1 - k0) // 8)

    model = Student(args.wan, rank=args.rank, geo=args.geo, meanflow=args.meanflow, ego_dim=ego_in).cuda().eval()
    missing = model.load_state_dict(sd, strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    vae = AutoencoderKLWan.from_pretrained(args.wan, subfolder="vae", torch_dtype=torch.bfloat16).cuda().eval()
    lm = torch.tensor(vae.config.latents_mean, device="cuda").view(1, -1, 1, 1, 1)
    ls = torch.tensor(vae.config.latents_std, device="cuda").view(1, -1, 1, 1, 1)
    enc = lambda x: (vae.encode(x.bfloat16()).latent_dist.mode().float() - lm) / ls           # noqa: E731
    dec = lambda z: vae.decode((z * ls + lm).bfloat16()).sample.float().clamp(-1, 1)          # noqa: E731
    to_t = lambda im: torch.from_numpy(im[..., ::-1].copy()).cuda().permute(2, 0, 1)[:, None][None].float() / 127.5 - 1  # noqa: E731
    to_im = lambda x: ((x + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()[..., ::-1]  # noqa: E731
    teach = lambda k: cv2.imread(os.path.join(tdir, meta["frames"][k]["chase"][args.chase]["file"] + "_fix.jpg"))  # noqa: E731

    sig = torch.linspace(1, 0, args.steps + 1)
    sig = args.shift * sig / (1 + (args.shift - 1) * sig)
    g = torch.Generator(device="cuda").manual_seed(0)
    frames, psnr = [teach(k0)], []
    with torch.no_grad():
        zc = enc(to_t(teach(k0))) if args.start == "teacher" else None
        t0 = time.time()
        for c in range(n_chunks):
            k = k0 + 8 * c
            f = torch.from_numpy(np.stack([dino[k + o] for o in FEAT_TIMES]).astype(np.float32))[None].cuda()
            if free:
                e = torch.from_numpy(ego_vector(ego, k, cams=[cams[args.chase, k + o] for o in CAM_FRAMES]))[None].cuda()
            else:
                e = torch.from_numpy(ego_vector(ego, k, idx["chase"][args.chase]))[None].cuda()
            geo = None
            if args.geo:
                lp = lambda p: os.path.join(cdir, "lat", f"{p}{args.chase}_{k:04d}.pt")      # noqa: E731
                if not os.path.exists(lp("w")):
                    print("no warp latent for chunk at", k, "- stopping"); break
                geo = torch.cat([torch.load(lp("m")), torch.load(lp("w"))]).float()[None].cuda()
            x = torch.randn(1, 16, 3, 60, 104, device="cuda", generator=g)
            if zc is None:
                tc = 1.0; ctx = x[:, :, :1].clone()
            else:
                tc = args.ctx_noise; ctx = (1 - tc) * zc + tc * torch.randn(zc.shape, device="cuda", generator=g)
            for i in range(args.steps):
                x[:, :, :1] = ctx
                tf = torch.tensor([[tc, sig[i], sig[i]]], device="cuda")
                hd = float(sig[i] - sig[i + 1])
                v = model(x, tf, f, e, geo, torch.tensor([[0.0, hd, hd]], device="cuda") if args.meanflow else None)
                if args.cfg != 1.0:
                    vu = model(x, tf, f * 0, e, geo)
                    v = vu + args.cfg * (v - vu)
                if args.geo and args.gcfg != 1.0:
                    vg = model(x, tf, f, e, geo * 0)
                    v = vg + args.gcfg * (v - vg)
                x[:, :, 1:] = x[:, :, 1:] + (sig[i + 1] - sig[i]) * v[:, :, 1:]
            x[:, :, :1] = zc if zc is not None else x[:, :, :1]
            y = to_im(dec(x)[0])[1:]                                  # 8 new frames
            frames += list(y)
            gt = [teach(k + j) for j in range(1, 9)]
            psnr.append(np.mean([10 * np.log10(255 ** 2 / np.mean((a.astype(np.float32) - b) ** 2)) for a, b in zip(y, gt)]))
            print(f"chunk {c} frames {k + 1}-{k + 8}  PSNR vs teacher {psnr[-1]:.2f}  {(time.time() - t0) / (c + 1):.1f}s/chunk", flush=True)
            zc = enc(to_t(y[-1]))
    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.scene}_c{args.chase}_{args.start}_s{args.steps}_cfg{args.cfg}" + ("_geo" if args.geo else "") + (f"_g{args.gcfg}" if args.gcfg != 1.0 else "") + ("_mf" if args.meanflow else "")
    tmp = os.path.join("/tmp", "rv_" + tag); os.makedirs(tmp, exist_ok=True)
    for i, im in enumerate(frames):
        im = im.copy(); gt = teach(k0 + i).copy()
        cv2.putText(im, "student", (12, 30), 0, 0.8, (255, 255, 255), 2)
        cv2.putText(gt, "teacher (NuRec + Difix)", (12, 30), 0, 0.8, (255, 255, 255), 2)
        rows = [im, gt]
        wp = os.path.join(tdir, meta["frames"][k0 + i]["chase"][args.chase]["file"] + "_warp.jpg")
        if args.geo and os.path.exists(wp):
            w = cv2.imread(wp)
            cv2.putText(w, "input: warped 6-camera point cloud", (12, 30), 0, 0.8, (255, 255, 255), 2)
            rows.append(w)
        cv2.imwrite(os.path.join(tmp, f"{i:04d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    out = os.path.join(args.out, tag + ".mp4")
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-framerate", "10", "-i", os.path.join(tmp, "%04d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", out], check=True)
    subprocess.run(["rm", "-rf", tmp])
    print("mean PSNR %.2f  video %s" % (np.mean(psnr), out))


if __name__ == "__main__":
    main()
