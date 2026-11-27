"""Precompute training tensors for the video student from the cleaned teacher sequences. Incremental: re-running it
after render_orbit.py only adds what is missing.

    python scripts/student/prepare.py [--teacher $RV_WORK/teacher] [--out $RV_WORK/student_cache] [--dino3]

Per scene it writes
    dino.npy          [N, 6, 112, 1024] fp16   DINOv2-L tokens of the 6 rig cameras for every frame
    dino3.npy         same with DINOv3-L/16 (--dino3)
    ego.npy           [N, 4, 4]                ego2world per frame
    cams.npy          [C, N, 4, 4]             c2rig of every teacher camera (chase / orbit track) per frame
    lat/c<i>_<k>.pt   [16, 3, 60, 104] bf16    normalized Wan VAE latent of camera i, frames k..k+8 (Difix-cleaned)
    index.json        chunk list, camera configs
Chase chunks (stride 2) are kept only if all 9 frames lie on the driven path (off-path chase renders are extrapolated
and poor). Orbit tracks are rig-fixed and use stride 4.
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

CAMS = ["F", "FL", "FR", "BL", "BR", "B"]
CHUNK = 9          # 1 context frame + 8 new frames -> 3 Wan latent frames
STRIDE = {"chase": 2, "orbit": 4}

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", default=os.path.join(paths.WORK, "teacher"))
ap.add_argument("--out", default=os.path.join(paths.WORK, "student_cache"))
ap.add_argument("--wan", default=paths.model("wan2.1-1.3b"))
ap.add_argument("--dino3", action="store_true", help="also compute DINOv3 tokens (dino3.npy)")
ap.add_argument("--no_latents", action="store_true")
args = ap.parse_args()


def load_vae(path, device="cuda"):
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=torch.bfloat16).to(device).eval()
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    return vae, mean, std


def features(d, frames, load, tokens):
    feats = np.zeros((len(frames), 6, 112, 1024), np.float16)
    net = load()
    for i, f in enumerate(frames):
        imgs = [cv2.imread(os.path.join(d, f"inputs/{f['k']:04d}_{c}.jpg"))[..., ::-1].copy() for c in CAMS]
        feats[i] = tokens(net, imgs).half().cpu().numpy()
    del net
    torch.cuda.empty_cache()
    return feats


def main():
    from realityview.features import dino3_tokens, dino_tokens, load_dinov2, load_dinov3
    vae = None
    for d in sorted(os.path.dirname(p) for p in glob.glob(os.path.join(args.teacher, "*", "meta.json"))):
        meta = json.load(open(os.path.join(d, "meta.json")))
        o = os.path.join(args.out, meta["scene"])
        os.makedirs(os.path.join(o, "lat"), exist_ok=True)
        t0 = time.time()
        frames = meta["frames"]
        np.save(os.path.join(o, "ego.npy"), np.array([f["ego2world"] for f in frames], np.float32))
        np.save(os.path.join(o, "cams.npy"), np.array([[c["c2rig"] for c in f["chase"]] for f in frames], np.float32).transpose(1, 0, 2, 3))
        if not os.path.exists(os.path.join(o, "dino.npy")):
            np.save(os.path.join(o, "dino.npy"), features(d, frames, load_dinov2, dino_tokens))
        if args.dino3 and not os.path.exists(os.path.join(o, "dino3.npy")):
            np.save(os.path.join(o, "dino3.npy"), features(d, frames, load_dinov3, dino3_tokens))
        ip = os.path.join(o, "index.json")
        old = json.load(open(ip)) if os.path.exists(ip) else {"chunks": []}
        chunks, have = list(old["chunks"]), {ci for ci, _ in old["chunks"]}
        new = 0
        for ci, cfg in enumerate(meta["chase"]):
            if ci in have or args.no_latents:
                continue
            kind = cfg.get("kind", "chase")
            ok = [f["chase"][ci]["on_path"] for f in frames]
            for s in range(0, len(frames) - CHUNK + 1, STRIDE[kind]):
                if not all(ok[s:s + CHUNK]):
                    continue
                if vae is None:
                    vae, lmean, lstd = load_vae(args.wan)
                fr = [cv2.imread(os.path.join(d, frames[s + j]["chase"][ci]["file"] + "_fix.jpg"))[..., ::-1] for j in range(CHUNK)]
                x = torch.from_numpy(np.stack(fr)).cuda().permute(3, 0, 1, 2)[None].float().div(127.5).sub(1)
                with torch.no_grad():
                    z = vae.encode(x.bfloat16()).latent_dist.mode()
                z = ((z.float() - lmean) / lstd)[0].bfloat16().cpu()
                torch.save(z.clone(), os.path.join(o, "lat", f"c{ci}_{s:04d}.pt"))
                chunks.append([ci, s]); new += 1
        json.dump({"scene": meta["scene"], "n": len(frames), "chase": meta["chase"], "chunks": chunks, "chunk": CHUNK},
                  open(ip, "w"))
        print("done", meta["scene"], len(frames), "frames", len(chunks), "chunks", f"(+{new})", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
