"""Clean the rendered teacher views (chase + orbit) with Difix-ref (one-step diffusion fixer, NVIDIA, non-commercial license).

    python scripts/nurec/clean_teacher.py [--teacher $RV_WORK/teacher] [--scenes id,...] [--fp16]

Writes <k>_fix.jpg next to each render; existing outputs are skipped. The reference view is an input camera at the
same timestamp (on the recorded trajectory, so NuRec is clean there): the front camera for chase views, the rig camera
facing most like the view for orbit views. --fp16 is ~2x faster with the same output (PSNR ~60 dB vs fp32).

Difix needs the diffusers 0.25 stack, which conflicts with the rest of the project, so it is loaded from a
separate folder:
    pip install --no-deps --target $DIFIX_DEPS diffusers==0.25.1 huggingface_hub==0.25.2 transformers==4.46.3 tokenizers==0.20.3 peft==0.13.2
    curl -o $DIFIX_CODE/pipeline_difix.py https://raw.githubusercontent.com/nv-tlabs/Difix3D/main/src/pipeline_difix.py
and the weights from https://huggingface.co/nvidia/difix_ref into $RV_MODELS/difix_ref.
"""
import argparse
import glob
import json
import os
import sys
import time

THIRD = os.path.expanduser(os.environ.get("RV_THIRD_PARTY", "~/realityview/third_party"))
sys.path.insert(0, os.environ.get("DIFIX_DEPS", os.path.join(THIRD, "difix_deps")))
sys.path.insert(0, os.environ.get("DIFIX_CODE", os.path.join(THIRD, "difix3d")))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from realityview import paths  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", default=os.path.join(paths.WORK, "teacher"))
ap.add_argument("--scenes", default="")
ap.add_argument("--model", default=paths.model("difix_ref"))
ap.add_argument("--ref_cam", default="F")
ap.add_argument("--timestep", type=int, default=199)
ap.add_argument("--fp16", action="store_true", help="half precision (faster)")
args = ap.parse_args()
DW, DH = 1024, 576   # Difix working resolution

sys.path.insert(0, args.model)    # custom unet / vae modules shipped with the weights
from pipeline_difix import DifixPipeline  # noqa: E402


def ref_cam(meta, c):
    """chase views use --ref_cam (front); orbit views the rig camera looking most like the view (side / rear views)."""
    if not c["file"].startswith("orbit"):
        return args.ref_cam
    fwd = np.array(c["c2rig"])[:3, 2]
    return max(meta["rig"], key=lambda r: np.array(r["c2rig"])[:3, 2] @ fwd)["name"]


pipe = DifixPipeline.from_pretrained(args.model, torch_dtype=torch.float16 if args.fp16 else torch.float32).to("cuda")
pipe.set_progress_bar_config(disable=True)

scene_dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(args.teacher, "*", "meta.json")))
if args.scenes:
    scene_dirs = [d for d in scene_dirs if any(k in d for k in args.scenes.split(","))]
for d in scene_dirs:
    meta = json.load(open(os.path.join(d, "meta.json")))
    jobs = [(fr["k"], c["file"], ref_cam(meta, c)) for fr in meta["frames"] for c in fr["chase"]
            if not os.path.exists(os.path.join(d, c["file"] + "_fix.jpg"))]
    t0 = time.time()
    for n, (k, f, rc) in enumerate(jobs):
        img = Image.open(os.path.join(d, f + ".jpg")).convert("RGB")
        ref = Image.open(os.path.join(d, f"inputs/{k:04d}_{rc}.jpg")).convert("RGB")
        with torch.no_grad():
            out = pipe("remove degradation", image=img.resize((DW, DH), Image.LANCZOS), ref_image=ref.resize((DW, DH), Image.LANCZOS),
                       num_inference_steps=1, timesteps=[args.timestep], guidance_scale=0.0, height=DH, width=DW).images[0]
        out.resize(img.size, Image.LANCZOS).save(os.path.join(d, f + "_fix.jpg"), quality=95)
        if n % 50 == 0:
            print(f"{os.path.basename(d)} {n}/{len(jobs)}  {(n + 1) / (time.time() - t0):.2f} img/s", flush=True)
    print("done", os.path.basename(d), len(jobs), "images", f"{time.time() - t0:.0f}s", flush=True)
