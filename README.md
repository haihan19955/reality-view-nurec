# Reality View (NuRec)

Free-viewpoint ego surround view from six cameras. Any orbit view around the car (chase, top-down, side, front) is rendered in real time: where the cameras saw, real pixels are reprojected with monocular metric depth; where no camera saw (under and right next to the car), a one-step MeanFlow diffusion model fills the gap; a stylised ego car is drawn on top.

Trained on NVIDIA NuRec neural reconstructions of real drives.

<p align="center">
  <img src="docs/demo.gif" width="720" alt="Viewer demo: generated view (left) vs geometry only (right)">
  <br><sub>Left: generated view (real pixels + MeanFlow inpainting + ego car). Right: geometry only, black = seen by no camera. Held-out scene, unseen in training. <a href="docs/demo.mp4">Full recording with the six input cameras</a>.</sub>
</p>

## Method

```
                 ┌──────────── once per frame ────────────┐   ┌────────────── per view (~44 ms) ──────────────┐
6 surround RGB ─▶ DA3METRIC depth + sky mask ─▶ point cloud ─▶ z-buffer splat into orbit camera ─┐
              └─▶ DINOv2-L tokens (PCA 256) ─▶ camera context ───────────────────────────────────┤
                                                                MeanFlow DiT, 1 step (VAE latent) ◀┘
                                                                  ─▶ keep real pixels, fill holes ─▶ draw ego car
```

| Component | Details |
|---|---|
| Depth | Depth Anything 3 metric (DA3METRIC-LARGE), sky set to zero; ~100 ms per frame, computed once per frame |
| Generator | 77M-parameter DiT trained with the MeanFlow objective, one-step sampling; conditioned on camera tokens, target-camera Plücker rays and the latent of the geometry splat |
| Training data | Rendered from NuRec scenes: 6 pinhole inputs + 6 random chase/orbit target views per frame; the 3–5 m around the ego, where NuRec has artifacts, is excluded from the loss |
| Compositing | Real splatted pixels where covered, generated pixels in the holes, feathered seam |

## Results

PSNR outside the ego zone (384×208) on 2 held-out scenes:

| Method | PSNR (dB) |
|---|---|
| Geometry only (holes black) | 14.06 |
| MeanFlow 1-step, no geometry | 16.80 |
| MeanFlow 1-step + geometry (7.5k steps) | **18.27** |
| Composite: real pixels + MeanFlow in the holes | 18.25 (visibly sharper) |

Depth against rendered lidar (6 scenes, 144 views):

| Method | AbsRel ↓ | δ<1.25 ↑ | Novel-view PSNR | Time |
|---|---|---|---|---|
| Depth Anything V2 metric | 0.860 | 0.126 | 19.12 | 666 ms |
| **DA3METRIC** (used here) | 0.255 | 0.623 | 19.94 | 102 ms |
| DA3 with known poses, 12 frames | 0.234 | 0.665 | 20.31 | 252 ms |
| DrivingForward (zero-shot) | 0.361 | 0.427 | 19.29 | 312 ms |

## Layout

```
realityview/            core library (pip install -e .)
  geometry.py           lifting, z-buffer splat, hole filling, compositing, Plücker rays, ego zone, orbit cameras
  model.py              MeanFlow DiT and one-step sampler
  depth.py              DA3 metric depth
  features.py           DINOv2 camera tokens
  latents.py            SD VAE encode / decode
  dataset.py            reading the rendered NuRec dataset
  ego_car.py            low-poly ego car overlay
  pipeline.py           RealityView: prepare(6 images) -> render(any camera)
  paths.py              all paths, overridable by environment variables
scripts/
  nurec/                download list, renderer container, dataset rendering, dataset check
  setup/                install Depth Anything 3 / DrivingForward
  preprocess.py         dataset -> training cache
  train.py              training
  evaluate.py           checkpoint evaluation
  run_pipeline.sh       preprocess -> train -> evaluate
viewer/                 interactive viewer (server.py + index.html)
benchmarks/depth/       depth comparison (DA2 / DA3 / DrivingForward)
tools/                  cache inspection, ego car preview
```

## Usage

Requirements: Linux or WSL2, a CUDA GPU (developed on a single Ada-generation card, ~0.25 s per training step), Docker with the NVIDIA Container Toolkit.

```bash
pip install -e ".[nurec]"
bash scripts/setup/install_da3.sh                       # Depth Anything 3 + weights -> $RV_MODELS (default ~/models)

# 1. Data: accept the terms of nvidia/PhysicalAI-Autonomous-Vehicles-NuRec on Hugging Face and log in
python scripts/nurec/make_download_list.py 16           # links -> work/nurec_links.tsv; download into work/nurec_scenes/
bash scripts/nurec/setup_client.sh <carla>/PythonAPI/examples/nvidia/nurec   # generate the gRPC client
bash scripts/nurec/render_all_scenes.sh                 # one renderer container per scene -> work/nurec_data/
python scripts/nurec/check_dataset.py

# 2. Train and evaluate
bash scripts/run_pipeline.sh 8000                       # preprocess -> train 8k steps -> evaluate every checkpoint

# 3. Viewer
python viewer/server.py --ckpt work/checkpoints/step07.5k.pt    # http://localhost:8770
```

All data, caches, weights and outputs live in `work/` (gitignored). Paths can be overridden with `RV_WORK`, `RV_MODELS`, `RV_CACHE`, `NUREC_SCENES`, `NUREC_DATA` and `NUREC_PROTO`.

NuRec notes:
- the renderer container needs `--test-scenes-are-valid`, otherwise rendering fails with a GUTRenderer CUDA-resource error;
- one container can load only one scene;
- scenes without rear cameras are skipped automatically (5 of 17 in our download).

The v0.1 results above were trained from a checkpoint pretrained on CARLA (`--init`); the CARLA pretraining code is not part of this repository yet.

## Roadmap

- **v0.2 quality**: pixel-accurate feathered seams (currently an 8 px staircase), more stable training (diverges after ~10k steps), a more realistic ego car
- **v0.3 data**: more NuRec scenes, recover the scenes without rear cameras, temporal consistency across frames
- **v1.0 real time**: TensorRT, higher resolution

## Notes

This repository contains code only. NuRec scene data, the rendered dataset and trained weights are not distributed.

Built on [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), [MeanFlow](https://arxiv.org/abs/2505.13447), [DrivingForward](https://github.com/fangzhou2000/DrivingForward), NVIDIA NuRec, DINOv2 and the Stable Diffusion VAE; each is subject to its own license.

Not affiliated with Tesla.
