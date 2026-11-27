# Changelog

All notable changes are listed here. Versions follow [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-27

Video student with free viewpoint and a real-time drag viewer.

### Added
- NuRec teacher pipeline for video: chase-camera sequences (`render_teacher.py`), random smooth orbit tracks
  (`render_orbit.py`) and Difix-ref clean-up with per-view reference cameras and fp16 (`clean_teacher.py`).
- Video student (`realityview/student.py`): Wan2.1-1.3B + LoRA, 8 frames per chunk, conditioned on the previous
  frame, DINOv2 camera tokens, ego motion and, with `--free_view`, the target camera pose.
- Pixel-aligned point-cloud conditioning (`realityview/warp.py`, `prepare_warp.py`): DA3 depth scaled to lidar,
  accumulated over 2 s and splatted into the target camera.
- Training (`scripts/student/train.py`) with context corruption, observed-pixel loss weighting, A/B helpers
  (`--stop_at`, `--start_step`) and per-camera-kind validation; MeanFlow distillation to 1 step (`distill.py`).
- Optional DINOv3 features (`prepare.py --dino3`, `--feat dino3`).
- Real-time viewer (`viewer/live.py`, `live.html`): TAEHV decoding, mouse-drag orbit, instant reprojection preview
  while dragging.
- Benchmarks: NuRec extrapolation quality, video student rollouts.
- Bilingual README, reproduction guide, contributing guide, CI (ruff, compile, shell syntax).

### Changed
- Point-cloud splatting runs in one pass over all layers: 3.5× faster with identical output.

## [0.1.0] - 2026-11-24

Frame model: DA3 metric depth + one-step MeanFlow inpainting on NuRec scenes, interactive viewer, depth benchmark.

[0.2.0]: https://github.com/haihan19955/reality-view-nurec/compare/dc7adf0...v0.2.0
[0.1.0]: https://github.com/haihan19955/reality-view-nurec/commit/dc7adf0
