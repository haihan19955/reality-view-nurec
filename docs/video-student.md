# Video student (v0.2): reproduction guide

This guide trains the free-view video student from scratch and runs the live viewer. The numbers below were measured
on one RTX 4090 (24 GB) with 12 NuRec scenes: 10 for training, 2 held out (`e8cf41e8`, `ff28f961`).

## 1. Models and third-party code

Everything goes under `$RV_MODELS` (default `~/models`):

| Folder | Source | Used by |
|---|---|---|
| `wan2.1-1.3b/` | [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) | student backbone + VAE |
| `DA3METRIC-LARGE/` | `bash scripts/setup/install_da3.sh` | metric depth for the point cloud |
| `taehv/` (`taehv.py`, `taew2_1.pth`) | [madebyollin/taehv](https://github.com/madebyollin/taehv) | fast VAE in the live viewer |
| `difix_ref/` | [nvidia/difix_ref](https://huggingface.co/nvidia/difix_ref) (non-commercial) | teacher clean-up |
| `dinov2/` + `dinov2_vitl14_pretrain.pth` | optional, otherwise downloaded through `torch.hub` | camera features |
| `dinov3-vitl16/` | optional, gated: [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | `--feat dino3` |

Difix needs an older diffusers stack, so it is loaded from its own folder. See the docstring of
`scripts/nurec/clean_teacher.py` (`RV_THIRD_PARTY`, `DIFIX_DEPS`, `DIFIX_CODE`).

## 2. Teacher data

NuRec scenes (`.usdz`) go into `$NUREC_SCENES`, and the gRPC stubs are generated with `scripts/nurec/setup_client.sh`.
The renderer is a Docker container (`carlasimulator/nvidia-nurec-grpc:0.2.0`) that can load one scene at a time;
the `render_all_*` scripts start one container per scene.

| Step | Command | Output | Time (12 scenes) |
|---|---|---|---|
| Chase views + 6 rig inputs + lidar depth | `bash scripts/nurec/render_all_teacher.sh` | `$RV_WORK/teacher/<scene>/` | not timed |
| 6 orbit tracks per scene | `bash scripts/nurec/render_all_orbit.sh` | `orbit<i>/`, `meta.json` updated | ~12 min |
| Difix clean-up | `python scripts/nurec/clean_teacher.py --fp16` | `<k>_fix.jpg` | ~2 h |

Orbit tracks (`render_orbit.py`) are random but smooth (keyframes every 1.5 s, smoothstep) and seeded per scene.
Their range is limited to what NuRec renders cleanly: pitch 8–35°, distance 6–12 m, camera at most 5 m high. Higher
views smear, and cameras inside trees happen. Check `work/outputs/` contact sheets before training.

## 3. Training cache

| Step | Command | Writes to `$RV_WORK/student_cache/<scene>/` | Time |
|---|---|---|---|
| Latents, features, ego motion, camera poses | `python scripts/student/prepare.py [--dino3]` | `lat/c*.pt`, `dino.npy`, `ego.npy`, `cams.npy`, `index.json` | ~25 min |
| Point-cloud warps | `python scripts/student/prepare_warp.py` | `lat/w*.pt`, `lat/m*.pt`, `warp_done` | ~35 min |

Both are incremental: new cameras (for example new orbit tracks) are added without redoing the rest. Chunks are 9
frames (1 context + 8 new), every 2 frames on chase cameras and every 4 frames on orbit tracks: 4617 training chunks.
Disk: teacher ~9 GB, cache ~10 GB.

## 4. Training

```bash
# v1: chase-only student without geometry
python scripts/student/train.py --out $RV_WORK/checkpoints/student
# chase-only student with point-cloud conditioning (geo v3), from v1
python scripts/student/train.py --geo --init $RV_WORK/checkpoints/student/last.pt \
    --p_corrupt 0.5 --obs_w 2 --p_drop_geo 0.1 --out $RV_WORK/checkpoints/student_geo3

# free view: chase + orbit chunks, camera pose in the ego tokens (~4.3 s/step, 6000 steps ≈ 7 h)
python scripts/student/train.py --geo --free_view --init $RV_WORK/checkpoints/student_geo3/last.pt \
    --p_corrupt 0.5 --obs_w 2 --p_drop_geo 0.1 --steps 6000 --out $RV_WORK/checkpoints/student_free
```

- `--free_view` widens the ego-token input layer from 34 to 50. With `--init` from a chase-only checkpoint, the 32
  motion columns are copied and the camera columns start at zero.
- Validation is reported per camera kind and noise level (`VAL step N chase0.2:… orbit0.8:…`) in `log.txt`.
- `--stop_at N` saves and stops early on the full learning-rate schedule (for A/B runs); `--start_step N` continues
  such a run from its `last.pt`.
- Always run `--smoke` first after code changes: it trains 3 steps, validates once and saves.

## 5. Distillation

```bash
python scripts/student/distill.py --teacher $RV_WORK/checkpoints/student_free/last.pt --steps 2500 \
    --out $RV_WORK/checkpoints/student_mf_free         # ~7 s/step, ≈ 5 h
```

The teacher checkpoint decides free view vs chase-only and the feature type (`feat.txt` next to it). Validation
reports the x0 error of 1-step / 2-step sampling against the 8-step teacher. The free-view run reached 0.0149 / 0.0067.

## 6. Evaluation and viewer

```bash
python scripts/student/sample.py --ckpt $RV_WORK/checkpoints/student_mf_free/last.pt --geo --meanflow --steps 1 \
    --scene e8cf41e8 --chase 0                         # rollout video vs teacher, PSNR per chunk
python benchmarks/student/evaluate.py ...              # chase-view metrics table (see its docstring)
bash viewer/run_live.sh [--scene e8cf41e8] [--ckpt …]  # http://localhost:8771
```

Viewer controls: drag = orbit / tilt, wheel = distance, double-click = back to chase, presets below the video.
The status line shows the generation speed and the per-chunk timing (warp / encode / generate / decode).

When measuring latency, keep the page visible: a hidden tab throttles `requestAnimationFrame` and makes the viewer
look several times slower than it is.

## 7. Known issues

- Side and front views of held-out scenes are blurry; the training curves were still improving at 6000 steps.
- DINOv3 must run in bf16 (fp16 overflows to NaN); its tokens are scaled ×4.8 to match DINOv2's statistics.
- Large drag rotations stretch near objects in the reprojection preview until generated frames arrive.
