#!/usr/bin/env bash
# Optional, only for benchmarks/depth: DrivingForward + its Gaussian rasterizer.
# Needs nvcc matching your torch CUDA version (e.g. conda install -c nvidia/label/cuda-12.4.1 cuda-nvcc cuda-cudart-dev cuda-cccl)
# and the released weights unpacked into $RV_MODELS/DrivingForward/weights_SF (see the DrivingForward README).
set -euo pipefail
M=${RV_MODELS:-$HOME/models}
mkdir -p "$M" && cd "$M"
[ -d DrivingForward ] || git clone --depth 1 https://github.com/fangzhou2000/DrivingForward.git
cd DrivingForward/models/gaussian/gaussian-splatting
pip install -q --no-build-isolation submodules/diff-gaussian-rasterization
pip install -q e3nn
python -c "import diff_gaussian_rasterization; print('diff_gaussian_rasterization OK')"
