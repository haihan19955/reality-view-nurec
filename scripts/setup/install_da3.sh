#!/usr/bin/env bash
# Install Depth Anything 3 and download DA3METRIC-LARGE (used by the pipeline) and DA3-LARGE-1.1 (depth benchmark only).
set -euo pipefail
M=${RV_MODELS:-$HOME/models}
mkdir -p "$M" && cd "$M"
[ -d Depth-Anything-3 ] || git clone --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3.git
pip install -q --no-deps -e Depth-Anything-3
pip install -q omegaconf plyfile typer jaxtyping addict moviepy==1.0.3 pycolmap trimesh evo
for repo in DA3METRIC-LARGE DA3-LARGE-1.1; do
  huggingface-cli download "depth-anything/$repo" config.json model.safetensors --local-dir "$M/$repo"
done
python -c "from depth_anything_3.api import DepthAnything3; print('Depth Anything 3 OK')"
