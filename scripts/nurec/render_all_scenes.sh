#!/usr/bin/env bash
# Render the dataset for every downloaded NuRec scene: one renderer container per scene
# (the image's entrypoint shell-expands globs, so several scenes in one container break).
# Usage: bash scripts/nurec/render_all_scenes.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
WORK=${RV_WORK:-$PWD/work}
SCENES=${NUREC_SCENES:-$WORK/nurec_scenes}
DATA=${NUREC_DATA:-$WORK/nurec_data}

for S in $(find "$SCENES" -name "*.usdz" | sort); do
  id=$(basename "$S" .usdz)
  if [ -f "$DATA/clipgt-$id/meta.json" ]; then echo "skip $id"; continue; fi
  bash scripts/nurec/start_renderer.sh "$S"
  python scripts/nurec/render_dataset.py --out "$DATA" --stride 2 --nt 6
done
docker rm -f nurec_srv >/dev/null 2>&1 || true
du -sh "$DATA"
