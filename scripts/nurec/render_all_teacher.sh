#!/usr/bin/env bash
# Render teacher sequences (render_teacher.py) for every downloaded NuRec scene, one renderer container per scene.
# Usage: bash scripts/nurec/render_all_teacher.sh [extra render_teacher.py args]
set -euo pipefail
cd "$(dirname "$0")/../.."
WORK=${RV_WORK:-$PWD/work}
SCENES=${NUREC_SCENES:-$WORK/nurec_scenes}
OUT=$WORK/teacher
PY=${PYTHON:-python}

for S in $(find "$SCENES" -name "*.usdz" | sort); do
  id=$(basename "$S" .usdz)
  if [ -f "$OUT/clipgt-$id/meta.json" ]; then echo "skip $id"; continue; fi
  bash scripts/nurec/start_renderer.sh "$S"
  $PY -W ignore scripts/nurec/render_teacher.py --out "$OUT" "$@"
done
docker rm -f nurec_srv >/dev/null 2>&1 || true
du -sh "$OUT"
