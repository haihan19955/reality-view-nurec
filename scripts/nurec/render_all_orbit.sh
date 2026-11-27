#!/usr/bin/env bash
# Add free-viewpoint orbit tracks (render_orbit.py) to every scene that already has chase teacher renders.
# Usage: bash scripts/nurec/render_all_orbit.sh [extra render_orbit.py args]
set -euo pipefail
cd "$(dirname "$0")/../.."
WORK=${RV_WORK:-$PWD/work}
SCENES=${NUREC_SCENES:-$WORK/nurec_scenes}
OUT=$WORK/teacher
PY=${PYTHON:-python}

for S in $(find "$SCENES" -name "*.usdz" | sort); do
  id=$(basename "$S" .usdz)
  M="$OUT/clipgt-$id/meta.json"
  [ -f "$M" ] || continue
  if grep -q '"kind": "orbit"' "$M"; then echo "skip $id"; continue; fi
  bash scripts/nurec/start_renderer.sh "$S"
  $PY -W ignore scripts/nurec/render_orbit.py --teacher "$OUT" "$@"
done
docker rm -f nurec_srv >/dev/null 2>&1 || true
du -sh "$OUT"
