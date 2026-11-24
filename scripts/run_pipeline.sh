#!/usr/bin/env bash
# Preprocess -> smoke test -> train -> evaluate every checkpoint. Assumes the dataset is already rendered.
# Usage: bash scripts/run_pipeline.sh [train steps, default 8000]
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${RV_WORK:-$PWD/work}
STEPS=${1:-8000}
mkdir -p "$WORK"
[ -f "${RV_CACHE:-$WORK/cache}/index.json" ] || python scripts/preprocess.py 2>&1 | tee "$WORK/preprocess.log"
python scripts/train.py --smoke --out "$WORK/smoke" && rm -rf "$WORK/smoke"
python scripts/train.py --steps "$STEPS" 2>&1 | tee "$WORK/train.log"
python scripts/evaluate.py "$WORK"/checkpoints/step*.pt
