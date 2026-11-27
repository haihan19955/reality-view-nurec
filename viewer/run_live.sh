#!/usr/bin/env bash
# Start the live Reality View viewer (viewer/live.py) on http://localhost:8771.
#   bash viewer/run_live.sh [--scene e8cf41e8] [--ckpt <distilled student>] [--port 8771] [--enc full --dec full]
# RV_VENV: optional python venv to activate (torch / diffusers / peft / depth_anything_3)
# RV_WORK: working dir with teacher/, student_cache/, checkpoints/ (default: <repo>/work, see realityview/paths.py)
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -n "$RV_VENV" ]; then source "$RV_VENV/bin/activate"; fi
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
exec python -u -W ignore viewer/live.py "$@"
