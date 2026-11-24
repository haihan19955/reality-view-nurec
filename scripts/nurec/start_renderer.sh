#!/usr/bin/env bash
# Start the NuRec gRPC renderer for one scene on localhost:46435.
# Usage: bash scripts/nurec/start_renderer.sh /path/to/scene.usdz
# --test-scenes-are-valid is required with this image, otherwise rendering fails with a GUTRenderer CUDA-resource error.
set -euo pipefail
S=$(realpath "$1"); D=$(dirname "$S")
docker rm -f nurec_srv >/dev/null 2>&1 || true
docker run -d --name nurec_srv --gpus 'all,"capabilities=compute,video,utility"' --net=host -v "$D:$D:ro" \
  carlasimulator/nvidia-nurec-grpc:0.2.0 --artifact-glob "$S" --port=46435 --host=0.0.0.0 --test-scenes-are-valid >/dev/null
for i in $(seq 1 60); do
  if docker logs nurec_srv 2>&1 | grep -q -i -E "serving on|traceback|error:"; then break; fi
  sleep 3
done
docker logs nurec_srv 2>&1 | grep -i -E "serving on|traceback|error:" | tail -3
