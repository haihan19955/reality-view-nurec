#!/usr/bin/env bash
# Generate the NuRec gRPC python stubs from the .proto files shipped with CARLA's NuRec example
# (PythonAPI/examples/nvidia/nurec in the CARLA source tree).
# Usage: bash scripts/nurec/setup_client.sh /path/to/carla/PythonAPI/examples/nvidia/nurec
set -euo pipefail
SRC=$(realpath "$1")
cd "$(dirname "$0")/../.."
OUT=${NUREC_PROTO:-${RV_WORK:-$PWD/work}/nurec_proto}
mkdir -p "$OUT"
pip install -q grpcio grpcio-tools
(cd "$SRC" && python -m grpc_tools.protoc -I . --python_out="$OUT" --grpc_python_out="$OUT" \
  nre/grpc/protos/common.proto nre/grpc/protos/sensorsim.proto)
touch "$OUT/nre/__init__.py" "$OUT/nre/grpc/__init__.py" "$OUT/nre/grpc/protos/__init__.py"
echo "stubs in $OUT"
