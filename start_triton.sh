#!/usr/bin/env bash
set -euo pipefail

# Resolve the directory of this script to mount the correct model repo
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_REPO="$SCRIPT_DIR/model-repository"

if [[ ! -d "$MODEL_REPO" ]]; then
  echo "Model repository not found at: $MODEL_REPO" >&2
  exit 1
fi

# Should take about 5-10 minutes to be able to perform inference 
docker run --rm -it --name triton --gpus=all \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v "$MODEL_REPO:/models" \
  nvcr.io/nvidia/tritonserver:25.07-py3 \
  tritonserver --model-repository=/models --log-verbose=1
