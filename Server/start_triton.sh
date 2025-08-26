docker run --rm --gpus=all \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v ~/Inference/model-repository:/models \
  tritonserver:latest \
  tritonserver --model-repository=/models