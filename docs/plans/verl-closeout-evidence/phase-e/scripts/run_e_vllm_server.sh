#!/bin/bash
# Launch the E-stage vLLM-backed infer server on GPU1.
set -euo pipefail
export PATH=/home/cxr/miniconda3/envs/vllm/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_VISIBLE_DEVICES=1
exec /home/cxr/verl-closeout/venv/bin/python \
  /home/cxr/verl-closeout/smoke/stage_e_infer_server_vllm.py "$@"
