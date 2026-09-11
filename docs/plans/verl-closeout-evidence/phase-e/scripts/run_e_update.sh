#!/bin/bash
# Stage E: dual-GPU native GRPO update for one round.
set -euo pipefail
export PATH=/home/cxr/miniconda3/envs/vllm/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0
exec /home/cxr/verl-closeout/venv/bin/python -m torch.distributed.run \
  --nproc_per_node=2 --master_port=29571 \
  /home/cxr/verl-closeout/smoke/stage_e_grpo_update.py "$@"
