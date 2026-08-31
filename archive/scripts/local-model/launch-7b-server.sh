#!/bin/bash
# ARCHIVED: launch the retired local base-7B experiment server.
pkill -9 -f openai_server.py 2>/dev/null
sleep 1
cd /root/agentic-rl
setsid nohup env PYTHONPATH=/root/agentic-rl \
  LOCAL_MODEL=/root/rivermind-data/models/qwen2.5-7b-instruct \
  PORT=8000 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 experiments/local_model/openai_server.py \
  >/root/rivermind-data/self-evolve-server.log 2>&1 </dev/null &
sleep 2
pgrep -f openai_server.py >/dev/null && echo "SERVER_UP" || echo "SERVER_FAIL"
