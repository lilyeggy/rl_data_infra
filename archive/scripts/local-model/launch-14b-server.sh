#!/bin/bash
# ARCHIVED: launch the retired local Coder-14B experiment server.
# 14B bf16 ~28G weights; on the 40G A100 this leaves ~10G headroom for KV.
# NOTE: serve-only. Training (SFT/RAFT) must run AFTER stopping this server,
# because 14B train+serve simultaneously exceeds 40G.
pkill -9 -f openai_server.py 2>/dev/null
sleep 2
cd /root/agentic-rl
setsid nohup env PYTHONPATH=/root/agentic-rl \
  LOCAL_MODEL=/root/rivermind-data/models/qwen2.5-coder-14b-instruct \
  LOCAL_ADAPTER="${LOCAL_ADAPTER:-}" \
  PORT=8000 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 experiments/local_model/openai_server.py \
  >/root/rivermind-data/server-14b.log 2>&1 </dev/null &
sleep 3
pgrep -f openai_server.py >/dev/null && echo "SERVER_14B_LAUNCHING" || echo "SERVER_14B_FAIL"
