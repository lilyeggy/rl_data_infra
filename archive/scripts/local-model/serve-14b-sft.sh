#!/bin/bash
# ARCHIVED: serve the retired 14B SFT prototype.
LOG=/root/rivermind-data/server-14b-sft.log
echo "=== serve-14b-sft $(date) ===" > "$LOG"
pkill -9 -f openai_server.py 2>/dev/null
sleep 5
cd /root/agentic-rl
setsid nohup env PYTHONPATH=/root/agentic-rl \
  LOCAL_MODEL=/root/rivermind-data/models/qwen2.5-coder-14b-instruct \
  LOCAL_ADAPTER=/root/rivermind-data/models/qwen-14b-sft-v1 \
  PORT=8000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 -u experiments/local_model/openai_server.py >> "$LOG" 2>&1 </dev/null &
sleep 3
echo "launched pid=$(pgrep -f openai_server.py | head -1)" >> "$LOG"
echo "SERVE_SFT_DONE"
