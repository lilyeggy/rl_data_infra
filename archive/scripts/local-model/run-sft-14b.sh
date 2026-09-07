#!/bin/bash
# ARCHIVED: run the retired 14B LoRA SFT prototype.
LOG=/root/rivermind-data/sft-14b.log
echo "=== SFT launch $(date) ===" > "$LOG"
pkill -9 -f openai_server.py 2>/dev/null
pkill -9 -f sft_14b.py 2>/dev/null
sleep 5
echo "GPU before: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)" >> "$LOG"
cd /root/agentic-rl
export PYTHONPATH=/root/agentic-rl
export LOCAL_MODEL=/root/rivermind-data/models/qwen2.5-coder-14b-instruct
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
setsid nohup python3 -u experiments/local_model/sft_14b.py \
  --data /root/rivermind-data/14b-sft-data/sft_tokens.jsonl \
  --output /root/rivermind-data/models/qwen-14b-sft-v1 \
  --epochs 3 --lr 1e-4 \
  >> "$LOG" 2>&1 </dev/null &
sleep 3
echo "launched pid=$(pgrep -f sft_14b.py | head -1)" >> "$LOG"
echo "SFT_SCRIPT_DONE"
