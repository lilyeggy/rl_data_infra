#!/bin/bash
# ARCHIVED: serve the retired lab SFT prototype.
LOG=/home/f630/homePLUS/agentic/run/server-14b-sft.log
echo "=== serve-14b-sft-lab $(date) ===" > "$LOG"
pkill -9 -f openai_server.py 2>/dev/null
sleep 3
cd /home/f630/homePLUS/agentic/code
source /home/f630/homePLUS/agentic/run/env.sh
export LOCAL_MODEL=/home/f630/homePLUS/agentic/models/qwen2.5-coder-14b-instruct
export LOCAL_ADAPTER=/home/f630/homePLUS/agentic/envs/qwen-14b-sft-lab
export PORT=8000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
setsid nohup python3 -u experiments/local_model/openai_server.py >> "$LOG" 2>&1 </dev/null &
sleep 3
echo "launched pid=$(pgrep -f openai_server.py | head -1)" >> "$LOG"
echo SERVE_SFT_LAB_DONE
