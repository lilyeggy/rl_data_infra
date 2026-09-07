#!/bin/bash
# ARCHIVED: run the retired self-evolution A/B prototype.
#   $1 = output dir   $2 = tasks (csv)   $3 = arms (csv)
cd /root/agentic-rl
export PYTHONPATH=/root/agentic-rl
OUT=${1:-/root/rivermind-data/self-evolve-v1}
TASKS=${2:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
ARMS=${3:-control,candidate}
python3 src/real_pi_v3.py \
  --output "$OUT" \
  --provider local-qwen --model qwen2.5-7b-instruct \
  --tasks "$TASKS" --arms "$ARMS" --timeout 300
