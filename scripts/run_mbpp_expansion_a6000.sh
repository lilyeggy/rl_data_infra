#!/usr/bin/env bash
set -euo pipefail

repo="${AGENT_REPO:?AGENT_REPO is required}"
py="/home/f630/homePLUS/agentic/envs/agenticml/bin/python"
log="$repo/sft-runs/mbpp-expansion-301-500-260830.log"
exec >>"$log" 2>&1

echo "mbpp_expansion_start=$(date -Is) range=301-500 purpose=raw-data-quality"
revision="qwen2.5-coder-14b-base"
"$repo/scripts/serve_a6000_model.sh" base
AGENT_MODEL_MAX_CALLS=32 PYTHONPATH="$repo" "$py" "$repo/scripts/run_mbpp.py" \
  --dataset "$repo/experiments/mbpp/mbpp-manifest.json" \
  --workspace-root "$repo/mbpp-workspaces-expansion-260830" \
  --output-root "$repo/mbpp-expansion-301-500-260830" \
  --pi /home/f630/homePLUS/agentic/run/node22/bin/pi --python "$py" \
  --upstream-url http://127.0.0.1:8000/v1/chat/completions \
  --model qwen2.5-coder-14b-instruct --model-revision "$revision" \
  --attempt 1 --min-task-id 301 --max-task-id 500 --limit 200 --timeout 600
echo "mbpp_expansion_finished=$(date -Is)"
