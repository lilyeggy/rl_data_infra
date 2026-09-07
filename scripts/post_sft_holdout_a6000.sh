#!/usr/bin/env bash
set -euo pipefail

repo="${AGENT_REPO:?AGENT_REPO is required}"
train_pid="${AGENT_TRAIN_PID:?AGENT_TRAIN_PID is required}"
py="/home/f630/homePLUS/agentic/envs/agenticml/bin/python"
log="${repo}/sft-runs/mbpp-post-sft-holdout-260827.log"
exec >>"$log" 2>&1

echo "post_sft_wait_start=$(date -Is) train_pid=$train_pid"
while kill -0 "$train_pid" 2>/dev/null; do sleep 30; done
candidate="${repo}/sft-runs/qwen14b-mbpp-train-260827"
test -f "$candidate/adapter_model.safetensors"

run_mbpp() {
  local output="$1" min_id="$2" max_id="$3" limit="$4"
  AGENT_MODEL_MAX_CALLS=32 PYTHONPATH="$repo" "$py" "$repo/scripts/run_mbpp.py" \
    --dataset "$repo/experiments/mbpp/mbpp-manifest.json" \
    --workspace-root "$repo/mbpp-workspaces-holdout" --output-root "$repo/$output" \
    --pi /home/f630/homePLUS/agentic/run/node22/bin/pi --python "$py" \
    --upstream-url http://127.0.0.1:8000/v1/chat/completions \
    --model qwen2.5-coder-14b-instruct --model-revision "$MODEL_REVISION" \
    --attempt 1 --min-task-id "$min_id" --max-task-id "$max_id" --limit "$limit" \
    --timeout 300
}

echo "base_eval_start=$(date -Is)"
"$repo/scripts/serve_a6000_model.sh" base
export MODEL_REVISION="qwen2.5-coder-14b-base"
run_mbpp "holdout-mbpp-base-260827" 511 600 90
PYTHONPATH="$repo" "$py" "$repo/scripts/run_mbpp.py" \
  --dataset "$repo/experiments/humaneval/humaneval-manifest.json" \
  --workspace-root "$repo/humaneval-workspaces" --output-root "$repo/holdout-humaneval-base-260827" \
  --pi /home/f630/homePLUS/agentic/run/node22/bin/pi --python "$py" \
  --upstream-url http://127.0.0.1:8000/v1/chat/completions --model qwen2.5-coder-14b-instruct \
  --model-revision "$MODEL_REVISION" --attempt 1 --limit 164 --timeout 300

MODEL_REVISION="mbpp-qwen14b-lora/sha256:$(sha256sum "$candidate/adapter_model.safetensors" | cut -d " " -f1)"
export AGENT_SFT_ADAPTER_ROOT="$candidate" AGENT_SFT_MODEL_REVISION="$MODEL_REVISION"
echo "candidate_eval_start=$(date -Is) revision=$MODEL_REVISION"
"$repo/scripts/serve_a6000_model.sh" candidate
run_mbpp "holdout-mbpp-candidate-260827" 511 600 90
PYTHONPATH="$repo" "$py" "$repo/scripts/run_mbpp.py" \
  --dataset "$repo/experiments/humaneval/humaneval-manifest.json" \
  --workspace-root "$repo/humaneval-workspaces" --output-root "$repo/holdout-humaneval-candidate-260827" \
  --pi /home/f630/homePLUS/agentic/run/node22/bin/pi --python "$py" \
  --upstream-url http://127.0.0.1:8000/v1/chat/completions --model qwen2.5-coder-14b-instruct \
  --model-revision "$MODEL_REVISION" --attempt 1 --limit 164 --timeout 300
echo "post_sft_holdout_finished=$(date -Is)"
