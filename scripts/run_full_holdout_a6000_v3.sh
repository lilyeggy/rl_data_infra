#!/usr/bin/env bash
set -euo pipefail

repo="${AGENT_REPO:?AGENT_REPO is required}"
py="/home/f630/homePLUS/agentic/envs/agenticml/bin/python"
candidate="$repo/sft-runs/qwen14b-mbpp-train-260827"
log="$repo/sft-runs/mbpp-post-sft-holdout-v3-260829.log"
exec >>"$log" 2>&1

run_eval() {
  local dataset="$1" output="$2" workspace="$3" revision="$4" limit="$5"
  AGENT_MODEL_MAX_CALLS=32 PYTHONPATH="$repo" "$py" "$repo/scripts/run_mbpp.py" \
    --dataset "$dataset" --workspace-root "$workspace" --output-root "$output" \
    --pi /home/f630/homePLUS/agentic/run/node22/bin/pi --python "$py" \
    --upstream-url http://127.0.0.1:8000/v1/chat/completions \
    --model qwen2.5-coder-14b-instruct --model-revision "$revision" \
    --attempt 1 --limit "$limit" --timeout 600
}

mbpp="$repo/experiments/mbpp/mbpp-manifest.json"
humaneval="$repo/experiments/humaneval/humaneval-manifest.json"
base_revision="qwen2.5-coder-14b-base"
candidate_revision="mbpp-qwen14b-lora/sha256:$(sha256sum "$candidate/adapter_model.safetensors" | cut -d ' ' -f1)"

echo "full_holdout_v3_start=$(date -Is)"
"$repo/scripts/serve_a6000_model.sh" base
run_eval "$mbpp" "$repo/holdout-v3-mbpp-base-260829" "$repo/mbpp-workspaces-v3-base" "$base_revision" 90
run_eval "$humaneval" "$repo/holdout-v3-humaneval-base-260829" "$repo/humaneval-workspaces-v3-base" "$base_revision" 164

export AGENT_SFT_ADAPTER_ROOT="$candidate" AGENT_SFT_MODEL_REVISION="$candidate_revision"
"$repo/scripts/serve_a6000_model.sh" candidate
run_eval "$mbpp" "$repo/holdout-v3-mbpp-candidate-260829" "$repo/mbpp-workspaces-v3-candidate" "$candidate_revision" 90
run_eval "$humaneval" "$repo/holdout-v3-humaneval-candidate-260829" "$repo/humaneval-workspaces-v3-candidate" "$candidate_revision" 164
echo "full_holdout_v3_finished=$(date -Is)"
