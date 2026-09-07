#!/usr/bin/env bash
# Reproducible base/SFT comparison on the frozen BigCodeBench Complete release.
set -euo pipefail

repo="${AGENT_REPO:?AGENT_REPO is required}"
candidate="${AGENT_SFT_ADAPTER_ROOT:?AGENT_SFT_ADAPTER_ROOT is required}"
tag="${AGENT_BCB_TAG:?AGENT_BCB_TAG is required}"
py="/home/f630/homePLUS/agentic/envs/agenticml/bin/python"
pi="/home/f630/homePLUS/agentic/run/node22/bin/pi"
dataset="$repo/experiments/bigcodebench/bigcodebench-complete-v0.1.1-manifest-v2.json"
log="$repo/sft-runs/bigcodebench-complete-${tag}.log"
exec >>"$log" 2>&1

test -f "$candidate/adapter_model.safetensors"
test -f "$dataset"

run_eval() {
  local output="$1" workspace="$2" revision="$3"
  PYTHONPATH="$repo" AGENT_MODEL_MAX_CALLS=32 "$py" "$repo/scripts/run_bigcodebench.py" \
    --dataset "$dataset" --workspace-root "$workspace" --output-root "$output" \
    --pi "$pi" --python "$py" --upstream-url http://127.0.0.1:8000/v1/chat/completions \
    --model qwen2.5-coder-14b-instruct --model-revision "$revision" \
    --image agent-data-plane-bigcodebench:v0.1.1 \
    --verifier-entrypoint "$repo/scripts/bigcodebench_container_verifier.py" \
    --prompt-mode complete --limit 1140 --timeout 900
}

episode_count() {
  local output="$1"
  if [[ ! -d "$output" ]]; then
    echo 0
    return
  fi
  find "$output" -type f -name episode.json | wc -l
}

require_complete() {
  local label="$1" output="$2"
  local count
  count="$(episode_count "$output")"
  if [[ "$count" -ne 1140 ]]; then
    printf '%s evaluation incomplete: expected 1140 episode artifacts, found %s\n' "$label" "$count" >&2
    exit 5
  fi
}

base_revision="qwen2.5-coder-14b-base"
candidate_revision="bigcodebench-qwen14b-lora/sha256:$(sha256sum "$candidate/adapter_model.safetensors" | cut -d ' ' -f1)"
echo "bigcodebench_complete_start=$(date -Is) tag=$tag"
base_output="$repo/bigcodebench-complete-${tag}-base"
base_workspace="$repo/bigcodebench-complete-workspaces-${tag}-base"
if [[ "$(episode_count "$base_output")" -lt 1140 ]]; then
  "$repo/scripts/serve_a6000_model.sh" base
  run_eval "$base_output" "$base_workspace" "$base_revision"
fi
require_complete "base" "$base_output"
export AGENT_SFT_ADAPTER_ROOT="$candidate" AGENT_SFT_MODEL_REVISION="$candidate_revision"
candidate_output="$repo/bigcodebench-complete-${tag}-candidate"
candidate_workspace="$repo/bigcodebench-complete-workspaces-${tag}-candidate"
if [[ "$(episode_count "$candidate_output")" -lt 1140 ]]; then
  "$repo/scripts/serve_a6000_model.sh" candidate
  run_eval "$candidate_output" "$candidate_workspace" "$candidate_revision"
fi
require_complete "candidate" "$candidate_output"
echo "bigcodebench_complete_finished=$(date -Is) tag=$tag"
