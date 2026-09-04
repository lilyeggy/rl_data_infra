#!/usr/bin/env bash

# Run a leakage-free base-versus-adapter evaluation on the A6000 host.
#
# The SFT package uses MBPP task ids 601--900.  This runner intentionally
# evaluates only MBPP 511--600 plus the separate HumanEval manifest.
set -euo pipefail

repo="${AGENT_REPO:?AGENT_REPO is required}"
candidate="${AGENT_SFT_ADAPTER_ROOT:?AGENT_SFT_ADAPTER_ROOT is required}"
run_tag="${AGENT_HOLDOUT_TAG:?AGENT_HOLDOUT_TAG is required}"
# When supplied, reuse the two completed base result directories from this
# earlier tag instead of recomputing them.  Callers are responsible for using
# this only when the evaluation contract below is unchanged.
baseline_tag="${AGENT_BASELINE_TAG:-}"
py="/home/f630/homePLUS/agentic/envs/agenticml/bin/python"
log="$repo/sft-runs/holdout-${run_tag}.log"

if [[ ! -f "$candidate/adapter_model.safetensors" ]]; then
  printf 'missing candidate adapter: %s\n' "$candidate/adapter_model.safetensors" >&2
  exit 2
fi

exec >>"$log" 2>&1

run_eval() {
  local dataset="$1" output="$2" workspace="$3" revision="$4" min_id="$5" max_id="$6" limit="$7"
  local -a bounds=()
  if [[ "$min_id" != "-" ]]; then
    bounds+=(--min-task-id "$min_id" --max-task-id "$max_id")
  fi

  AGENT_MODEL_MAX_CALLS=32 PYTHONPATH="$repo" "$py" "$repo/scripts/run_mbpp.py" \
    --dataset "$dataset" --workspace-root "$workspace" --output-root "$output" \
    --pi /home/f630/homePLUS/agentic/run/node22/bin/pi --python "$py" \
    --upstream-url http://127.0.0.1:8000/v1/chat/completions \
    --model qwen2.5-coder-14b-instruct --model-revision "$revision" \
    --attempt 1 --limit "$limit" --timeout 600 "${bounds[@]}"
}

require_completed_episodes() {
  local result_root="$1" expected="$2"
  local actual
  actual="$(find "$result_root" -type f -name episode.json 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$actual" != "$expected" ]]; then
    printf 'baseline is incomplete: %s has %s/%s episodes\n' "$result_root" "$actual" "$expected" >&2
    exit 2
  fi
}

mbpp="$repo/experiments/mbpp/mbpp-manifest.json"
humaneval="$repo/experiments/humaneval/humaneval-manifest.json"
base_revision="qwen2.5-coder-14b-base"
candidate_revision="mbpp-qwen14b-lora/sha256:$(sha256sum "$candidate/adapter_model.safetensors" | cut -d ' ' -f1)"

echo "holdout_start=$(date -Is) tag=$run_tag"

if [[ -n "$baseline_tag" ]]; then
  baseline_mbpp="$repo/holdout-${baseline_tag}-mbpp-base"
  baseline_humaneval="$repo/holdout-${baseline_tag}-humaneval-base"
  require_completed_episodes "$baseline_mbpp" 90
  require_completed_episodes "$baseline_humaneval" 164
  echo "base_reused=$(date -Is) tag=$baseline_tag mbpp=$baseline_mbpp humaneval=$baseline_humaneval revision=$base_revision"
else
  "$repo/scripts/serve_a6000_model.sh" base
  run_eval "$mbpp" "$repo/holdout-${run_tag}-mbpp-base" "$repo/holdout-workspaces-${run_tag}-mbpp-base" "$base_revision" 511 600 90
  run_eval "$humaneval" "$repo/holdout-${run_tag}-humaneval-base" "$repo/holdout-workspaces-${run_tag}-humaneval-base" "$base_revision" - - 164
fi

export AGENT_SFT_ADAPTER_ROOT="$candidate"
export AGENT_SFT_MODEL_REVISION="$candidate_revision"
"$repo/scripts/serve_a6000_model.sh" candidate
run_eval "$mbpp" "$repo/holdout-${run_tag}-mbpp-candidate" "$repo/holdout-workspaces-${run_tag}-mbpp-candidate" "$candidate_revision" 511 600 90
run_eval "$humaneval" "$repo/holdout-${run_tag}-humaneval-candidate" "$repo/holdout-workspaces-${run_tag}-humaneval-candidate" "$candidate_revision" - - 164

echo "holdout_finished=$(date -Is) tag=$run_tag"
