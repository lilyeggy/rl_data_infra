#!/usr/bin/env bash
# Standard, non-agent pass@1 evaluation for a frozen Base/SFT comparison.
set -euo pipefail

repo="${AGENT_REPO:?AGENT_REPO is required}"
eval_python="/home/f630/homePLUS/agentic/envs/evalplus/bin/python"
evalplus="/home/f630/homePLUS/agentic/envs/evalplus/bin/evalplus.codegen"
evaluate="/home/f630/homePLUS/agentic/envs/evalplus/bin/evalplus.evaluate"
model="qwen2.5-coder-14b-instruct"
root="$repo/standard-evalplus-260901"
log="$repo/sft-runs/standard-evalplus-260901.log"
exec >>"$log" 2>&1

run_suite() {
  local label="$1"
  local out="$root/$label"
  for dataset in humaneval mbpp; do
    OPENAI_API_KEY=none "$evalplus" "$model" "$dataset" --backend openai \
      --base_url http://127.0.0.1:8000/v1 --root "$out" --greedy --resume
    "$evaluate" "$dataset" --samples "$out/$dataset/${model}_openai_temp_0.0.jsonl" --i_just_wanna_run
  done
}

mkdir -p "$root"
echo "standard_evalplus_started=$(date -Is)"
"$repo/scripts/serve_a6000_model.sh" base
run_suite base
export AGENT_SFT_ADAPTER_ROOT="${AGENT_SFT_ADAPTER_ROOT:?AGENT_SFT_ADAPTER_ROOT is required}"
export AGENT_SFT_MODEL_REVISION="${AGENT_SFT_MODEL_REVISION:?AGENT_SFT_MODEL_REVISION is required}"
"$repo/scripts/serve_a6000_model.sh" candidate
run_suite sft
echo "standard_evalplus_finished=$(date -Is)"
