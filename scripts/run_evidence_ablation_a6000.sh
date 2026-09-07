#!/usr/bin/env bash
# Alternate RL evidence capture on/off while holding workload and serving fixed.
set -euo pipefail

repo="${AGENT_REPO:?AGENT_REPO is required}"
trials="${TRIALS:-3}"
root="${ABLATION_ROOT:-$repo/sft-runs/evidence-ablation-260831}"
python_bin="${PYTHON_BIN:-/home/f630/homePLUS/agentic/envs/agenticml/bin/python}"
image="${HARNESS_IMAGE:-agent-data-plane-pi:0.84.2}"
image_digest="${HARNESS_IMAGE_DIGEST:-df5e334b934d22475cf8465ce18e49b5eb247104a05bb27cbdfc8ea2c5ac76fb}"
model="${MODEL_ID:-qwen2.5-coder-14b-instruct}"
model_revision="${MODEL_REVISION:-qwen2.5-coder-14b-base/sha256:8c6caec78ff329ee9f0523cea6b85bbe7eb06b010a0b9e3faade7b7606d466fd}"
lock="$root/.run.lock"

cd "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$root/trials"
mkdir "$lock" 2>/dev/null || { echo "ablation already running"; exit 1; }
trap 'rmdir "$lock" 2>/dev/null || true' EXIT
exec > >(tee -a "$root/runner.log") 2>&1

run_mode() {
  local trial="$1"
  local mode="$2"
  local concurrency="$3"
  local out="$root/trials/trial-$trial/$mode"
  mkdir -p "$out"
  local mode_arg=()
  [[ "$mode" == "observability-only" ]] && mode_arg=(--no-capture-response-logprobs)
  "$python_bin" "$repo/scripts/benchmark_data_plane_a6000.py" \
    --concurrency "$concurrency" --output-dir "$out" --image "$image" \
    --image-digest "$image_digest" --model "$model" --model-revision "$model_revision" \
    --experiment "evidence-ablation/260831" --allow-verifier-failures "${mode_arg[@]}" \
    >"$out/benchmark-c${concurrency}.log" 2>&1
}

for trial in $(seq 1 "$trials"); do
  echo "trial=$trial started=$(date -Is)"
  for concurrency in 1 4 8; do
    if (( trial % 2 )); then
      run_mode "$trial" rl-logprobs "$concurrency"
      run_mode "$trial" observability-only "$concurrency"
    else
      run_mode "$trial" observability-only "$concurrency"
      run_mode "$trial" rl-logprobs "$concurrency"
    fi
    echo "trial=$trial concurrency=$concurrency completed=$(date -Is)"
  done
done

"$python_bin" "$repo/scripts/summarize_evidence_ablation.py" \
  --root "$root" --output "$root/summary.md"
echo "ablation_status=completed root=$root finished=$(date -Is)"
