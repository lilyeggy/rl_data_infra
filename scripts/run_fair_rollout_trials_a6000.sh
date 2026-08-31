#!/usr/bin/env bash
# Repeated, alternating-order comparison of the complete Local Data Plane and Polar.
set -euo pipefail

repo="${AGENT_REPO:?AGENT_REPO is required}"
trials="${TRIALS:-3}"
root="${BENCHMARK_ROOT:-$repo/sft-runs/fair-rollout-repeats-260830}"
python_bin="${PYTHON_BIN:-/home/f630/homePLUS/agentic/envs/agenticml/bin/python}"
image="${HARNESS_IMAGE:-agent-data-plane-pi:0.84.2}"
image_digest="${HARNESS_IMAGE_DIGEST:-df5e334b934d22475cf8465ce18e49b5eb247104a05bb27cbdfc8ea2c5ac76fb}"
model="${MODEL_ID:-qwen2.5-coder-14b-instruct}"
model_revision="${MODEL_REVISION:-qwen2.5-coder-14b-base/sha256:8c6caec78ff329ee9f0523cea6b85bbe7eb06b010a0b9e3faade7b7606d466fd}"
upstream_url="${UPSTREAM_URL:-http://127.0.0.1:8001/v1/chat/completions}"
lock="$root/.run.lock"

cd "$repo"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$root/trials"
mkdir "$lock" 2>/dev/null || { echo "benchmark already running: $lock"; exit 1; }
trap 'rmdir "$lock" 2>/dev/null || true' EXIT
exec > >(tee -a "$root/runner.log") 2>&1

run_local() {
  local trial="$1" concurrency="$2" out="$root/trials/trial-$trial/local"
  mkdir -p "$out"
  "$python_bin" "$repo/scripts/benchmark_data_plane_a6000.py" \
    --concurrency "$concurrency" --output-dir "$out" --image "$image" \
    --image-digest "$image_digest" --model "$model" --model-revision "$model_revision" \
    --upstream-url "$upstream_url" --experiment "polar-fair-repeat/260830" \
    --allow-verifier-failures \
    >"$out/benchmark-c${concurrency}.log" 2>&1
}

run_polar() {
  local trial="$1" concurrency="$2" out="$root/trials/trial-$trial/polar"
  mkdir -p "$out"
  POLAR_OUTPUT_DIR="$out" POLAR_LOG_FILE="$out/service-c${concurrency}.log" \
    POLAR_NUM_SAMPLES="$concurrency" POLAR_MODEL_ID="$model" \
    POLAR_INFERENCE_BASE_URL="http://127.0.0.1:8001" POLAR_RUNTIME_IMAGE="$image" \
    AGENT_REPO="$repo" "$repo/scripts/run_polar_live_comparison_a6000.sh"
}

for trial in $(seq 1 "$trials"); do
  echo "trial=$trial started=$(date -Is)"
  for concurrency in 1 4 8; do
    if (( trial % 2 )); then
      run_local "$trial" "$concurrency"
      run_polar "$trial" "$concurrency"
    else
      run_polar "$trial" "$concurrency"
      run_local "$trial" "$concurrency"
    fi
    echo "trial=$trial concurrency=$concurrency completed=$(date -Is)"
  done
done

"$python_bin" "$repo/scripts/summarize_fair_rollout_trials.py" \
  --root "$root" --output "$root/repeat-summary.md"
echo "benchmark_status=completed root=$root finished=$(date -Is)"
