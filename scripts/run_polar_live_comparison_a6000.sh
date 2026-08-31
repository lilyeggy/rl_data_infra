#!/usr/bin/env bash
set -u

# Best-effort Polar live comparison.  It is intentionally time-sliced after
# the Local/Pi batch so both systems do not compete for the A6000.
repo="${AGENT_REPO:?AGENT_REPO is required}"
out="${POLAR_OUTPUT_DIR:-$repo/sft-runs/polar-live-comparison-260830}"
log="${POLAR_LOG_FILE:-$out.log}"
polar_dir="${POLAR_DIR:-$repo/../ProRL-Agent-Server-official}"
rollout_port="${POLAR_ROLLOUT_PORT:-18080}"
gateway_port="${POLAR_GATEWAY_PORT:-18100}"
topology="$out/topology.yaml"
venv="$out/.venv"
polar_image="polar-localhost-calculator:latest"
polar_python="/home/f630/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12"
polar_site="/home/f630/homePLUS/polar-runtime/site"
polar_samples="${POLAR_NUM_SAMPLES:-4}"
model_id="${POLAR_MODEL_ID:-qwen2.5-coder-14b-instruct}"
inference_base_url="${POLAR_INFERENCE_BASE_URL:-http://127.0.0.1:8000}"
smoke_runner="$repo/scripts/polar_smoke_runner.py"

mkdir -p "$out"
exec >>"$log" 2>&1
echo "polar_live_start=$(date -Is)"

# These ports are dedicated to this comparison. Clean up only listeners on
# them so a supervisor restart cannot leave an orphaned Polar process behind.
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${rollout_port}/tcp" >/dev/null 2>&1 || true
  fuser -k "${gateway_port}/tcp" >/dev/null 2>&1 || true
  sleep 1
fi

fail() {
  echo "polar_live_status=failed stage=$1"
  echo "polar_live_finished=$(date -Is)"
  exit 0
}

if [[ ! -d "$polar_dir/.git" ]]; then
  git clone --depth 1 https://github.com/NVIDIA-NeMo/ProRL-Agent-Server.git "$polar_dir" || fail clone
fi

cp "$polar_dir/examples/calculator/topology.vllm.yaml" "$topology" || fail config
sed -i \
  -e 's#save_dir: ./rollout_results#save_dir: /out/rollout_results#' \
  -e "s#/out/rollout_results#$out/rollout_results#" \
  -e "s/port: 8080/port: $rollout_port/" \
  -e "s#127.0.0.1:8080#127.0.0.1:$rollout_port#" \
  -e "s/port: 8100/port: $gateway_port/" \
  -e "s#127.0.0.1:8100#127.0.0.1:$gateway_port#" \
  -e '/id: localhost-node-02/,$d' \
  "$topology"
sed -i \
  -e "s#model_served: .*#model_served: $model_id#" \
  -e "s#base_url: http://127.0.0.1:8000#base_url: $inference_base_url#" \
  "$topology"

if ! docker image inspect polar-localhost-calculator:latest >/dev/null 2>&1; then
  timeout 600 docker build --network host \
    --build-arg POLAR_CALCULATOR_IMAGE_VERSION=3 \
    --tag polar-localhost-calculator:latest \
    "$polar_dir/examples/calculator/runtime" || fail image
fi
mkdir -p "$out/rollout_results"

rollout_log="$out/rollout.log"
gateway_log="$out/gateway.log"
PYTHONPATH="$polar_site:$polar_dir/src" nohup "$polar_python" -m polar.cli serve_rollout -c "$topology" >"$rollout_log" 2>&1 &
rollout_pid=$!
PYTHONPATH="$polar_site:$polar_dir/src" nohup "$polar_python" -m polar.cli serve_gateway -c "$topology" --node-id localhost-node-01 >"$gateway_log" 2>&1 &
gateway_pid=$!
cleanup() {
  kill "$gateway_pid" "$rollout_pid" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 60); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$rollout_port/health" >/dev/null 2>&1 && \
     curl -fsS --max-time 2 "http://127.0.0.1:$gateway_port/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" == 1 ]] || fail service

cd "$polar_dir"
resource_log="$out/resource-c${polar_samples}.csv"
echo "timestamp,utilization.gpu.percent,memory.used.mib,memory.total.mib,power.draw.watts" >"$resource_log"
timeout 1800 env AGENT_REPO="$repo" POLAR_DIR="$polar_dir" POLAR_TOPOLOGY="$topology" PYTHONPATH="$polar_site:$polar_dir/src" POLAR_NUM_SAMPLES="$polar_samples" POLAR_MODEL_ID="$model_id" POLAR_RUNTIME_IMAGE="${POLAR_RUNTIME_IMAGE:-agent-data-plane-pi:0.84.2}" "$polar_python" "$smoke_runner" \
  >"$out/benchmark.log" 2>&1 &
benchmark_pid=$!
while kill -0 "$benchmark_pid" 2>/dev/null; do
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' >>"$resource_log" || true
  sleep 2
done
wait "$benchmark_pid" || fail benchmark
cp "$out/benchmark.log" "$out/benchmark-c${polar_samples}.log"
if ! grep -Eq "[[:space:]]${polar_samples}/${polar_samples}[[:space:]]*$" "$out/benchmark.log"; then
  fail benchmark_result
fi
cp "$polar_dir/examples/calculator/topology.vllm.yaml" "$out/upstream-topology-reference.yaml" 2>/dev/null || true
echo "polar_live_status=passed"
echo "polar_live_artifacts=$out"
echo "polar_live_finished=$(date -Is)"
