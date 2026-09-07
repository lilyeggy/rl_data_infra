#!/usr/bin/env bash
set -euo pipefail

# Fixed-workload proxy ablation.  Both arms retain vLLM logprob generation so
# the only intentional difference is Data Plane proxy/evidence processing.
PROJECT_ROOT="${PROJECT_ROOT:-/home/f630/homePLUS/agent-data-plane}"
PYTHON_BIN="${PYTHON_BIN:-/home/f630/homePLUS/agentic/envs/agenticml/bin/python}"
OUTPUT_ROOT="${1:-$PROJECT_ROOT/sft-runs/proxy-replay-repeats-260831}"
MODEL="${MODEL:-qwen2.5-coder-14b-instruct}"
MODEL_REVISION="${MODEL_REVISION:-qwen2.5-coder-14b-vllm-0.6.1}"
UPSTREAM_URL="${UPSTREAM_URL:-http://127.0.0.1:8001/v1/chat/completions}"
TRIALS="${TRIALS:-3}"
REQUESTS="${REQUESTS:-12}"

mkdir -p "$OUTPUT_ROOT"
for trial in $(seq 1 "$TRIALS"); do
  for concurrency in 1 4 8; do
    # Alternate the order to avoid systematically charging model warm-up to one arm.
    if (( trial % 2 )); then modes=(direct proxy); else modes=(proxy direct); fi
    for mode in "${modes[@]}"; do
      PYTHONPATH="$PROJECT_ROOT" "$PYTHON_BIN" "$PROJECT_ROOT/scripts/benchmark_proxy_replay_a6000.py" \
        --mode "$mode" --upstream-url "$UPSTREAM_URL" --model "$MODEL" \
        --model-revision "$MODEL_REVISION" --output-dir "$OUTPUT_ROOT/trial-$trial/c$concurrency/$mode" \
        --concurrency "$concurrency" --requests "$REQUESTS" --max-tokens 128 --seed 20260831
    done
  done
done
