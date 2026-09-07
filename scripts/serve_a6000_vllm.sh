#!/usr/bin/env bash

set -euo pipefail

# Production-style serving path for the fixed Qwen + Pi rollout pool.
# The legacy Transformers server remains available in serve_a6000_model.sh.
runtime_root="/home/f630/homePLUS/agentic"
model_root="$runtime_root/models/qwen2.5-coder-14b-instruct"
vllm_env="$runtime_root/envs/vllm-qwen14b"
server_log="$runtime_root/run/vllm-qwen14b.log"
model_revision="qwen2.5-coder-14b-base/sha256:8c6caec78ff329ee9f0523cea6b85bbe7eb06b010a0b9e3faade7b7606d466fd"
port="${VLLM_PORT:-8000}"
tool_compat_port="${VLLM_TOOL_COMPAT_PORT:-8001}"
tool_bridge_host="${VLLM_TOOL_BRIDGE_HOST:-172.17.0.1}"
data_plane_root="${AGENT_DATA_PLANE_ROOT:-/home/f630/homePLUS/agent-data-plane}"
max_model_len="${VLLM_MAX_MODEL_LEN:-32768}"
max_num_seqs="${VLLM_MAX_NUM_SEQS:-8}"
gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

if [[ ! -x "$vllm_env/bin/python" ]]; then
  printf 'missing vLLM environment: %s\n' "$vllm_env" >&2
  exit 2
fi
if [[ ! -f "$model_root/config.json" ]]; then
  printf 'missing model: %s\n' "$model_root" >&2
  exit 2
fi

for pid in $(pgrep -f 'experiments/local_model/openai_server.py' || true); do
  kill "$pid" 2>/dev/null || true
done
for _ in $(seq 1 60); do
  if ! pgrep -f 'experiments/local_model/openai_server.py' >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if pgrep -f 'experiments/local_model/openai_server.py' >/dev/null 2>&1; then
  printf 'legacy model server did not stop cleanly\n' >&2
  exit 3
fi

for pid in $(pgrep -f 'vllm.entrypoints.openai.api_server.*--port' || true); do
  kill "$pid" 2>/dev/null || true
done
for pid in $(pgrep -f 'vllm_tool_compat_proxy.py' || true); do
  kill "$pid" 2>/dev/null || true
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
printf '=== vLLM Qwen serving %s ===\n' "$(date -Is)" >"$server_log"
# Bind on the host interface so Docker Harness containers can use the
# explicit host-gateway route; the endpoint remains firewall-scoped locally.
setsid nohup "$vllm_env/bin/python" -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port "$port" \
  --model "$model_root" \
  --served-model-name qwen2.5-coder-14b-instruct \
  --dtype half \
  --trust-remote-code \
  --max-model-len "$max_model_len" \
  --max-num-seqs "$max_num_seqs" \
  --gpu-memory-utilization "$gpu_memory_utilization" \
  --guided-decoding-backend lm-format-enforcer \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --return-tokens-as-token-ids \
  --enable-prefix-caching \
  >>"$server_log" 2>&1 </dev/null &

for _ in $(seq 1 240); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
    setsid nohup "$vllm_env/bin/python" "$data_plane_root/scripts/vllm_tool_compat_proxy.py" \
      "http://127.0.0.1:$port/v1/chat/completions" 127.0.0.1 "$tool_compat_port" \
      >>"$runtime_root/run/vllm-tool-compat.log" 2>&1 </dev/null &
    setsid nohup "$vllm_env/bin/python" "$data_plane_root/scripts/vllm_tool_compat_proxy.py" \
      "http://127.0.0.1:$port/v1/chat/completions" "$tool_bridge_host" "$tool_compat_port" \
      >>"$runtime_root/run/vllm-tool-compat-bridge.log" 2>&1 </dev/null &
    printf 'vLLM ready: model=%s revision=%s port=%s\n' \
      "qwen2.5-coder-14b-instruct" "$model_revision" "$port"
    printf 'Pi tool-compat endpoint: http://127.0.0.1:%s/v1\n' "$tool_compat_port"
    exit 0
  fi
  sleep 2
done

printf 'vLLM failed to become healthy; inspect %s\n' "$server_log" >&2
exit 4
