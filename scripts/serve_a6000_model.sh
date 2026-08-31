#!/usr/bin/env bash

set -euo pipefail

server_root="/home/f630/homePLUS/agentic/code"
runtime_root="/home/f630/homePLUS/agentic"
model_root="$runtime_root/models/qwen2.5-coder-14b-instruct"
server_log="$runtime_root/run/server-14b-canonical-v2.log"
policy_variant="${1:-candidate}"

case "$policy_variant" in
  base)
    adapter_root=""
    model_revision="qwen2.5-coder-14b-base/sha256:8c6caec78ff329ee9f0523cea6b85bbe7eb06b010a0b9e3faade7b7606d466fd"
    ;;
  candidate)
    adapter_root="${AGENT_SFT_ADAPTER_ROOT:-/home/f630/homePLUS/agent-data-plane/sft-runs/qwen14b-canonical-generic-json-v2}"
    model_revision="${AGENT_SFT_MODEL_REVISION:-qwen2.5-coder-14b-canonical-v2/sha256:f6323f5116aecba15cbb242e2162193de06d0abcfd3c358116cd177265a42d2c}"
    ;;
  *)
    printf 'usage: %s [base|candidate]\n' "$0" >&2
    exit 2
    ;;
esac

for required_path in \
  "$server_root/experiments/local_model/openai_server.py" \
  "$runtime_root/run/env.sh" \
  "$model_root/model.safetensors.index.json"; do
  if [[ ! -f "$required_path" ]]; then
    printf 'missing required model-service input: %s\n' "$required_path" >&2
    exit 2
  fi
done
if [[ -n "$adapter_root" && ! -f "$adapter_root/adapter_model.safetensors" ]]; then
  printf 'missing required adapter: %s\n' "$adapter_root/adapter_model.safetensors" >&2
  exit 2
fi

server_pid="$(pgrep -f 'experiments/local_model/openai_server.py' | head -n 1 || true)"
if [[ -n "$server_pid" ]]; then
  kill "$server_pid"
  for _ in $(seq 1 30); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$server_pid" 2>/dev/null; then
    printf 'model server did not stop cleanly: pid=%s\n' "$server_pid" >&2
    exit 3
  fi
fi

cd "$server_root"
# The shared legacy env file appends to PYTHONPATH directly and therefore
# expects it to exist even when this launcher uses nounset.
export PYTHONPATH="${PYTHONPATH:-}"
source "$runtime_root/run/env.sh"
export LOCAL_MODEL="$model_root"
if [[ -n "$adapter_root" ]]; then
  export LOCAL_ADAPTER="$adapter_root"
else
  unset LOCAL_ADAPTER
fi
export LOCAL_MODEL_REVISION="$model_revision"
export PORT=8000
# Tool calls can contain complete Python implementations; keep this explicit
# so restarts cannot regress to a truncating serving default.
export QWEN_SERV_CONTEXT_TOKENS="${QWEN_SERV_CONTEXT_TOKENS:-32768}"
export QWEN_SERV_MAX_TOKENS="${QWEN_SERV_MAX_TOKENS:-8192}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf '=== canonical-v2 model service %s ===\n' "$(date -Is)" >"$server_log"
setsid nohup python3 -u experiments/local_model/openai_server.py >>"$server_log" 2>&1 </dev/null &

for _ in $(seq 1 180); do
  if curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    printf 'model service ready: revision=%s\n' "$model_revision"
    exit 0
  fi
  sleep 2
done

printf 'model service failed to become healthy; inspect %s\n' "$server_log" >&2
exit 4
