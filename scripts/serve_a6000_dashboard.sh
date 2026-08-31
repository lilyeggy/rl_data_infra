#!/usr/bin/env bash

set -euo pipefail

repo_root="/home/f630/homePLUS/agent-data-plane"
python_bin="/home/f630/homePLUS/agentic/envs/agenticml/bin/python"
dashboard_log="$repo_root/runs/dashboard-server.log"
model_launcher="$repo_root/scripts/serve_a6000_model.sh"

for required_path in "$python_bin" "$model_launcher"; do
  if [[ ! -e "$required_path" ]]; then
    printf 'missing dashboard dependency: %s\n' "$required_path" >&2
    exit 2
  fi
done

dashboard_pid="$(pgrep -f 'python -m src.dashboard_server' | head -n 1 || true)"
if [[ -n "$dashboard_pid" ]]; then
  kill "$dashboard_pid"
  for _ in $(seq 1 20); do
    if ! kill -0 "$dashboard_pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$dashboard_pid" 2>/dev/null; then
    printf 'dashboard did not stop cleanly: pid=%s\n' "$dashboard_pid" >&2
    exit 3
  fi
fi

cd "$repo_root"
setsid nohup env PYTHONPATH=. "$python_bin" -m src.dashboard_server \
  --repo-root "$repo_root" \
  --model-launcher "$model_launcher" \
  --python "$python_bin" \
  >"$dashboard_log" 2>&1 </dev/null &

for _ in $(seq 1 30); do
  if curl -fsS --max-time 2 http://127.0.0.1:8788/api/status >/dev/null 2>&1; then
    printf 'dashboard ready with launcher=%s\n' "$model_launcher"
    exit 0
  fi
  sleep 1
done

printf 'dashboard failed to become healthy; inspect %s\n' "$dashboard_log" >&2
exit 4
