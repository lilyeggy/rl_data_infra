#!/usr/bin/env bash
set -euo pipefail

repo="${AGENT_REPO:?AGENT_REPO is required}"
pid="${AGENT_EXPANSION_PID:?AGENT_EXPANSION_PID is required}"
log="$repo/sft-runs/mbpp-expansion-watch-260830.log"
exec >>"$log" 2>&1

echo "watch_start=$(date -Is) pid=$pid"
while kill -0 "$pid" 2>/dev/null; do sleep 30; done
echo "expansion_process_finished=$(date -Is)"

python="/home/f630/homePLUS/agentic/envs/agenticml/bin/python"
PYTHONPATH="$repo" "$python" "$repo/scripts/finalize_mbpp_expansion.py" \
  --root "$repo/mbpp-expansion-301-500-260830" \
  --output "$repo/sft-runs/mbpp-expansion-301-500-260830-summary.json" \
  --label "MBPP expansion 301-500"

echo "polar_contract_tests_start=$(date -Is)"
if PYTHONPATH="$repo" "$python" -m pytest -q \
  "$repo/tests/integrations/test_polar_client.py" \
  "$repo/tests/integrations/test_polar_adapter.py" \
  "$repo/tests/integrations/test_offline_polar_pipeline.py"; then
  echo "polar_contract_tests=passed"
else
  echo "polar_contract_tests=failed"
fi
echo "polar_live_comparison_start=$(date -Is)"
AGENT_REPO="$repo" "$repo/scripts/run_polar_live_comparison_a6000.sh"
echo "polar_comparison=see $repo/sft-runs/polar-live-comparison-260830"
echo "watch_finished=$(date -Is)"
