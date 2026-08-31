#!/usr/bin/env bash
set -u

repo="${AGENT_REPO:?AGENT_REPO is required}"
log="$repo/sft-runs/overnight-polar-compare-260830.log"
lock="$repo/sft-runs/.overnight-polar-compare.lock"
mkdir "$lock" 2>/dev/null || exit 0
trap 'rmdir "$lock" 2>/dev/null || true' EXIT
exec >>"$log" 2>&1
echo "overnight_start=$(date -Is)"

quality_ok() {
  local benchmark="$1" samples="$2"
  grep -Eq "[[:space:]]${samples}/${samples}[[:space:]]*$" "$benchmark" 2>/dev/null || return 1
  # Done only means the harness exited. Require every reported reward to be
  # positive so an empty/failed verifier result cannot be treated as valid.
  awk '
    /^[[:alnum:]_-]+[[:space:]]+-?[0-9]+(\.[0-9]+)?[[:space:]]+[0-9]+\/[0-9]+[[:space:]]*$/ {
      seen=1; if (($2 + 0) <= 0) bad=1
    }
    END { exit (!seen || bad) }
  ' "$benchmark"
}

for attempt in $(seq 1 6); do
  echo "attempt=$attempt model_restart=$(date -Is)"
  if ! "$repo/scripts/serve_a6000_model.sh" base; then
    echo "attempt=$attempt model_restart=failed"
    sleep 60
    continue
  fi

  for samples in 1 4 8; do
    echo "attempt=$attempt concurrency=$samples start=$(date -Is)"
    rm -f "$repo/sft-runs/polar-live-comparison-260830/benchmark-c${samples}.log"
    POLAR_NUM_SAMPLES="$samples" AGENT_REPO="$repo" \
      "$repo/scripts/run_polar_live_comparison_a6000.sh"
    benchmark="$repo/sft-runs/polar-live-comparison-260830/benchmark-c${samples}.log"
    if quality_ok "$benchmark" "$samples"; then
      echo "attempt=$attempt concurrency=$samples status=valid"
    else
      echo "attempt=$attempt concurrency=$samples status=invalid"
      break
    fi
  done

  if [[ -f "$repo/sft-runs/polar-live-comparison-260830/benchmark-c8.log" ]] && \
     quality_ok "$repo/sft-runs/polar-live-comparison-260830/benchmark-c8.log" 8; then
    echo "overnight_status=completed"
    echo "overnight_finished=$(date -Is)"
    local_root="$repo/sft-runs/local-pi-smoke-auto-260830"
    local_status=0
    if [[ -f "$repo/scripts/run_local_pi_smoke_a6000.py" ]]; then
      for local_samples in 1 4 8; do
        echo "local_pi concurrency=$local_samples start=$(date -Is)"
        if ! python3 "$repo/scripts/run_local_pi_smoke_a6000.py" \
          --concurrency "$local_samples" --output-dir "$local_root"; then
          echo "local_pi concurrency=$local_samples status=failed"
          local_status=1
          break
        fi
        echo "local_pi concurrency=$local_samples status=valid"
      done
    else
      echo "local_pi status=runner_missing"
      local_status=1
    fi
    echo "local_pi_status=$local_status"
    if [[ -f "$repo/scripts/build_rollout_comparison_report.py" ]]; then
      python3 "$repo/scripts/build_rollout_comparison_report.py" \
        --polar-root "$repo/sft-runs/polar-live-comparison-260830" \
        --local-root "$local_root" \
        --local-summary "$repo/sft-runs/mbpp-expansion-301-500-260830-summary.json" \
        --output "$repo/sft-runs/polar-live-comparison-260830/final-report.md" \
        && echo "comparison_report=generated" \
        || echo "comparison_report=failed"
    fi
    exit 0
  fi
  sleep 60
done

echo "overnight_status=blocked_after_retries"
echo "overnight_finished=$(date -Is)"
exit 0
