#!/usr/bin/env bash
# Loop certified MBPP batches until intra-group reward variance is found.
# One batch = 4 trajectories of Mbpp/118 via stage_d_certified_rerun.py.
# Stops early when a batch contains both PASSED and FAILED verifier_status.
# Max 6 batches (~60 min) to stay inside the overall GPU budget.
set -u
SMOKE=/home/cxr/verl-closeout/smoke
cd "$SMOKE" || exit 1
POLICY=749d7a9f0e47fc5ace0f0bd18d6285df58d38f97f8222c75a4d1adca69638578
for n in 31 32 33 34 35 36; do
  if ! curl -s -m 10 http://127.0.0.1:8931/healthz > /dev/null; then
    echo "batch $n: infer server down, aborting loop" >> loop.log
    break
  fi
  rm -rf "/home/cxr/verl-closeout/stage-d/run$n"
  echo "=== batch dstage$n start $(date -u +%FT%TZ) ===" >> loop.log
  PYTHONPATH=/home/cxr/agentic/code:/home/cxr/verl-closeout/smoke \
    /home/cxr/miniconda3/envs/vllm/bin/python stage_d_certified_rerun.py \
    --run-id "dstage$n" \
    --bridge-url http://127.0.0.1:8931/v1/chat/completions \
    --pi-bin /home/cxr/verl-closeout/pi-global/bin/pi \
    --output-dir "/home/cxr/verl-closeout/stage-d/run$n" \
    --policy-checksum "$POLICY" > "run$n-diag.log" 2>&1
  echo "=== batch dstage$n exit $? $(date -u +%FT%TZ) ===" >> loop.log
  HAS_PASS=$(grep -c '"verifier_status": "PASSED"' "run$n-diag.log" 2>/dev/null || true)
  HAS_FAIL=$(grep -c '"verifier_status": "FAILED"' "run$n-diag.log" 2>/dev/null || true)
  echo "batch dstage$n: PASSED=$HAS_PASS FAILED=$HAS_FAIL" >> loop.log
  if [ "$HAS_PASS" -ge 1 ] && [ "$HAS_FAIL" -ge 1 ]; then
    echo "VARIANCE dstage$n $(date -u +%FT%TZ)" > LOOP_DONE
    break
  fi
done
echo "LOOP-END $(date -u +%FT%TZ)" >> LOOP_DONE 2>/dev/null || echo "LOOP-END $(date -u +%FT%TZ)" > LOOP_DONE
