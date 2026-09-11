#!/bin/bash
# Stage E rollout round: 4 fresh Pi episodes on Mbpp/118 via the vLLM bridge.
# $1=round dir name  $2=run id  $3=model revision  $4=policy fingerprint checksum
set -u
ROUND=$1; RUNID=$2; REVISION=$3; POLICY=$4
SMOKE=/home/cxr/verl-closeout/smoke
cd "$SMOKE" || exit 1
rm -rf "/home/cxr/verl-closeout/stage-e/$ROUND"
PYTHONPATH=/home/cxr/agentic/code:/home/cxr/verl-closeout/smoke \
  /home/cxr/miniconda3/envs/vllm/bin/python stage_d_certified_rerun.py \
  --run-id "$RUNID" \
  --bridge-url http://127.0.0.1:8931/v1/chat/completions \
  --pi-bin /home/cxr/verl-closeout/pi-global/bin/pi \
  --output-dir "/home/cxr/verl-closeout/stage-e/$ROUND" \
  --policy-checksum "$POLICY" \
  --model-revision "$REVISION"
