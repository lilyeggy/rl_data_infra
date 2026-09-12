#!/usr/bin/env bash
set -euo pipefail

# Dual-Track Benchmark Verification for Policy-v3 vs SFT-v0 vs Base
# Track 1: APPS Holdout (50 tasks)
# Track 2: HumanEval & HumanEval+ (164 tasks)

PYTHON="/home/cxr/miniconda3/envs/vllm/bin/python"
BASE_DIR="/home/cxr/agentic"
CODE_DIR="${BASE_DIR}/code"
RUN_DIR="${BASE_DIR}/rl-runs/apps-rl-cycle-003"
MODEL_PATH="${BASE_DIR}/models/qwen2.5-coder-14b-instruct"
SFT_ADAPTER="${BASE_DIR}/checkpoints/sft-runs/qwen14b-apps-clean-v2-260906/epoch2"
RL_V3_ADAPTER="${RUN_DIR}/candidate_policy_adapter"
APPS_MANIFEST="${BASE_DIR}/datasets/apps/train.manifest.json"
HOLDOUT_TASKS="${RUN_DIR}/holdout_eval_tasks.json"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

LOG_FILE="${RUN_DIR}/dual_track_eval.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================================"
echo "Starting Dual-Track Benchmark Verification for Policy-v3"
echo "Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "========================================================"

# Track 1: In-Domain APPS Holdout Evaluation
echo ""
echo "=== Track 1: Evaluating APPS In-Domain Holdout (50 tasks) ==="
echo "--- Evaluating Base Model ---"
"${PYTHON}" "${CODE_DIR}/scripts/evaluate_apps_holdout.py" \
    --manifest "${APPS_MANIFEST}" \
    --tasks-file "${HOLDOUT_TASKS}" \
    --model "${MODEL_PATH}" \
    --output-dir "${RUN_DIR}/apps_holdout_eval" \
    --tag "base" \
    --device "cuda:0"

echo "--- Evaluating SFT Policy-v0 ---"
"${PYTHON}" "${CODE_DIR}/scripts/evaluate_apps_holdout.py" \
    --manifest "${APPS_MANIFEST}" \
    --tasks-file "${HOLDOUT_TASKS}" \
    --model "${MODEL_PATH}" \
    --adapter "${SFT_ADAPTER}" \
    --output-dir "${RUN_DIR}/apps_holdout_eval" \
    --tag "sft-v0" \
    --device "cuda:0"

echo "--- Evaluating RL Policy-v3 ---"
"${PYTHON}" "${CODE_DIR}/scripts/evaluate_apps_holdout.py" \
    --manifest "${APPS_MANIFEST}" \
    --tasks-file "${HOLDOUT_TASKS}" \
    --model "${MODEL_PATH}" \
    --adapter "${RL_V3_ADAPTER}" \
    --output-dir "${RUN_DIR}/apps_holdout_eval" \
    --tag "rl-v3" \
    --device "cuda:0"

# Track 2: Out-of-Domain HumanEval Evaluation for Policy-v3
echo ""
echo "=== Track 2: Evaluating HumanEval & HumanEval+ (164 tasks) for Policy-v3 ==="
"${PYTHON}" "${CODE_DIR}/scripts/run_official_qwen_evalplus.py" \
    --model "${MODEL_PATH}" \
    --dataset humaneval \
    --model-tag rl-v3 \
    --output-dir "${BASE_DIR}/rl-runs/official-qwen-evalplus" \
    --batch-size 8 \
    --device "cuda:0"

echo ""
echo "========================================================"
echo "Dual-Track Benchmark Verification Complete!"
echo "Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "========================================================"
