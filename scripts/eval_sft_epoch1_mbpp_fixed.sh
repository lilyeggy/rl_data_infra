#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/cxr/agentic/code:${PYTHONPATH:-}"

PYTHON="/home/cxr/miniconda3/envs/vllm/bin/python"
BASE_DIR="/home/cxr/agentic"
MODEL="${BASE_DIR}/models/qwen2.5-coder-14b-base"
ADAPTER="${BASE_DIR}/checkpoints/sft-runs/qwen14b-base-apps-clean-v2/epoch1"
OUTPUT="${BASE_DIR}/rl-runs/official-qwen-base-evalplus"

mkdir -p "${OUTPUT}"
LOG_FILE="${OUTPUT}/eval_sft_epoch1_mbpp_fixed.log"
exec > >(tee "${LOG_FILE}") 2>&1

echo "========================================================"
echo "Starting MBPP Evaluation on SFT Model (Fixed Sanitization)"
echo "Model: ${MODEL}"
echo "Adapter: ${ADAPTER}"
echo "Output: ${OUTPUT}"
echo "Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "========================================================"

"${PYTHON}" "${BASE_DIR}/code/scripts/run_official_qwen_evalplus.py" \
    --model "${MODEL}" \
    --adapter "${ADAPTER}" \
    --dataset mbpp \
    --model-tag sft-base-epoch1 \
    --output-dir "${OUTPUT}" \
    --prompt-format base \
    --batch-size 8 \
    --device cuda:0

echo "========================================================"
echo "MBPP SFT (Fixed) Evaluation Finished!"
echo "Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "========================================================"
