#!/usr/bin/env bash
set -euo pipefail

# Train genuine SFT LoRA from pure Qwen2.5-Coder-14B Base model
# GPU: CUDA_VISIBLE_DEVICES=1 (72GB free VRAM)

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/home/cxr/agentic/code:${PYTHONPATH:-}"

PYTHON="/home/cxr/miniconda3/envs/vllm/bin/python"
BASE_DIR="/home/cxr/agentic"
PACKAGE="${BASE_DIR}/checkpoints/sft-packages/apps-qwen14b-clean-v2-260906"
MODEL="${BASE_DIR}/models/qwen2.5-coder-14b-base"
OUTPUT="${BASE_DIR}/checkpoints/sft-runs/qwen14b-base-apps-clean-v2"

mkdir -p "${OUTPUT}"
LOG_FILE="${OUTPUT}/train.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================================"
echo "Starting SFT LoRA Training on Pure Base Model"
echo "Model: ${MODEL}"
echo "Package: ${PACKAGE}"
echo "Output: ${OUTPUT}"
echo "Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "========================================================"

"${PYTHON}" "${BASE_DIR}/code/scripts/train_apps_lora_sft_v2.py" \
    --package "${PACKAGE}" \
    --model "${MODEL}" \
    --output "${OUTPUT}" \
    --epochs 2 \
    --lr 1e-4 \
    --rank 8 \
    --token-budget 12288

echo "========================================================"
echo "SFT LoRA Training on Pure Base Completed!"
echo "Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "========================================================"
