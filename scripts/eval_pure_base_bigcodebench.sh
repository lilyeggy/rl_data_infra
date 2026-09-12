#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/home/cxr/agentic/code:${PYTHONPATH:-}"

PYTHON="/home/cxr/miniconda3/envs/vllm/bin/python"
BASE_DIR="/home/cxr/agentic"
MODEL="${BASE_DIR}/models/qwen2.5-coder-14b-base"
BCB_DATASET="${BASE_DIR}/caches/bigcodebench/BigCodeBench-Hard-v0.1.1.jsonl"
OUTPUT="${BASE_DIR}/rl-runs/official-qwen-base-bigcodebench"

mkdir -p "${OUTPUT}"
LOG_FILE="${OUTPUT}/eval_pure_base_bigcodebench.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================================"
echo "Starting BigCodeBench Evaluation on Pure Base Model"
echo "Model: ${MODEL}"
echo "Dataset: ${BCB_DATASET}"
echo "Output: ${OUTPUT}"
echo "Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "========================================================"

"${PYTHON}" "${BASE_DIR}/code/scripts/run_official_qwen_bigcodebench.py" \
    --dataset "${BCB_DATASET}" \
    --split complete \
    --subset hard \
    --base-model "${MODEL}" \
    --model-tag base \
    --direct-completion \
    --output-dir "${OUTPUT}" \
    --device cuda:0 \
    --batch-size 4 \
    --workers 8

echo "========================================================"
echo "BigCodeBench Pure Base Evaluation Finished!"
echo "Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "========================================================"
