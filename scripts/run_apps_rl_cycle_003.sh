#!/usr/bin/env bash
set -euo pipefail

# Complete End-to-End Scaled Agentic RL Cycle 003 Runner
# Model: Qwen2.5-Coder-14B-Instruct + APPS Clean-v2 Epoch 2 SFT Adapter
# GPU: CUDA_VISIBLE_DEVICES=0

PYTHON="/home/cxr/miniconda3/envs/vllm/bin/python"
BASE_DIR="/home/cxr/agentic"
CODE_DIR="${BASE_DIR}/code"
RUN_DIR="${BASE_DIR}/rl-runs/apps-rl-cycle-003"
MODEL_PATH="${BASE_DIR}/models/qwen2.5-coder-14b-instruct"
SFT_ADAPTER="${BASE_DIR}/checkpoints/sft-runs/qwen14b-apps-clean-v2-260906/epoch2"
APPS_MANIFEST="${BASE_DIR}/datasets/apps/train.manifest.json"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

mkdir -p "${RUN_DIR}"
LOG_FILE="${RUN_DIR}/cycle-003.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================================"
echo "Starting Agentic RL Cycle 003: Policy-v3 Scaled Training"
echo "Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "GPU 0 status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo "========================================================"

# Step 1: Collect Group Rollouts (50 tasks x G=4 = 200 rollouts)
echo ""
echo "=== Phase 1: On-Policy Rollout Collection with Soft Case Rewards ==="
"${PYTHON}" "${CODE_DIR}/scripts/run_apps_rl_rollout.py" \
    --manifest "${APPS_MANIFEST}" \
    --tasks-file "${RUN_DIR}/train_tasks_50.json" \
    --model "${MODEL_PATH}" \
    --adapter "${SFT_ADAPTER}" \
    --output-dir "${RUN_DIR}" \
    --group-size 4 \
    --max-new-tokens 640 \
    --device "cuda:0"

echo ""
echo "=== Phase 2: Inspecting Slime Admission Batch & Preflight ==="
"${PYTHON}" "${CODE_DIR}/scripts/train_grpo_lora.py" \
    --admission-batch "${RUN_DIR}/rollouts/slime-admission.json" \
    --output "${RUN_DIR}/candidate_policy_adapter" \
    --preflight-only

echo ""
echo "=== Phase 3: Executing GRPO Policy Gradient Update ==="
"${PYTHON}" "${CODE_DIR}/scripts/train_grpo_lora.py" \
    --admission-batch "${RUN_DIR}/rollouts/slime-admission.json" \
    --output "${RUN_DIR}/candidate_policy_adapter" \
    --model "${MODEL_PATH}" \
    --base-adapter "${SFT_ADAPTER}" \
    --lr 8e-6 \
    --beta 0.008 \
    --clip 0.2 \
    --epochs 1

echo ""
echo "========================================================"
echo "RL Cycle 003 Training Completed Successfully!"
echo "Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Output adapter: ${RUN_DIR}/candidate_policy_adapter"
echo "========================================================"
