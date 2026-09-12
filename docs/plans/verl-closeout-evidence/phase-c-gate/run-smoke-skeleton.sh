#!/usr/bin/env bash
# Stage C framework-smoke runner skeleton — NOT EXECUTABLE WITHOUT A WINDOW.
#
# Gate (all must hold before uncommenting the launch block):
#   1. MBPP SFT-eval finished (PID 1007953 exited; log shows completion).
#   2. P0 re-confirmed frozen (adapter sha256 6db6a40c...).
#   3. Explicit GPU UUIDs + deadline + run ID filled below.
#   4. Independent env under /data (CUDA 12.9 / torch 2.10 / vLLM 0.17.0 pinned).
# Budget: 60 min from first GPU touch (load + retries + cleanup).
set -euo pipefail

RUN_ID="${RUN_ID:-UNSET}"
GPU_UUIDS="${GPU_UUIDS:-UNSET}"   # e.g. GPU-4288d6c9-...,GPU-52776180-...
DEADLINE="${DEADLINE:-UNSET}"     # ISO-8601 usage-window deadline
SMOKE_CONFIG="${SMOKE_CONFIG:-docs/plans/verl-closeout-evidence/phase-c-gate/smoke-config-DRAFT.yaml}"

if [[ "${RUN_ID}" == "UNSET" || "${GPU_UUIDS}" == "UNSET" || "${DEADLINE}" == "UNSET" ]]; then
  echo "REFUSING: RUN_ID/GPU_UUIDS/DEADLINE must be explicit. No GPU touched." >&2
  exit 2
fi

# --- preflight (read-only, safe now) ---
# ssh cxr@172.17.43.193 'ps -p 1007953 > /dev/null && echo STILL-RUNNING || echo EVAL-DONE'
# ssh cxr@172.17.43.193 'sha256sum /home/cxr/agentic/checkpoints/sft-runs/qwen14b-base-apps-clean-v2/epoch1/adapter_model.safetensors'
# ssh cxr@172.17.43.193 'nvidia-smi --query-gpu=uuid,memory.free --format=csv'

# --- launch (requires window; keep commented until gate passes) ---
# echo "launching framework-smoke run ${RUN_ID} (deadline ${DEADLINE})"
# ... env setup + verl smoke launch + evidence capture go here ...
# All products tagged framework-smoke.

echo "skeleton ready; gate not passed; nothing executed."
