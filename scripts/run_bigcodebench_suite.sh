#!/usr/bin/env bash
# =============================================================================
# Automated Tri-Benchmark Pipeline: MBPP Completion -> BigCodeBench -> Tri-Report
# =============================================================================
set -euo pipefail

PYTHON="/home/cxr/miniconda3/envs/vllm/bin/python"
CODE_DIR="/home/cxr/agentic/code"
BASE_MODEL="/home/cxr/agentic/models/qwen2.5-coder-14b-instruct"
SFT_ADAPTER="/home/cxr/agentic/checkpoints/sft-runs/qwen14b-apps-clean-v2-260906/epoch2"
RL_ADAPTER="/home/cxr/agentic/rl-runs/apps-rl-cycle-002/candidate_policy_adapter"
BCB_DATASET="/home/cxr/agentic/caches/bigcodebench/BigCodeBench-Hard-v0.1.1.jsonl"
OUTPUT_ROOT="/home/cxr/agentic/rl-runs/official-qwen-bigcodebench"
MBPP_PID="${1:-2628927}"

echo "=== Tri-Benchmark Runner Started at $(date -Is) ==="
echo "Monitoring MBPP PID: $MBPP_PID"

# Step 1: Wait for MBPP process to finish if running
if ps -p "$MBPP_PID" > /dev/null 2>&1; then
    echo "Waiting for MBPP PID $MBPP_PID to complete..."
    while ps -p "$MBPP_PID" > /dev/null 2>&1; do
        sleep 15
    done
    echo "MBPP process $MBPP_PID finished at $(date -Is)!"
else
    echo "MBPP process $MBPP_PID is not currently running. Proceeding..."
fi

# Step 2: Ensure MBPP results exist
MBPP_EVAL="/home/cxr/agentic/rl-runs/official-qwen-evalplus/mbpp/rl-v2_samples_eval_results.json"
if [[ -f "$MBPP_EVAL" ]]; then
    echo "MBPP RL evaluation results verified: $MBPP_EVAL"
else
    echo "Warning: $MBPP_EVAL not found. If MBPP evaluation was not triggered automatically, running evaluation now..."
    MBPP_SAMPLES="/home/cxr/agentic/rl-runs/official-qwen-evalplus/mbpp/rl-v2_samples.jsonl"
    if [[ -f "$MBPP_SAMPLES" ]]; then
        MBPP_OVERRIDE_PATH=/home/cxr/agentic/caches/evalplus/MbppPlus-v0.2.0.jsonl \
        "$PYTHON" -m evalplus.evaluate --dataset mbpp --samples "$MBPP_SAMPLES" --i_just_wanna_run
    fi
fi

# Step 3: BigCodeBench-Hard Evaluation on GPU 0
echo ""
echo "=== Step 3: Running BigCodeBench-Hard (148 tasks, complete split) ==="

# 3A: Base Model
echo ">>> Evaluating BigCodeBench-Hard on Base 14B..."
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$CODE_DIR/scripts/run_official_qwen_bigcodebench.py" \
    --dataset "$BCB_DATASET" \
    --split complete \
    --subset hard \
    --base-model "$BASE_MODEL" \
    --model-tag base \
    --output-dir "$OUTPUT_ROOT" \
    --device cuda:0 \
    --batch-size 4 \
    --workers 8

# 3B: SFT Policy-v0
echo ">>> Evaluating BigCodeBench-Hard on SFT Policy-v0..."
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$CODE_DIR/scripts/run_official_qwen_bigcodebench.py" \
    --dataset "$BCB_DATASET" \
    --split complete \
    --subset hard \
    --base-model "$BASE_MODEL" \
    --adapter "$SFT_ADAPTER" \
    --model-tag sft-v0 \
    --output-dir "$OUTPUT_ROOT" \
    --device cuda:0 \
    --batch-size 4 \
    --workers 8

# 3C: RL Policy-v2
echo ">>> Evaluating BigCodeBench-Hard on RL Policy-v2..."
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$CODE_DIR/scripts/run_official_qwen_bigcodebench.py" \
    --dataset "$BCB_DATASET" \
    --split complete \
    --subset hard \
    --base-model "$BASE_MODEL" \
    --adapter "$RL_ADAPTER" \
    --model-tag rl-v2 \
    --output-dir "$OUTPUT_ROOT" \
    --device cuda:0 \
    --batch-size 4 \
    --workers 8

# Step 4: Compile Final Tri-Benchmark Summary
echo ""
echo "=== Step 4: Compiling Tri-Benchmark Summary ==="
"$PYTHON" "$CODE_DIR/scripts/compile_tri_benchmark_report.py"

echo "=== Tri-Benchmark Runner Completed at $(date -Is) ==="
