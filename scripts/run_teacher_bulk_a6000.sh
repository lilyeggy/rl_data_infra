#!/usr/bin/env bash

set -u

repo="/home/f630/homePLUS/agent-data-plane"
output="$repo/teacher-runs-bulk-260826-v1"
python="/home/f630/homePLUS/agentic/envs/agenticml/bin/python"
pi="/home/f630/homePLUS/agentic/run/node22/bin/pi"
selected="$repo/experiments/swebench/selected_instances.json"
workspace_root="$repo/swebench"
venv_root="$repo/swebench/venvs"
instances=(
  psf__requests-2931
  psf__requests-5414
  psf__requests-6028
  pytest-dev__pytest-10051
  sympy__sympy-23534
  sympy__sympy-23824
  sympy__sympy-23950
  sympy__sympy-24213
  sympy__sympy-24539
)

mkdir -p "$output"
cd "$repo"
echo "bulk_start=$(date -Is)"
echo "train_instances=${#instances[@]} rounds=4 max_trajectories=36"
echo "model_revision=qwen2.5-coder-14b-canonical-v2"

for attempt in 1 2 3 4; do
  echo "round_start attempt=$attempt time=$(date -Is)"
  PYTHONPATH="$repo" "$python" "$repo/scripts/run_selected_swebench.py" \
    --selected "$selected" \
    --workspace-root "$workspace_root" \
    --venv-root "$venv_root" \
    --output-root "$output/round-$attempt" \
    --pi "$pi" \
    --upstream-url http://127.0.0.1:8000/v1/chat/completions \
    --provider local-openai-compatible \
    --provider-api openai-completions \
    --model qwen2.5-coder-14b-instruct \
    --model-revision "qwen2.5-coder-14b-canonical-v2/sha256:f6323f5116aecba15cbb242e2162193de06d0abcfd3c358116cd177265a42d2c" \
    --policy baseline-v1 \
    --instances "${instances[@]}" \
    --attempt "$attempt" \
    --timeout 600
  status=$?
  echo "round_end attempt=$attempt status=$status time=$(date -Is)"
done

episode_count=$(find "$output" -path "*/finalized/episode.json" -print | wc -l | tr -d ' ')
echo "finalized_episode_count=$episode_count"
echo "bulk_end=$(date -Is)"
