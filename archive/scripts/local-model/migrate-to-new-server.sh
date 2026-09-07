#!/bin/bash
# ARCHIVED: environment-specific GPU server migration helper.
#
#   scripts/migrate-to-new-server.sh <host> <ssh_port> <ssh_password>
#
# What it does (and why it is cheap):
#   - pushes the code repo (which already contains the SFT dataset +
#     run scripts + server), ~12MB
#   - runs server-setup.sh on the new host (env + 15G model download + provider)
#   - re-trains the SFT adapter IN ~22s from the bundled dataset — no need to
#     transfer any trained adapter
#   - smoke-tests the local-model server + a real Pi call
#
# Nothing heavy is transferred: base model and SWE-bench parquet re-download on
# the new host; trained adapters are re-trained from the bundled dataset.
set -euo pipefail
HOST=${1:?host}; PORT=${2:?port}; PASS=${3:?password}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export SSHPASS="$PASS"
SSH="sshpass -e ssh -p $PORT -o StrictHostKeyChecking=accept-new root@$HOST"
SCP="sshpass -e scp -P $PORT -o StrictHostKeyChecking=accept-new"

echo "== [1/4] push code repo =="
tar czf /tmp/agentic-rl-code.tgz \
  --exclude=.git --exclude=__pycache__ --exclude='*.pyc' \
  --exclude=artifacts/swebench-v3 --exclude=node_modules \
  -C "$REPO" .
$SCP /tmp/agentic-rl-code.tgz root@"$HOST":/root/
$SSH "mkdir -p /root/agentic-rl && tar xzf /root/agentic-rl-code.tgz -C /root/agentic-rl && echo code-ok"

echo "== [2/4] env + model + provider on new host (slow) =="
$SCP "$REPO/scripts/server-setup.sh" root@"$HOST":/root/
$SSH "bash /root/server-setup.sh"

echo "== [3/4] re-train SFT adapter from bundled dataset (~22s) =="
$SSH "cd /root/agentic-rl && PYTHONPATH=/root/agentic-rl python3 experiments/local_model/sft_train.py \
  --model /root/rivermind-data/models/qwen2.5-7b-instruct \
  --data experiments/local_model/sft-dataset-v3.jsonl \
  --output /root/rivermind-data/models/qwen-sft-adapter-7b-v3 \
  --epochs 12 --lr 2e-4"

echo "== [4/4] smoke test: local server + real Pi =="
$SCP "$REPO/scripts/../experiments/local_model/openai_server.py" root@"$HOST":/root/agentic-rl/experiments/local_model/ 2>/dev/null || true
$SSH 'cd /root/agentic-rl && export PYTHONPATH=/root/agentic-rl
pkill -9 -f openai_server.py 2>/dev/null; sleep 2
LOCAL_ADAPTER=/root/rivermind-data/models/qwen-sft-adapter-7b-v3 PORT=8000 \
  setsid nohup python3 experiments/local_model/openai_server.py > /root/rivermind-data/qwen-server.log 2>&1 </dev/null &
sleep 20; curl -s http://127.0.0.1:8000/health'

echo ""
echo "MIGRATION_DONE — next: run P2/P3 via experiments/local_model/pi_local.py / grpo_pi.py"
