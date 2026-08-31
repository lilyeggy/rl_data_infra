#!/bin/bash
# rescue-critical.sh — pull server-only artifacts to local before a system restore.
# Priority: small irreplaceable configs/datasets first, then trained adapters.
# Retries each file through flaky-network windows.
: "${SSHPASS:?ARCHIVED script requires SSHPASS to be supplied explicitly}"
export SSHPASS
HOST=root@sh01-ssh.gpuhome.cc
# NB: ssh uses -p for port, scp uses -P. Remote must be host:path (joined).
SCP="sshpass -e scp -P 30741 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 -r"
OUT=/Users/mac/Desktop/agentic-rl/artifacts/server-rescue
mkdir -p "$OUT/models"
pull() { # remote_path local_dir
  local r="$1" d="$2"
  [ -e "$d/$(basename "$r")" ] && { echo "SKIP(have) $(basename "$r")"; return 0; }
  $SCP "$HOST:$r" "$d/" 2>/dev/null && echo "OK $(basename "$r")" || echo "FAIL $(basename "$r")"
}
SMALL=(
  /root/.pi/agent/models.json
  /root/rivermind-data/swebench/selected_instances.json
  /root/rivermind-data/swebench/swe-sft-dataset.jsonl
  /root/rivermind-data/swebench/validity/validity.json
  /root/rivermind-data/swebench/package-v3/episodes.jsonl
  /root/rivermind-data/swebench/package-v3/raw-events.jsonl
  /root/rivermind-data/swebench/results/base-7b/summary.json
  /root/rivermind-data/swebench/results/swe-7b/summary.json
  /root/rivermind-data/pi-system-prompt.txt
)
BIG=(
  /root/rivermind-data/models/qwen-grpo-pi-7b-o2
  /root/rivermind-data/models/qwen-swe-adapter-7b
  /root/rivermind-data/models/qwen-sft-adapter-7b-v3
)
for round in 1 2 3 4 5; do
  echo "=== round $round $(date +%H:%M:%S) ==="
  for f in "${SMALL[@]}"; do pull "$f" "$OUT"; done
  for f in "${BIG[@]}"; do pull "$f" "$OUT/models"; done
  # if we have the GRPO adapter dir, we're basically done
  [ -d "$OUT/models/qwen-grpo-pi-7b-o2" ] && [ -f "$OUT/episodes.jsonl" ] && { echo "CRITICAL_RESCUED"; break; }
  sleep 20
done
echo "RESCUE_PASS_DONE"; ls -la "$OUT" "$OUT/models"
