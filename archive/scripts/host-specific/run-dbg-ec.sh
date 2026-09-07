#!/bin/bash
# ARCHIVED: hard-coded lab-host diagnostic.
# Determinative test: run the PROVEN-WORKING dbg-ec.py through the exact same
# launch mechanism as run-sft.sh (env.sh + cd code + setsid nohup). If this
# fails too, the run-sft.sh launch environment is the culprit, not sft_14b.py.
LOG=/home/f630/homePLUS/agentic/run/dbg-ec-v2.log
echo "=== dbg-ec via run-mech $(date) ===" > "$LOG"
source /home/f630/homePLUS/agentic/run/env.sh
cd /home/f630/homePLUS/agentic/code
export LOCAL_MODEL=/home/f630/homePLUS/agentic/models/qwen2.5-coder-14b-instruct
setsid nohup python3 -u /home/f630/homePLUS/agentic/run/dbg-ec.py >> "$LOG" 2>&1 </dev/null &
sleep 3
echo "pid=$(pgrep -f dbg-ec.py | head -1)" >> "$LOG"
echo RUN_DBG_STARTED
