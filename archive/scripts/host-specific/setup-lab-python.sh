#!/bin/bash
# ARCHIVED: retired A6000 lab bootstrap for the custom training path.
# Install the Python/RL stack into the agentic conda env on the lab A6000.
LOG=/home/f630/homePLUS/agentic/run/setup-python.log
EXPORT_CACHE="export PIP_CACHE_DIR=/home/f630/homePLUS/agentic/data/pip-cache; export TMPDIR=/home/f630/homePLUS/agentic/data/tmp;"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate agentic
python --version > "$LOG" 2>&1
echo "=== installing torch (cu121) ===" >> "$LOG"
$EXPORT_CACHE pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision >> "$LOG" 2>&1
echo "torch rc=$?" >> "$LOG"
echo "=== installing ml stack ===" >> "$LOG"
$EXPORT_CACHE pip install transformers peft accelerate datasets modelscope >> "$LOG" 2>&1
echo "ml rc=$?" >> "$LOG"
echo "SETUP_PYTHON_CACHE=$? done=$(date)" >> "$LOG"
tail -3 "$LOG"
