"""ARCHIVED: run the old official SWE-bench Docker pilot."""
from __future__ import annotations

import os

import swebench

DATASET = "/home/f630/homePLUS/agentic/run/sweb-pilot/pilot.jsonl"
PREDS = "/home/f630/homePLUS/agentic/run/sweb-pilot/predictions.jsonl"
IDS = os.environ.get("PILOT_IDS", "sympy__sympy-11618").split(",")

swebench.run_evaluation(
    dataset_name=DATASET,
    split="test",
    instance_ids=IDS,
    predictions_path=PREDS,
    max_workers=1,
    open_file_limit=4096,
    run_id="pilot2",
    timeout=2400,
    rewrite_reports=False,  # fresh run, not a re-grade
    modal=False,
    report_dir="/home/f630/homePLUS/agentic/run/sweb-pilot/report",
)
print("PILOT_EVAL_DONE", flush=True)
