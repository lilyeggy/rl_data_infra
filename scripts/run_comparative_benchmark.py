#!/usr/bin/env python3
"""Run comparative benchmark on HumanEval across Base, SFT, and RL."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    models = [
        ("Base Model (Qwen-14B)", None, "/home/cxr/agentic/rl-runs/benchmark-eval/humaneval-base.json"),
        ("SFT Policy-v0 (Epoch 2)", "/home/cxr/agentic/checkpoints/sft-runs/qwen14b-apps-clean-v2-260906/epoch2", "/home/cxr/agentic/rl-runs/benchmark-eval/humaneval-sft-v0.json"),
        ("RL Policy-v2 (Cycle 002)", "/home/cxr/agentic/rl-runs/apps-rl-cycle-002/candidate_policy_adapter", "/home/cxr/agentic/rl-runs/benchmark-eval/humaneval-rl-v2.json"),
    ]

    base_model_path = "/home/cxr/agentic/models/qwen2.5-coder-14b-instruct"
    dataset_path = "/home/cxr/agentic/caches/evalplus/HumanEval.jsonl"
    python_bin = "/home/cxr/miniconda3/envs/vllm/bin/python"
    eval_script = "/home/cxr/agentic/code/scripts/eval_humaneval_clean.py"

    summary_rows = []

    for name, adapter, out_path in models:
        print(f"\n=======================================================", flush=True)
        print(f"Evaluating: {name}", flush=True)
        print(f"=======================================================", flush=True)

        cmd = [
            python_bin,
            eval_script,
            "--dataset", dataset_path,
            "--model", base_model_path,
            "--output", out_path,
            "--device", "cuda:0",
            "--limit", str(args.limit),
        ]
        if adapter:
            cmd.extend(["--adapter", adapter])

        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[error] failed running {name}", flush=True)
            continue

        data = json.loads(Path(out_path).read_text(encoding="utf-8"))
        summary_rows.append({
            "name": name,
            "total": data["total_tasks"],
            "passed": data["passed_tasks"],
            "pass_at_1": data["pass_at_1"],
            "pass_pct": data["pass_percentage"],
        })

    print("\n\n================ FINAL COMPARATIVE BENCHMARK REPORT ================")
    print(f"| Model / Policy | Evaluated Tasks | Passed | Pass@1 (%) |")
    print(f"| :--- | :--- | :--- | :--- |")
    for r in summary_rows:
        print(f"| {r['name']} | {r['total']} | {r['passed']} | {r['pass_pct']} |")
    print("===================================================================\n")

if __name__ == "__main__":
    main()
