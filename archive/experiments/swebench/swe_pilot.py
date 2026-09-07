"""ARCHIVED: SWE-bench Docker pilot from the old A6000 lab workflow.

Selects N instances from the local verified parquet, writes a local dataset
jsonl (SWE-bench format), and runs swebench's OFFICIAL run_evaluation
(Docker-based: build instance image -> apply patch -> run tests -> grade).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

PARQUET = Path("/home/f630/homePLUS/agentic/code/experiments/swebench/swebench_verified.parquet")
OUT = Path("/home/f630/homePLUS/agentic/run/sweb-pilot")
OUT.mkdir(parents=True, exist_ok=True)

_COLS = [
    "instance_id", "repo", "base_commit", "patch", "test_patch",
    "problem_statement", "hints_text", "created_at", "version",
    "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit", "difficulty",
]


def _as_list(v):
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str) and v.strip():
        return json.loads(v)
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5, help="instances to select")
    ap.add_argument("--repo", default="sympy", help="repo filter")
    ap.add_argument("--one", action="store_true", help="only build/run the first instance")
    args = ap.parse_args()

    df = pd.read_parquet(PARQUET)
    if args.repo:
        df = df[df["repo"].str.contains(args.repo, na=False)]
    # prefer small FAIL_TO_PASS (cleaner signal) and deterministic order
    df = df.assign(_nftp=df["FAIL_TO_PASS"].map(lambda v: len(_as_list(v))))
    df = df.sort_values(["_nftp", "instance_id"]).drop(columns=["_nftp"])
    df = df.head(1 if args.one else args.count)
    print("selected:", df["instance_id"].tolist(), flush=True)

    instances = []
    for r in df.to_dict("records"):
        inst = {c: r.get(c) for c in _COLS}
        inst["FAIL_TO_PASS"] = _as_list(r.get("FAIL_TO_PASS"))
        inst["PASS_TO_PASS"] = _as_list(r.get("PASS_TO_PASS"))
        instances.append(inst)
    dataset_path = OUT / "pilot.jsonl"
    with open(dataset_path, "w") as f:
        for inst in instances:
            f.write(json.dumps(inst) + "\n")

    # empty predictions (validation of the Docker path first)
    preds = OUT / "predictions.jsonl"
    with open(preds, "w") as f:
        for inst in instances:
            f.write(json.dumps({
                "instance_id": inst["instance_id"],
                "model_patch": "",  # empty -> P2P baseline check
                "model_name_or_path": "pilot-empty",
            }) + "\n")

    print(f"dataset: {dataset_path}  predictions: {preds}", flush=True)
    print("PILOT_PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
