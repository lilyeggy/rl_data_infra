#!/usr/bin/env python3
"""Audit APPS batch evidence, corrected eligibility, and exact solution duplicates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

REQUIRED_VALID_FILES = (
    "artifact-refs.json",
    "execution-run-manifest.json",
    "launch-plan.json",
    "policy-artifact.json",
    "producer-artifact.json",
    "raw-events.jsonl",
    "summary.json",
    "verifier-output.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reverify-report", type=Path, required=True)
    args = parser.parse_args()

    reverify = json.loads(args.reverify_report.read_text(encoding="utf-8"))
    corrected = {item["task_id"]: item["new_status"] for item in reverify["results"]}
    task_ids = []
    missing = Counter()
    solution_digests = Counter()
    evidence_complete = 0
    corrected_eligible = 0
    valid_count = 0

    for summary_path in sorted(args.run_root.glob("*/summary.json")):
        run_dir = summary_path.parent
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        task_id = summary["task_id"]
        task_ids.append(task_id)
        if summary.get("execution_validity") != "VALID":
            continue
        valid_count += 1
        complete = summary.get("integrity") == "COMPLETE"
        for name in REQUIRED_VALID_FILES:
            path = run_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                missing[name] += 1
                complete = False
        if summary.get("model_call_count", 0) <= 0:
            missing["positive_model_call_count"] += 1
            complete = False
        launch_plan = json.loads(
            (run_dir / "launch-plan.json").read_text(encoding="utf-8")
        )
        solution = Path(launch_plan["workspace"]) / "solution.py"
        if solution.is_file() and solution.stat().st_size:
            digest = hashlib.sha256(solution.read_bytes()).hexdigest()
            solution_digests[digest] += 1
        else:
            missing["solution.py"] += 1
            complete = False
        if complete:
            evidence_complete += 1
        if complete and corrected.get(task_id) == "PASSED":
            corrected_eligible += 1

    duplicate_solutions = sum(count - 1 for count in solution_digests.values() if count > 1)
    report = {
        "total": len(task_ids),
        "valid": valid_count,
        "corrected_eligible": corrected_eligible,
        "evidence_complete": evidence_complete,
        "evidence_complete_rate": evidence_complete / valid_count if valid_count else 0.0,
        "duplicate_task_ids": len(task_ids) - len(set(task_ids)),
        "duplicate_solutions": duplicate_solutions,
        "duplicate_solution_rate": duplicate_solutions / valid_count if valid_count else 0.0,
        "missing_or_empty": dict(sorted(missing.items())),
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if evidence_complete == valid_count and not report["duplicate_task_ids"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
