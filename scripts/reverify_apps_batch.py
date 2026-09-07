#!/usr/bin/env python3
"""Reverify an APPS batch into a separate audit directory."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _reverify_one(
    *,
    summary_path: Path,
    manifest: Path,
    verifier: Path,
    python: str,
    output_root: Path,
) -> dict[str, object]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    task_id = summary["task_id"]
    launch_plan = json.loads(
        summary_path.with_name("launch-plan.json").read_text(encoding="utf-8")
    )
    task_output = output_root / f"{task_id}.json"
    result = subprocess.run(
        [
            python,
            str(verifier),
            "--manifest",
            str(manifest),
            "--task-id",
            task_id,
            "--source-worktree",
            launch_plan["workspace"],
            "--python",
            python,
            "--output",
            str(task_output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    report = json.loads(task_output.read_text(encoding="utf-8"))
    new_status = "PASSED" if report.get("resolved") else "FAILED"
    return {
        "task_id": task_id,
        "old_status": summary["verifier_status"],
        "new_status": new_status,
        "returncode": result.returncode,
        "reason": report.get("reason"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    if args.output_root.exists():
        raise ValueError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)

    all_summaries = []
    valid_summary_paths = []
    for path in sorted(args.run_root.glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        all_summaries.append(summary)
        if summary.get("execution_validity") == "VALID":
            valid_summary_paths.append(path)

    verifier = Path(__file__).with_name("verify_apps.py").resolve()
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(
                _reverify_one,
                summary_path=path,
                manifest=args.manifest.resolve(),
                verifier=verifier,
                python=args.python,
                output_root=args.output_root,
            )
            for path in valid_summary_paths
        ]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: str(item["task_id"]))
    old_counts = Counter(item.get("verifier_status", "UNKNOWN") for item in all_summaries)
    validity_counts = Counter(
        item.get("execution_validity", "UNKNOWN") for item in all_summaries
    )
    new_counts = Counter(item["new_status"] for item in results)
    new_counts["ERROR"] = validity_counts["INFRA_INVALID"]
    transitions = Counter(
        f"{item['old_status']}->{item['new_status']}" for item in results
    )
    report = {
        "source_run_root": str(args.run_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "total": len(all_summaries),
        "reverified_valid": len(results),
        "old_status_counts": dict(sorted(old_counts.items())),
        "execution_validity_counts": dict(sorted(validity_counts.items())),
        "new_status_counts": dict(sorted(new_counts.items())),
        "transitions": dict(sorted(transitions.items())),
        "results": results,
    }
    report_path = args.output_root / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "results"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
