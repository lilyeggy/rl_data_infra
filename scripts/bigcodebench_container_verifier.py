#!/usr/bin/env python3
"""Entrypoint executed *inside* the pinned BigCodeBench evaluator container."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bigcodebench.eval import PASS, untrusted_check


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task = json.loads(args.task.read_text())
    report: dict[str, object] = {"task_id": task["task_id"], "resolved": False}
    try:
        status, details = untrusted_check(
            args.solution.read_text(), task["test"], task["entry_point"],
            30 * 1024, 30 * 1024, 10, 1.0, 20.0,
        )
        report.update({"status": status, "details": details, "resolved": status == PASS})
    except Exception as exc:  # Evaluator faults must be observable, never pass.
        report.update({"status": "INFRA_ERROR", "error": repr(exc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False) + "\n")
    return 0 if report["resolved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
