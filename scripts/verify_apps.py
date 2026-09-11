#!/usr/bin/env python3
"""Verify one APPS stdin/stdout program in an isolated workspace."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


def _line_text(value: object) -> str:
    """Normalize one APPS stdin/stdout line without losing nested values."""
    if isinstance(value, list):
        return " ".join(_line_text(item) for item in value)
    if isinstance(value, str):
        return value
    return str(value)


def _stdio_text(value: object) -> str:
    """Normalize an APPS stdin/stdout case, whose top-level items are lines."""
    if isinstance(value, list):
        return "\n".join(_line_text(item) for item in value)
    return _line_text(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--source-worktree", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=10, help="Per-case timeout in seconds")
    parser.add_argument("--eval-all", action="store_true", default=True, help="Evaluate up to max cases to compute partial pass rate")
    args = parser.parse_args()
    task = json.loads(args.manifest.read_text())["tasks"][args.task_id]
    io = task["input_output"]
    inputs, outputs = io.get("inputs", []), io.get("outputs", [])
    report = {
        "task_id": args.task_id,
        "case_count": len(inputs),
        "passed_cases": 0,
        "pass_rate": 0.0,
        "resolved": False,
    }
    solution = args.source_worktree / "solution.py"
    if io.get("fn_name"):
        report["reason"] = "call-based APPS task is unsupported by the stdin/stdout verifier"
    elif not solution.is_file() or len(inputs) != len(outputs):
        report["reason"] = "missing solution or malformed test cases"
    else:
        cases = []
        # Evaluate cases up to 10 cases to balance speed and granular signal
        max_eval = min(len(inputs), 10)
        for index in range(len(inputs)):
            stdin, expected = inputs[index], outputs[index]
            try:
                result = subprocess.run(
                    [args.python, str(solution)],
                    cwd=args.source_worktree,
                    input=_stdio_text(stdin),
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                    check=False,
                )
                actual = result.stdout.strip()
                wanted = _stdio_text(expected).strip()
                passed = (result.returncode == 0 and actual == wanted)
                cases.append({
                    "index": index,
                    "returncode": result.returncode,
                    "passed": passed,
                    "stdout": result.stdout[-2000:],
                    "stderr": result.stderr[-2000:],
                })
            except subprocess.TimeoutExpired:
                cases.append({
                    "index": index,
                    "returncode": -1,
                    "passed": False,
                    "stdout": "",
                    "stderr": f"TimeoutExpired after {args.timeout}s",
                })
            # If not eval_all and failed, break early
            if not args.eval_all and not cases[-1]["passed"]:
                break
            # If we reached 10 cases and already had failures, stop testing remaining
            if index >= max_eval - 1 and not all(c["passed"] for c in cases):
                break

        passed_count = sum(1 for item in cases if item["passed"])
        is_resolved = (len(cases) == len(inputs) and passed_count == len(inputs))
        report.update(
            {
                "resolved": is_resolved,
                "tested_cases": len(cases),
                "passed_cases": passed_count,
                "pass_rate": round(passed_count / max(1, len(inputs)), 4),
                "cases": cases,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if report["resolved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
