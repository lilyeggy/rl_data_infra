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
    args = parser.parse_args()
    task = json.loads(args.manifest.read_text())["tasks"][args.task_id]
    io = task["input_output"]
    inputs, outputs = io.get("inputs", []), io.get("outputs", [])
    report = {"task_id": args.task_id, "case_count": len(inputs), "resolved": False}
    solution = args.source_worktree / "solution.py"
    if io.get("fn_name"):
        report["reason"] = "call-based APPS task is unsupported by the stdin/stdout verifier"
    elif not solution.is_file() or len(inputs) != len(outputs):
        report["reason"] = "missing solution or malformed test cases"
    else:
        cases = []
        for index, (stdin, expected) in enumerate(zip(inputs, outputs)):
            result = subprocess.run(
                [args.python, str(solution)], cwd=args.source_worktree,
                input=_stdio_text(stdin), capture_output=True, text=True, timeout=30, check=False,
            )
            actual = result.stdout.strip()
            wanted = _stdio_text(expected).strip()
            cases.append({"index": index, "returncode": result.returncode,
                          "passed": result.returncode == 0 and actual == wanted,
                          "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]})
            if result.returncode != 0 or actual != wanted:
                break
        report.update(
            {
                "resolved": len(cases) == len(inputs)
                and all(item["passed"] for item in cases),
                "cases": cases,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if report["resolved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
