#!/usr/bin/env python3
"""Fail-closed verifier for one MBPP task workspace."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--source-worktree", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task = json.loads(args.manifest.read_text())[args.task_id]
    solution = args.source_worktree / "solution.py"
    tests = [*task.get("test_list", []), *task.get("challenge_test_list", [])]
    report: dict[str, object] = {"task_id": args.task_id, "tests": tests}
    if not solution.is_file():
        report.update({"resolved": False, "reason": "solution.py missing"})
    else:
        code = "\n".join([
            task.get("test_setup_code", ""),
            f"exec(compile(open({str(solution)!r}, encoding='utf-8').read(), {str(solution)!r}, 'exec'))",
            *tests,
        ])
        with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as handle:
            handle.write(code)
            test_file = handle.name
        try:
            result = subprocess.run([args.python, test_file], cwd=args.source_worktree,
                                    capture_output=True, text=True, timeout=120, check=False)
            report.update({"resolved": result.returncode == 0,
                           "returncode": result.returncode,
                           "stdout": result.stdout[-8000:], "stderr": result.stderr[-8000:]})
        finally:
            Path(test_file).unlink(missing_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 0 if report.get("resolved") else 1


if __name__ == "__main__":
    raise SystemExit(main())
