#!/usr/bin/env python3
"""Fail-closed BigCodeBench verifier running generated code in Docker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--source-worktree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--verifier-entrypoint", type=Path, required=True)
    args = parser.parse_args()
    task = json.loads(args.manifest.read_text())["tasks"][args.task_id]
    solution = args.source_worktree / "solution.py"
    report: dict[str, object] = {"task_id": args.task_id, "resolved": False}
    if not solution.is_file():
        report["reason"] = "solution.py missing"
    else:
        with tempfile.TemporaryDirectory(prefix="bigcodebench-verifier-") as temporary:
            root = Path(temporary)
            (root / "task.json").write_text(json.dumps(task))
            output_dir = root / "output"; output_dir.mkdir()
            container_name = "agent-data-plane-bcb-" + hashlib.sha256(
                f"{args.task_id}:{solution.resolve()}".encode()
            ).hexdigest()[:20]
            command = [
                "docker", "run", "--rm", "--network", "none", "--read-only", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--pids-limit", "128", "--memory", "4g",
                "--name", container_name, "--user", f"{os.getuid()}:{os.getgid()}",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=1g",
                "-v", f"{root / 'task.json'}:/input/task.json:ro",
                "-v", f"{solution.resolve()}:/input/solution.py:ro",
                "-v", f"{args.verifier_entrypoint.resolve()}:/input/verifier.py:ro",
                "-v", f"{output_dir}:/output:rw", args.image, "python3", "/input/verifier.py",
                "--task", "/input/task.json", "--solution", "/input/solution.py", "--output", "/output/result.json",
            ]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
            except subprocess.TimeoutExpired as exc:
                # Docker's client timeout does not guarantee that the workload
                # has exited.  Remove it explicitly before reporting failure.
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)
                report.update({"reason": "verifier container timeout", "stderr": str(exc.stderr or "")[-8000:]})
                result = None
            result_path = output_dir / "result.json"
            if result_path.is_file():
                report = json.loads(result_path.read_text())
            elif result is not None:
                report.update({"reason": "verifier container produced no report", "returncode": result.returncode,
                               "stderr": result.stderr[-8000:]})
            report["container_name"] = container_name
            report["container_returncode"] = result.returncode if result is not None else 124
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if report.get("resolved") else 1


if __name__ == "__main__":
    raise SystemExit(main())
