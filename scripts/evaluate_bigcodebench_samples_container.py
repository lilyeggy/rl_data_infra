#!/usr/bin/env python3
"""Evaluate standard BigCodeBench samples through the restricted verifier."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--workspaces", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--verifier", type=Path, required=True)
    p.add_argument("--entrypoint", type=Path, required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    tasks = manifest["tasks"]
    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    args.workspaces.mkdir(parents=True, exist_ok=True); args.results.mkdir(parents=True, exist_ok=True)

    def run(sample: dict) -> dict:
        task_id = sample["task_id"]
        safe = task_id.replace("/", "_")
        workspace = args.workspaces / safe
        output = args.results / f"{safe}.json"
        if output.exists():
            return json.loads(output.read_text())
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "solution.py").write_text(sample["solution"], encoding="utf-8")
        cmd = [sys.executable, str(args.verifier), "--manifest", str(args.manifest),
               "--task-id", task_id, "--source-worktree", str(workspace),
               "--output", str(output), "--image", args.image,
               "--verifier-entrypoint", str(args.entrypoint)]
        completed = subprocess.run(cmd, cwd=args.manifest.parent.parent.parent,
                                   capture_output=True, text=True, check=False)
        if not output.exists():
            output.write_text(json.dumps({"task_id": task_id, "resolved": False,
                                          "execution_validity": "INFRA_INVALID",
                                          "error": completed.stderr[-4000:]}))
        return json.loads(output.read_text())

    # Preserve sample order in the aggregate while allowing verifier parallelism.
    results = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run, sample): index for index, sample in enumerate(samples)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    payload = {"schema_version": "bigcodebench-container-eval/v1", "count": len(results),
               "resolved": sum(item.get("resolved") is True for item in results),
               "infra_invalid": sum(item.get("execution_validity") == "INFRA_INVALID" for item in results),
               "results": results}
    (args.results / "aggregate.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("count", "resolved", "infra_invalid")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
