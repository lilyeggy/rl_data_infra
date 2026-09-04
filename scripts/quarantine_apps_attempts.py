#!/usr/bin/env python3
"""Move retryable APPS infrastructure failures aside without deleting evidence."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--quarantine-root", type=Path, required=True)
    args = parser.parse_args()

    if args.quarantine_root.exists():
        raise ValueError(f"refusing to reuse {args.quarantine_root}")

    candidates: list[tuple[Path, str, str]] = []
    for run_dir in sorted(args.run_root.glob("apps-train-*-attempt-1")):
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            task_id = run_dir.name[: -len("-attempt-1")]
            candidates.append((run_dir, task_id, "incomplete"))
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("execution_validity") == "INFRA_INVALID"
            and summary.get("pi_returncode") == 1
            and summary.get("model_call_count") == 0
        ):
            candidates.append((run_dir, summary["task_id"], "node-launch-failure"))

    run_destination = args.quarantine_root / "runs"
    workspace_destination = args.quarantine_root / "workspaces"
    run_destination.mkdir(parents=True)
    workspace_destination.mkdir(parents=True)
    records = []
    for run_dir, task_id, reason in candidates:
        workspace = args.workspace_root / task_id
        shutil.move(str(run_dir), run_destination / run_dir.name)
        if workspace.exists():
            shutil.move(str(workspace), workspace_destination / workspace.name)
        records.append({"task_id": task_id, "reason": reason})

    (args.quarantine_root / "manifest.json").write_text(
        json.dumps({"count": len(records), "records": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"quarantined": len(records), "destination": str(args.quarantine_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
