#!/usr/bin/env python3
"""Freeze only unattempted stdin APPS tasks into a new immutable manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    completed = {
        json.loads(path.read_text(encoding="utf-8"))["task_id"]
        for path in args.run_root.glob("*/summary.json")
    }
    tasks = {
        task_id: task
        for task_id, task in manifest["tasks"].items()
        if task_id not in completed and not task["input_output"].get("fn_name")
    }
    manifest["tasks"] = tasks
    manifest["schema_version"] = "apps-task-manifest/remaining-stdin-v2"
    manifest["source"] = {
        **manifest["source"],
        "parent_manifest": str(args.manifest),
        "excluded_completed_count": len(completed),
        "filter": "unattempted and input_output.fn_name is absent",
    }
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"tasks": len(tasks), "excluded_completed": len(completed)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
