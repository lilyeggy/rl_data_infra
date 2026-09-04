#!/usr/bin/env python3
"""Create an immutable stdin-only APPS manifest from a mixed APPS manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")

    manifest = json.loads(args.input.read_text(encoding="utf-8"))
    original_tasks = manifest["tasks"]
    manifest["tasks"] = {
        task_id: task
        for task_id, task in original_tasks.items()
        if not task["input_output"].get("fn_name")
    }
    manifest["source"] = {
        **manifest["source"],
        "parent_manifest": str(args.input),
        "filter": "input_output.fn_name is absent",
        "parent_task_count": len(original_tasks),
    }
    manifest["schema_version"] = "apps-task-manifest/stdin-v2"
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"tasks": len(manifest["tasks"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
