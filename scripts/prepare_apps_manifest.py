#!/usr/bin/env python3
"""Freeze the public APPS train split into a small immutable task manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")
    rows = []
    paths = sorted(args.input_root.rglob("problem.json"))
    jsonl_paths = sorted(args.input_root.glob("*.jsonl"))
    records = []
    for path in paths:
        records.append((path, json.loads(path.read_text(encoding="utf-8"))))
    for path in jsonl_paths:
        with path.open(encoding="utf-8") as handle:
            records.extend((path, json.loads(line)) for line in handle if line.strip())
    for path, raw in records:
        io = raw.get("input_output")
        if isinstance(io, str):
            try:
                io = json.loads(io)
            except json.JSONDecodeError:
                continue
        if not isinstance(raw.get("question"), str) or not isinstance(io, dict):
            continue
        rows.append({
            "task_id": f"apps-train-{len(rows):06d}",
            "question": raw["question"],
            "input_output": io,
            "difficulty": raw.get("difficulty"),
            "source_file": str(path.relative_to(args.input_root)),
        })
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("no APPS problem.json or JSONL records found")
    payload = {
        "schema_version": "apps-task-manifest/v1",
        "source": {
            "dataset": "codeparrot/apps",
            "split": "train",
            "input_root": args.input_root.name,
            "files_sha256": hashlib.sha256(
                "\n".join(item["source_file"] for item in rows).encode()
            ).hexdigest(),
        },
        "tasks": {item["task_id"]: item for item in rows},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"tasks": len(rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
