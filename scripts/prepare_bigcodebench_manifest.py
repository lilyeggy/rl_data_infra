#!/usr/bin/env python3
"""Freeze a BigCodeBench release into the data-plane task-manifest contract."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")
    raw = args.input.read_bytes()
    rows = [json.loads(line) for line in gzip.decompress(raw).splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    required = {
        "task_id", "instruct_prompt", "complete_prompt", "code_prompt", "test", "entry_point"
    }
    if not rows or any(not required.issubset(row) for row in rows):
        raise ValueError("input is not a complete BigCodeBench release")
    manifest = {
        "schema_version": "bigcodebench-task-manifest/v1",
        "source": {
            "benchmark": "BigCodeBench-Hard",
            "release_file": args.input.name,
            "release_sha256": hashlib.sha256(raw).hexdigest(),
        },
        "tasks": {row["task_id"]: {key: row[key] for key in required} for row in rows},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"tasks": len(rows), "sha256": manifest["source"]["release_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
