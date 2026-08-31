#!/usr/bin/env python3
"""Convert the official HumanEval JSONL into the local executable-task manifest."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = gzip.open(args.input, "rt", encoding="utf-8") if args.input.suffix == ".gz" else args.input.open(encoding="utf-8")
    with raw as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    manifest = {}
    for index, row in enumerate(rows, start=1):
        entry = row["entry_point"]
        manifest[str(index)] = {
            "task_id": row["task_id"],
            "text": row["prompt"],
            "test_list": [row["test"] + f"\ncheck({entry})"],
            "challenge_test_list": [],
            "test_setup_code": "",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False) + "\n")
    print(json.dumps({"tasks": len(manifest), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
