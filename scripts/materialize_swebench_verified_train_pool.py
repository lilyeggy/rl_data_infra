#!/usr/bin/env python3
"""Create a deterministic, repository-balanced TRAIN-only SWE-bench pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-per-repo", type=int, default=12)
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--exclude-json", type=Path)
    args = parser.parse_args()
    if args.limit <= 0 or args.max_per_repo <= 0:
        raise ValueError("limit and max-per-repo must be positive")
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")

    import pandas as pd

    records = pd.read_parquet(args.dataset).to_dict(orient="records")
    excluded = set(args.exclude)
    if args.exclude_json:
        payload = json.loads(args.exclude_json.read_text())
        excluded.update(payload.keys() if isinstance(payload, dict) else payload)
    eligible = []
    for row in records:
        instance_id = row.get("instance_id")
        repo = row.get("repo")
        if not isinstance(instance_id, str) or instance_id in excluded:
            continue
        if not isinstance(repo, str) or not row.get("problem_statement"):
            continue
        if not _as_list(row.get("FAIL_TO_PASS")):
            continue
        row["FAIL_TO_PASS"] = _as_list(row.get("FAIL_TO_PASS"))
        row["PASS_TO_PASS"] = _as_list(row.get("PASS_TO_PASS"))
        row["test_complexity"] = len(row["FAIL_TO_PASS"]) + len(row["PASS_TO_PASS"])
        eligible.append(row)
    eligible.sort(key=lambda row: (row["test_complexity"], len(row["problem_statement"]), row["instance_id"]))

    chosen: list[dict[str, Any]] = []
    by_repo: dict[str, int] = {}
    while eligible and len(chosen) < args.limit:
        progressed = False
        for row in tuple(eligible):
            repo = row["repo"]
            if by_repo.get(repo, 0) >= args.max_per_repo:
                continue
            eligible.remove(row)
            row.pop("test_complexity")
            chosen.append(row)
            by_repo[repo] = by_repo.get(repo, 0) + 1
            progressed = True
            if len(chosen) == args.limit:
                break
        if not progressed:
            break
    if not chosen:
        raise ValueError("no eligible TRAIN tasks")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({row["instance_id"]: row for row in chosen}, indent=2) + "\n")
    print(json.dumps({"selected": len(chosen), "repos": by_repo, "excluded": len(excluded)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
