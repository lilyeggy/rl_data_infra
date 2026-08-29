#!/usr/bin/env python3
"""Rank small reproducible SWE-bench candidates from Polar's cached dataset.

The script is read-only. It does not download datasets, build images, or submit
rollouts. Its output is a deterministic shortlist for human/agent preflight;
runtime image availability and baseline replay still decide whether a task is usable.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--distinct-repos",
        action="store_true",
        help="Prefer at most one candidate from each repository.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _list_field(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, list) else None
    return None


def rank_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        instance_id = row.get("instance_id")
        base_commit = row.get("base_commit")
        problem = row.get("problem_statement")
        fail_to_pass = _list_field(row.get("FAIL_TO_PASS"))
        pass_to_pass = _list_field(row.get("PASS_TO_PASS"))
        repo = row.get("repo")
        if not isinstance(instance_id, str) or not instance_id.strip():
            continue
        if not isinstance(base_commit, str) or not FULL_SHA.fullmatch(base_commit):
            continue
        if not isinstance(problem, str) or not problem.strip():
            continue
        if fail_to_pass is None or not fail_to_pass or pass_to_pass is None:
            continue
        if not isinstance(repo, str) or not repo.strip():
            owner, separator, repo_issue = instance_id.partition("__")
            repo_name, issue_separator, _ = repo_issue.rpartition("-")
            repo = (
                f"{owner}/{repo_name}"
                if separator and issue_separator and owner and repo_name
                else instance_id
            )

        test_count = len(fail_to_pass) + len(pass_to_pass)
        problem_chars = len(problem)
        # Fewer tests dominate. Shorter prompts break ties; ID makes ordering stable.
        score = test_count * 1_000_000 + min(problem_chars, 999_999)
        candidates.append(
            {
                "instance_id": instance_id,
                "repo": repo,
                "base_commit": base_commit,
                "fail_to_pass_count": len(fail_to_pass),
                "pass_to_pass_count": len(pass_to_pass),
                "problem_chars": problem_chars,
                "ranking_score": score,
            }
        )
    return sorted(candidates, key=lambda item: (item["ranking_score"], item["instance_id"]))


def select_candidates(
    rows: list[dict[str, Any]],
    limit: int,
    *,
    distinct_repos: bool = False,
) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    ranked = rank_candidates(rows)
    if not distinct_repos:
        return ranked[:limit]
    selected: list[dict[str, Any]] = []
    seen_repos: set[str] = set()
    for candidate in ranked:
        if candidate["repo"] in seen_repos:
            continue
        selected.append(candidate)
        seen_repos.add(candidate["repo"])
        if len(selected) == limit:
            break
    return selected


def main() -> int:
    args = parse_args()
    try:
        payload = json.loads(args.dataset_json.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise ValueError("dataset JSON must be an array of objects")
        selected = select_candidates(
            payload,
            args.limit,
            distinct_repos=args.distinct_repos,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"FAIL candidate_selection: {exc}")
        return 1
    if not selected:
        print("FAIL candidate_selection: no valid candidates")
        return 1

    rendered = json.dumps(selected, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            print(f"FAIL candidate_selection: refusing to overwrite {args.output}")
            return 1
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
