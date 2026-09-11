#!/usr/bin/env python3
"""Curriculum & Variance-Guaranteed Task Selector for Agentic RL.

Selects balanced, solvable yet non-trivial APPS tasks from train.manifest.json
for RL exploration, and reserves a distinct holdout evaluation split.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


def select_curriculum_tasks(
    manifest_path: Path,
    num_train: int = 80,
    num_eval: int = 50,
    seed: int = 42,
    intro_ratio: float = 0.35,
    interview_ratio: float = 0.55,
    comp_ratio: float = 0.10,
    exclude_tasks: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Select training curriculum and holdout evaluation tasks."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = data.get("tasks", {})

    exclude = exclude_tasks or set()

    by_diff = defaultdict(list)
    for tid, t in tasks.items():
        if tid in exclude:
            continue
        io = t.get("input_output", {})
        if io.get("fn_name"):
            continue
        inputs = io.get("inputs", [])
        outputs = io.get("outputs", [])
        if len(inputs) == 0 or len(inputs) != len(outputs):
            continue
        q_len = len(t.get("question", ""))
        # Filter out extremely long or empty questions
        if q_len < 80 or q_len > 3500:
            continue
        diff = t.get("difficulty", "interview")
        by_diff[diff].append(tid)

    rng = random.Random(seed)
    for diff in by_diff:
        rng.shuffle(by_diff[diff])

    target_intro = int(num_train * intro_ratio)
    target_comp = int(num_train * comp_ratio)
    target_interview = num_train - target_intro - target_comp

    train_tasks: list[str] = []
    train_tasks.extend(by_diff["introductory"][:target_intro])
    train_tasks.extend(by_diff["interview"][:target_interview])
    train_tasks.extend(by_diff["competition"][:target_comp])

    # If any difficulty ran short, fill from interview
    if len(train_tasks) < num_train:
        remaining = [
            t for t in by_diff["interview"][target_interview:]
            if t not in train_tasks
        ]
        train_tasks.extend(remaining[: num_train - len(train_tasks)])

    train_set = set(train_tasks)

    # Select Holdout Eval tasks strictly disjoint from train_set and exclude
    eval_tasks: list[str] = []
    avail_intro = [t for t in by_diff["introductory"] if t not in train_set]
    avail_interview = [t for t in by_diff["interview"] if t not in train_set]
    avail_comp = [t for t in by_diff["competition"] if t not in train_set]

    eval_intro = int(num_eval * 0.4)
    eval_interview = num_eval - eval_intro

    eval_tasks.extend(avail_intro[:eval_intro])
    eval_tasks.extend(avail_interview[:eval_interview])

    if len(eval_tasks) < num_eval:
        remaining = [t for t in avail_interview[eval_interview:] if t not in eval_tasks]
        eval_tasks.extend(remaining[: num_eval - len(eval_tasks)])

    return train_tasks, eval_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Path to APPS train.manifest.json")
    parser.add_argument("--num-train", type=int, default=80, help="Number of training tasks")
    parser.add_argument("--num-eval", type=int, default=50, help="Number of holdout eval tasks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--exclude", nargs="*", default=None, help="Task IDs to exclude")
    parser.add_argument("--out-train", required=True, type=Path, help="Output JSON for train tasks")
    parser.add_argument("--out-eval", required=True, type=Path, help="Output JSON for eval tasks")
    args = parser.parse_args()

    exclude_set = set(args.exclude) if args.exclude else set()
    train_tasks, eval_tasks = select_curriculum_tasks(
        args.manifest,
        num_train=args.num_train,
        num_eval=args.num_eval,
        seed=args.seed,
        exclude_tasks=exclude_set,
    )

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    args.out_eval.parent.mkdir(parents=True, exist_ok=True)

    args.out_train.write_text(json.dumps(train_tasks, indent=2) + "\n", encoding="utf-8")
    args.out_eval.write_text(json.dumps(eval_tasks, indent=2) + "\n", encoding="utf-8")

    print(f"Selected {len(train_tasks)} curriculum train tasks -> {args.out_train}")
    print(f"Selected {len(eval_tasks)} holdout eval tasks -> {args.out_eval}")


if __name__ == "__main__":
    main()
