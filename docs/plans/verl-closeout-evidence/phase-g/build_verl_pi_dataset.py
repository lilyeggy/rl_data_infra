#!/usr/bin/env python3
"""Build the jsonl prompt set the verl trainer feeds to the Pi agent loop.

One row per task. The trainer expands each row into ``rollout.n`` samples
sharing a uid, so a row is a *group*: n independent Pi episodes of the same
task, which is exactly the GRPO grouping the closeout specifies.

Row fields the framework and our loop rely on:

* ``prompt``        chat messages; the framework needs the column to exist.
                    Our loop builds Pi's actual instruction from extra_info.
* ``data_source``   reward routing key (kept for the framework's reward path,
                    which we bypass because the loop sets reward_score).
* ``reward_model``  ``ground_truth``; same story.
* ``extra_info``     task_id / task_slug / task_prompt / group_id, plus
                    ``index`` (the dataset default is 0 for every row, which
                    would collide, so it is set explicitly here).
* ``agent_name``    selects our registered agent loop.

``uid`` is deliberately NOT written: ``RayPPOTrainer.fit`` creates a fresh uid
per row before expanding by n, so the n episodes of a row already share a group.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-ids", required=True, help="comma-separated, e.g. Mbpp/118")
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--agent-name", default="pi_agent")
    args = parser.parse_args()

    from evalplus.data import get_mbpp_plus

    problems = get_mbpp_plus()
    task_ids = [item.strip() for item in args.task_ids.split(",") if item.strip()]
    rows = []
    for index, task_id in enumerate(task_ids):
        problem = problems.get(task_id)
        if problem is None:
            raise SystemExit(f"task {task_id} is not in evalplus MBPP+")
        entry_point = problem["entry_point"]
        task_prompt = (
            "Task %s. Implement the function '%s' in solution.py.\n%s"
            % (task_id, entry_point, problem["prompt"])
        )
        slug = task_id.replace("/", "-")
        rows.append({
            "prompt": [
                {"role": "system", "content": "You are a coding agent working in a workspace."},
                {"role": "user", "content": task_prompt},
            ],
            "data_source": "mbpp",
            "reward_model": {
                "ground_truth": {"task_id": task_id, "entry_point": entry_point}
            },
            "extra_info": {
                "index": index,
                "task_id": task_id,
                "task_slug": slug,
                "task_prompt": task_prompt,
                "group_id": args.group_id,
            },
            "agent_name": args.agent_name,
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "rows": len(rows),
                      "task_ids": task_ids, "group_id": args.group_id}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
