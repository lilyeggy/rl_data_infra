"""ARCHIVED EXAMPLE: V3 synthetic task suite used by historical experiments.

This is not a production benchmark or a source of trainable evidence.

The 15 graded tasks were shared by real Pi, micro-harness,
data production, and evaluation.

Difficulty axes (graded so small models degrade predictably):
- record count            : 3..15 records per task file
- distractor statuses     : files may contain SUCCESS/ERROR/UNKNOWN; only
                            records with status exactly FAILURE count
- discovery noise         : extra non-task files in the workspace

Splits: TRAIN (SFT/GRPO data), DEV (model selection), EVAL (held-out).
Tasks 1-5 preserve the historical totals from V2/V2.1 so earlier fixtures and
training artifacts stay comparable; tasks 6-15 add the graded difficulty.
"""

from __future__ import annotations

TRAIN_TASKS = (1, 2, 3, 4, 5, 6, 7, 8)
DEV_TASKS = (9, 10, 11)
EVAL_TASKS = (12, 13, 14, 15)
ALL_TASKS = TRAIN_TASKS + DEV_TASKS + EVAL_TASKS

# failures  : number of records with status FAILURE (the expected answer)
# total     : total number of records in the task file
# distractors: file mixes SUCCESS/ERROR/UNKNOWN; only FAILURE counts
# extra_files: number of non-task noise files in the workspace
SPEC = {
    1: {"failures": 2, "total": 4},
    2: {"failures": 2, "total": 4},
    3: {"failures": 3, "total": 5},
    4: {"failures": 1, "total": 3},
    5: {"failures": 4, "total": 6},
    6: {"failures": 3, "total": 8},
    7: {"failures": 5, "total": 10, "distractors": True},
    8: {"failures": 4, "total": 10, "distractors": True},
    9: {"failures": 4, "total": 10, "distractors": True},
    10: {"failures": 7, "total": 14, "distractors": True},
    11: {"failures": 5, "total": 12, "distractors": True},
    12: {"failures": 8, "total": 15, "distractors": True},
    13: {"failures": 6, "total": 15},
    14: {"failures": 5, "total": 12, "distractors": True, "extra_files": 2},
    15: {"failures": 9, "total": 15, "distractors": True, "extra_files": 2},
}

STATUS_POOL = ("FAILURE", "SUCCESS", "ERROR", "UNKNOWN")

EXTRA_FILE_TEMPLATES = {
    "README.md": "workspace notes for task automation demo",
    "data-{task}.csv": "id,status\n1,SUCCESS\n2,FAILURE\n",
    "archive-{task}.json": '{"kind": "archive", "records": []}',
}


def task_spec(task_index: int) -> dict:
    if task_index not in SPEC:
        raise KeyError(f"task-{task_index} not in V3 suite")
    return SPEC[task_index]


def failure_count(task_index: int) -> int:
    return SPEC[task_index]["failures"]


def expected_answer(task_index: int) -> dict:
    return {
        "task_id": f"task-{task_index}",
        "status": "SUCCESS",
        "failure_count": SPEC[task_index]["failures"],
    }


def task_file_payload(task_index: int) -> dict:
    """Deterministic payload: first N records FAILURE, remainder cycle through
    the other statuses (SUCCESS/ERROR/UNKNOWN when distractors are enabled)."""
    spec = SPEC[task_index]
    statuses = ("FAILURE", "SUCCESS") if not spec.get("distractors") else STATUS_POOL
    non_failure = [s for s in statuses if s != "FAILURE"]
    records = []
    for i in range(spec["total"]):
        status = "FAILURE" if i < spec["failures"] else non_failure[i % len(non_failure)]
        records.append({"id": chr(97 + i), "status": status})
    return {"task_id": f"task-{task_index}", "records": records}


def extra_files(task_index: int) -> dict[str, str]:
    spec = SPEC[task_index]
    out: dict[str, str] = {}
    templates = list(EXTRA_FILE_TEMPLATES.items())
    for i in range(spec.get("extra_files", 0)):
        name, body = templates[i % len(templates)]
        out[name.format(task=task_index)] = body.format(task=task_index)
    return out


def expected_counts_map(task_indices) -> dict[int, int]:
    return {t: SPEC[t]["failures"] for t in task_indices}
