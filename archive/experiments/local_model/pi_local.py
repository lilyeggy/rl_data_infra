"""ARCHIVED: run the local student model through the real Pi harness.

This replaces the micro-harness for student rollouts: the local Qwen model is
served by ``openai_server.py`` and Pi drives it via the ``local-qwen`` provider,
executing real tools and emitting real NDJSON traces — exactly like the teacher.

Used by:
  - P2 eval   : greedy, score all 15 tasks through real Pi
  - P3 GRPO   : sampling, produce on-policy rollouts + verifier rewards

Run on the server (PYTHONPATH=/root/agentic-rl).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.capture.pi_adapter import PiRunConfig, dump_pi_ndjson, read_pi_ndjson
from src.capture.pi_runner import run_pi_process
from src.real_pi_v2 import verify_reference_answer
from src.task_suite import (
    ALL_TASKS,
    expected_counts_map,
    extra_files,
    task_file_payload,
)

PROVIDER = "local-qwen"
MODEL = "qwen2.5-7b-instruct"
EXPECTED = expected_counts_map(ALL_TASKS)

# Environment-accurate student policy. In this real Pi deployment the `find`
# tool is backed by `fd`, which is unavailable; `ls` works. The teacher
# (deepseek) recovered by pivoting to ls; the student is given an
# environment-accurate instruction (list the directory) rather than a tool
# (find) that is broken here. Same output contract and tool budget as the
# teacher arms.
STUDENT_POLICY = (
    "Harness policy: if reading the requested file fails because it does not "
    "exist, list this directory to find the unique file whose name matches "
    "task-*.json, then read that file. If zero or multiple candidates are "
    "found, stop. Never retry the missing path."
)


def build_student_prompt(task_index: int) -> str:
    return (
        f"Start by reading missing-{task_index}.json. "
        f"{STUDENT_POLICY} "
        "Return only JSON (no markdown, no code fences) with fields "
        "task_id, status, failure_count. "
        "status must be exactly the string SUCCESS when you successfully read "
        "the task file, regardless of the records inside it. "
        "failure_count is the number of FAILURE records inside that task file. "
        "Use at most 4 tool calls total."
    )


def setup_workspace(ws_root: Path, task_index: int) -> Path:
    cwd = ws_root / f"task-{task_index}"
    cwd.mkdir(parents=True, exist_ok=True)
    # clean prior task files so reruns are deterministic
    for f in cwd.glob("*.json"):
        f.unlink()
    (cwd / f"task-{task_index}.json").write_text(
        json.dumps(task_file_payload(task_index), ensure_ascii=False) + "\n"
    )
    for name, body in extra_files(task_index).items():
        (cwd / name).write_text(body)
    return cwd


def _read_task_file_ok(records, task_index: int) -> bool:
    """Did the agent successfully read the task-N.json file? Robust to records
    whose fields were frozen into mappingproxy by secret redaction (never
    json.dumps the whole record). The read result's text begins with the task
    file's JSON, so we look for the task_id marker."""
    marker = f'"task_id": "task-{task_index}"'
    for rec in records:
        if rec.get("type") != "tool_execution_end":
            continue
        if rec.get("isError"):
            continue
        if rec.get("toolName") != "read":
            continue
        res = rec.get("result")
        try:
            content = res.get("content") if hasattr(res, "get") else None
            if isinstance(content, (list, tuple)):
                for part in content:
                    text = part.get("text") if hasattr(part, "get") else None
                    if isinstance(text, str) and marker in text:
                        return True
        except Exception:  # noqa: BLE001
            continue
    return False


def run_task(
    task_index: int,
    ws_root: Path,
    *,
    arm: str = "candidate",
    timeout_seconds: float = 180,
    raw_output_dir: Path | None = None,
    tag: str = "local",
):
    """Run one task through real Pi with the local model. Returns
    (records, reward, declaration). Reward shaping: 1.0 exact, 0.5 if the task
    file was read but the final answer was wrong, else 0.0."""
    cwd = setup_workspace(ws_root, task_index)
    prompt = build_student_prompt(task_index)
    capture = run_pi_process(
        config=PiRunConfig(model=MODEL, provider=PROVIDER, thinking="minimal"),
        prompt=prompt,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
    )
    records, issues = read_pi_ndjson(capture.stdout)
    if raw_output_dir is not None:
        # Best-effort raw capture: observation must never break task scoring.
        try:
            raw_output_dir.mkdir(parents=True, exist_ok=True)
            (raw_output_dir / f"{tag}-{task_index}.ndjson").write_text(
                dump_pi_ndjson(records)
            )
        except Exception:  # noqa: BLE001
            pass
    declaration, _ev = verify_reference_answer(
        records, task_index=task_index, expected_counts=EXPECTED
    )
    exact = float(declaration.score)  # 1.0 or 0.0
    if exact == 1.0:
        reward = 1.0
    elif _read_task_file_ok(records, task_index):
        reward = 0.5
    else:
        reward = 0.0
    return records, reward, declaration


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="/root/rivermind-data/pi-local-ws")
    parser.add_argument("--tasks", default=",".join(str(t) for t in ALL_TASKS))
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--raw-dir", default="/root/rivermind-data/pi-local-raw")
    args = parser.parse_args()

    ws = Path(args.workspace)
    raw = Path(args.raw_dir)
    tasks = [int(t) for t in args.tasks.split(",")]
    results = []
    for t in tasks:
        try:
            records, reward, decl = run_task(
                t, ws, timeout_seconds=args.timeout, raw_output_dir=raw
            )
            results.append((t, reward))
            print(f"task {t}: reward={reward} status={decl.task_status}", flush=True)
        except Exception as exc:  # noqa: BLE001
            results.append((t, 0.0))
            print(f"task {t}: ERROR {exc}", flush=True)
    full = sum(1 for _, r in results if r == 1.0)
    print(f"\nPI_LOCAL_EVAL full={full}/{len(tasks)} -> " +
          json.dumps({t: r for t, r in results}))


if __name__ == "__main__":
    main()
