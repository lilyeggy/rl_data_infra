"""ARCHIVED: generate SFT trajectories by driving the 14B through the REAL Pi harness.

This uses the real Pi runner (run_pi_process) — NOT the custom harness.py — so
every trajectory is a canonical, auditable Pi record stream. The only difference
from the adversarial self-evolve baseline is a *guiding* prompt that scaffolds
recovery + output format, to elicit successful trajectories from a model that
otherwise gives up (zero-shot 0/15). Successful trajectories become SFT data;
at SFT time the input is the NEUTRAL prompt so the model internalizes recovery
(STaR-style hint bootstrapping) rather than depending on the hint.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping as AbcMapping
from pathlib import Path

from src.capture.pi_adapter import PiRunConfig, dump_pi_ndjson, read_pi_ndjson
from src.capture.pi_runner import run_pi_process
from src.real_pi_v2 import verify_reference_answer
from src.task_suite import ALL_TASKS, expected_counts_map, extra_files, task_file_payload

PROVIDER = "local-qwen"
MODEL = os.environ.get("LOCAL_MODEL_NAME", "qwen2.5-coder-14b-instruct")
EXPECTED = expected_counts_map(ALL_TASKS)
ROLLOUT_LOG = os.environ.get("ROLLOUT_LOG", "/root/rivermind-data/qwen-rollout-log.jsonl")


def _log_offset() -> int:
    if not Path(ROLLOUT_LOG).exists():
        return 0
    with open(ROLLOUT_LOG) as f:
        return sum(1 for _ in f)


def _read_log_from(offset: int):
    """New server-logged turns (prompt_ids/completion_ids) since offset. These are
    the exact token sequences the served model produced -> SFT on them is
    guaranteed consistent with inference (same chat template + tools)."""
    entries = []
    if not Path(ROLLOUT_LOG).exists():
        return entries
    with open(ROLLOUT_LOG) as f:
        for i, line in enumerate(f):
            if i >= offset and line.strip():
                entries.append(json.loads(line))
    return entries

TOOL_BUDGET = 8

# Guiding prompt for DATA GENERATION ONLY (elicits success through real Pi).
# It scaffolds the two things the 14B failed at zero-shot: (1) recover from the
# missing-file probe by finding the real task file, (2) emit raw JSON (no code
# fences). At SFT time we pair the trajectory with the NEUTRAL prompt instead.
GUIDED_PROMPT = (
    "You are auditing JSON record files in the current working directory.\n"
    "The file missing-__TI__.json does NOT exist — that is expected. The real task "
    "file IS present in this directory; find it first (use the ls or find tool), "
    "then read it.\n"
    "Count the records whose \"status\" is exactly \"FAILURE\" (ignore SUCCESS, "
    "ERROR, UNKNOWN). Only the real task file's records count; ignore noise files. "
    "Count precisely — you may use the bash or grep tool to count exactly rather "
    "than counting in your head.\n"
    f"Use at most {TOOL_BUDGET} tool calls. Then output the final answer as RAW "
    "JSON ONLY — no markdown, no code fences, no explanation. Exactly this shape: "
    "{\"task_id\": \"<the real file's task_id>\", \"status\": \"SUCCESS\", "
    "\"failure_count\": <number>}."
)

# The NEUTRAL prompt used as the SFT *input* (identical to the self-evolve
# baseline), so the model learns to recover without the scaffold.
NEUTRAL_PROMPT = (
    "Start by reading missing-__TI__.json. "
    "Return only JSON (no markdown, no code fences) with fields "
    "task_id, status, failure_count. "
    "status must be exactly the string SUCCESS when you have successfully "
    "read the task file, regardless of the records inside it. "
    "failure_count is the number of records whose status is exactly FAILURE; "
    "ignore SUCCESS, ERROR, UNKNOWN."
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def setup_workspace(ws_root: Path, task_index: int) -> Path:
    cwd = ws_root / f"task-{task_index}"
    cwd.mkdir(parents=True, exist_ok=True)
    for f in cwd.glob("*.json"):
        f.unlink()
    (cwd / f"task-{task_index}.json").write_text(
        json.dumps(task_file_payload(task_index), ensure_ascii=False) + "\n"
    )
    for name, body in extra_files(task_index).items():
        (cwd / name).write_text(body)
    return cwd


def _final_text(records) -> str | None:
    final = None
    for r in records:
        if r.get("type") != "message_end":
            continue
        m = r.get("message")
        if not isinstance(m, AbcMapping):
            continue
        if m.get("role") != "assistant" or m.get("stopReason") != "stop":
            continue
        content = m.get("content")
        texts = []
        if isinstance(content, (list, tuple)):
            for it in content:
                # NOTE: redaction freezes dicts->mappingproxy, so use Mapping not dict
                if isinstance(it, AbcMapping) and it.get("type") == "text" and isinstance(it.get("text"), str):
                    texts.append(it["text"])
        elif isinstance(content, str):
            texts.append(content)
        if texts:
            final = "".join(texts)
    return final


def _parse_answer_lenient(records) -> dict | None:
    """Parse the final answer, tolerating code fences (data-collection only;
    the strict frozen verifier stays fence-intolerant for scoring)."""
    text = _final_text(records)
    if not text:
        return None
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        return None


def _correct(records, task_index: int) -> bool:
    ans = _parse_answer_lenient(records)
    if not isinstance(ans, dict):
        return False
    return (
        ans.get("task_id") == f"task-{task_index}"
        and ans.get("status") == "SUCCESS"
        and ans.get("failure_count") == EXPECTED[task_index]
    )


def run_task(task_index: int, ws_root: Path, raw_dir: Path, timeout: float):
    cwd = setup_workspace(ws_root, task_index)
    config = PiRunConfig(
        model=MODEL, provider=PROVIDER, thinking="minimal",
        tools=("read", "grep", "find", "ls", "bash"),
    )
    off = _log_offset()
    cap = run_pi_process(
        config=config, prompt=GUIDED_PROMPT.replace("__TI__", str(task_index)), cwd=cwd, timeout_seconds=timeout
    )
    records, issues = read_pi_ndjson(cap.stdout)
    turns = _read_log_from(off)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"gen-{task_index}.ndjson").write_text(dump_pi_ndjson(records))
    ok = _correct(records, task_index)
    return records, ok, turns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--workspace", default="/root/rivermind-data/14b-datagen-ws")
    ap.add_argument("--tasks", default=",".join(str(t) for t in ALL_TASKS))
    ap.add_argument("--samples", type=int, default=4, help="samples per task (rejection sampling)")
    ap.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    ws = Path(args.workspace)
    tasks = [int(t) for t in args.tasks.split(",")]

    successes: dict[int, int] = {}
    kept = 0
    for ti in tasks:
        got = 0
        for s in range(args.samples):
            try:
                records, ok, turns = run_task(ti, ws, raw, args.timeout)
            except Exception as exc:  # noqa: BLE001
                print(f"[gen] task-{ti} sample-{s} ERROR {type(exc).__name__}: {exc}", flush=True)
                continue
            if ok:
                # keep every successful trajectory (rejection sampling) for SFT
                (out / f"sft-raw-task-{ti}-s{got}.ndjson").write_text(dump_pi_ndjson(records))
                # save the exact server-side token sequences for consistent SFT
                with open(out / "sft_tokens.jsonl", "a") as f:
                    for tn in turns:
                        if tn.get("prompt_ids") and tn.get("completion_ids"):
                            f.write(json.dumps({"task": ti, "prompt_ids": tn["prompt_ids"], "completion_ids": tn["completion_ids"]}) + "\n")
                got += 1
            print(f"[gen] task-{ti} sample-{s} ok={ok}", flush=True)
        successes[ti] = got
        kept += 1 if got > 0 else 0
        print(f"[gen] task-{ti} successes={got}/{args.samples}", flush=True)

    summary = {
        "model": MODEL,
        "tasks_with_success": kept,
        "per_task_successes": {f"task-{t}": successes[t] for t in tasks},
        "samples_per_task": args.samples,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("DATAGEN_DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
