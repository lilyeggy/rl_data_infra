"""ARCHIVED: build SFT dataset from real Pi candidate fixtures.

Generalized: reads sanitized NDJSON traces (one per task, candidate arm) and
renders them into chat-format training examples with a verifier flag.

Usage:
  python3 build_sft_dataset.py \
    --fixtures tests/fixtures/pi/v3-live \
    --tasks 1,2,3,4,5,6,7,8 \
    --out experiments/local_model/sft-dataset-v3.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.task_suite import expected_answer


def _missing_tool_result_pairing(steps):
    """Steps (tool calls) whose tool call has no paired result."""
    missing = []
    for step in steps:
        if isinstance(step, dict):
            content = step.get("content")
            result = step.get("result")
        else:
            content, result = step
        if content is not None and result is None:
            missing.append(content)
    return missing


def classify_example(example):
    """Return (trainable, reasons) for a candidate SFT example (fail-closed).

    A sample is trainable only when it has a verifiable non-empty final answer
    and no unresolved tool-result pairing.  Empty final answers, unparsable
    answers and missing tool-result pairings are quarantined, never trained.
    """
    reasons = []
    final = (example.get("final_answer") or "").strip()
    if not final:
        reasons.append("empty final answer")
    if not example.get("verified"):
        reasons.append("verified=False (no verifier PASSED evidence)")
    unpaired = _missing_tool_result_pairing(example.get("steps", []))
    if unpaired:
        reasons.append(f"tool result pairing missing for {len(unpaired)} step(s)")
    messages = example.get("messages", [])
    if not messages:
        reasons.append("no chat messages")
    return (not reasons, reasons)


def extract_trajectory(fixture_path: Path):
    records = [
        json.loads(line) for line in fixture_path.read_text().splitlines() if line.strip()
    ]
    steps = []
    tool_results = {}
    final = None

    def collect(v, texts):
        if isinstance(v, dict):
            if isinstance(v.get("text"), str):
                texts.append(v["text"])
            for x in v.values():
                collect(x, texts)
        elif isinstance(v, list):
            for x in v:
                collect(x, texts)

    calls = []
    for rec in records:
        if rec.get("type") == "tool_execution_end":
            calls.append(rec)
            texts = []
            collect(rec.get("result"), texts)
            err = "ERROR: " if rec.get("isError") else ""
            tool_results[rec.get("toolCallId")] = err + " ".join(texts).strip()
    for rec in records:
        if rec.get("type") != "message_end":
            continue
        msg = rec.get("message") or {}
        if msg.get("role") != "assistant":
            continue
        for item in msg.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "toolCall":
                args = item.get("arguments") or {}
                name = item.get("name")
                action = {"action": "read" if name == "read" else "find"}
                if name == "read":
                    action["path"] = args.get("path")
                else:
                    action["pattern"] = args.get("pattern")
                steps.append((json.dumps(action, ensure_ascii=False), None))
            elif (
                isinstance(item, dict)
                and item.get("type") == "text"
                and msg.get("stopReason") == "stop"
            ):
                final = item.get("text")
    for i, (content, _) in enumerate(steps):
        if i < len(calls):
            steps[i] = (content, tool_results.get(calls[i].get("toolCallId"), "ERROR"))
    return steps, final


def main():
    from experiments.local_model.harness import SYSTEM, user_prompt

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures",
        default=str(Path(__file__).parents[2] / "tests" / "fixtures" / "pi" / "v2.1-live"),
    )
    parser.add_argument("--tasks", default="1,2,3")
    parser.add_argument("--out", default=str(Path(__file__).parent / "sft-dataset.jsonl"))
    parser.add_argument("--prefix", default="candidate")
    args = parser.parse_args()

    fixtures = Path(args.fixtures)
    tasks = [int(t) for t in args.tasks.split(",")]
    examples = []
    quarantine = []
    for task_index in tasks:
        path = fixtures / f"{args.prefix}-{task_index}.ndjson"
        if not path.exists():
            print("SKIP missing fixture:", path)
            continue
        steps, final = extract_trajectory(path)
        m = re.search(r"\{.*\}", (final or "").strip(), re.S)
        final_json = m.group(0) if m else (final or "").strip()
        try:
            ok = json.loads(final_json) == expected_answer(task_index)
        except Exception:
            ok = False
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt(task_index)},
        ]
        for content, result in steps:
            messages.append({"role": "assistant", "content": content})
            if result is not None:
                messages.append({"role": "user", "content": f"Tool result: {result}"})
        messages.append({"role": "assistant", "content": final_json})
        example = {
            "messages": messages,
            "task": task_index,
            "verified": ok,
            "final_answer": final_json,
            "steps": [
                {"content": content, "result": result} for content, result in steps
            ],
        }
        trainable, reasons = classify_example(example)
        print(
            "TASK", task_index,
            "verified_final=", ok,
            "tool_steps=", len(steps),
            "trainable=", trainable,
        )
        if trainable:
            examples.append(
                {
                    "messages": example["messages"],
                    "task": task_index,
                    "verified": True,
                    "steps": len(steps),
                }
            )
        else:
            quarantine.append(
                {
                    "task": task_index,
                    "trainable": False,
                    "reasons": reasons,
                    "has_final_answer": bool((final_json or "").strip()),
                }
            )
    out = Path(args.out)
    with out.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    quarantine_path = out.with_suffix(".quarantine.jsonl")
    with quarantine_path.open("w") as f:
        for item in quarantine:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(
        "EXAMPLES=", len(examples),
        "QUARANTINED=", len(quarantine),
        "OUT=", out,
        "QUARANTINE=", quarantine_path,
    )


if __name__ == "__main__":
    main()
