#!/usr/bin/env python3
"""Build a leakage-safe, turn-level LoRA SFT package from certified trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.contracts._json import canonical_json_bytes, sha256_json


PACKAGE_VERSION = "teacher-sft-training-package/v7"
QUALITY_POLICY_VERSION = "teacher-trajectory-quality/v2"
RUNTIME_FORMAT_POLICY_VERSION = "pi-qwen-runtime-format/v1"
TURN_SELECTION_POLICY_VERSION = "teacher-turn-selection/v2"
MESSAGE_CHAR_CAP = 2000


def _choose_gradient_accumulation(example_count: int) -> int:
    if example_count <= 0:
        raise ValueError("cannot choose accumulation for an empty dataset")
    # Prefer the proven single-A6000 effective batch size of 3, then nearby
    # values. Exact divisibility is mandatory so Transformers cannot silently
    # drop an epoch tail.
    return next(value for value in (3, 4, 5, 2, 1) if example_count % value == 0)


def _bytes_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _tool_signature(message: Mapping[str, Any]) -> tuple[str, ...]:
    calls = message.get("tool_calls", ())
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
        return ()
    result = []
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        function = call.get("function")
        if not isinstance(function, Mapping):
            continue
        result.append(
            sha256_json(
                {"name": function.get("name"), "arguments": function.get("arguments")}
            )
        )
    return tuple(result)


def _cap_text(value: str) -> str:
    if len(value) <= MESSAGE_CHAR_CAP:
        return value
    half = MESSAGE_CHAR_CAP // 2
    return (
        value[:half]
        + f"\n...[+{len(value) - MESSAGE_CHAR_CAP} chars truncated]...\n"
        + value[-half:]
    )


def _qwen_runtime_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Match the controlled Qwen server's message representation.

    OpenAI transports function arguments as JSON strings, while Qwen's chat
    template expects an object when rendering an assistant tool-call target.
    Tool-action prose is omitted so the action cannot be truncated behind a
    plan before the runtime's 200-token generation ceiling.
    """
    result: list[dict[str, Any]] = []
    for source in messages:
        message = dict(source)
        role = message.get("role")
        content = message.get("content")
        if role != "system" and isinstance(content, str):
            message["content"] = _cap_text(content)
        calls = message.get("tool_calls")
        if role == "assistant" and isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)) and calls:
            normalized_calls = []
            for call in calls:
                if not isinstance(call, Mapping):
                    raise ValueError("assistant tool call is not an object")
                normalized_call = dict(call)
                function = call.get("function")
                if not isinstance(function, Mapping):
                    raise ValueError("assistant tool call lacks function object")
                normalized_function = dict(function)
                arguments = normalized_function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise ValueError("assistant tool arguments are not valid JSON") from exc
                if not isinstance(arguments, Mapping):
                    raise ValueError("assistant tool arguments must decode to an object")
                normalized_function["arguments"] = dict(arguments)
                normalized_call["function"] = normalized_function
                normalized_calls.append(normalized_call)
            message["tool_calls"] = normalized_calls
            message["content"] = None
        result.append(message)
    return result


def _trajectory_quality(row: Mapping[str, Any]) -> dict[str, Any]:
    messages = row.get("messages", ())
    steps = row.get("steps", ())
    assistant = [
        item for item in messages
        if isinstance(item, Mapping) and item.get("role") == "assistant"
    ] if isinstance(messages, Sequence) else []
    step_rows = [item for item in steps if isinstance(item, Mapping)] if isinstance(steps, Sequence) else []
    unpaired = sum(item.get("result") is None for item in step_rows)
    tool_errors = sum(item.get("result_status") == "ERROR" for item in step_rows)
    signatures = [sig for item in assistant for sig in _tool_signature(item)]
    duplicate_calls = len(signatures) - len(set(signatures))
    tool_count = len(step_rows)
    error_rate = tool_errors / tool_count if tool_count else 0.0
    duplicate_rate = duplicate_calls / len(signatures) if signatures else 0.0
    final_answer = row.get("final_answer")
    reasons: list[str] = []
    if row.get("schema_version") != "teacher-sft-example/v2":
        reasons.append("example lacks action-observation-closed SFT schema v2")
    if row.get("certification_verdict") != "ELIGIBLE":
        reasons.append("trajectory is not SFT ELIGIBLE")
    if not isinstance(final_answer, str) or not final_answer.strip():
        reasons.append("final answer is empty")
    if len(assistant) < 2:
        reasons.append("fewer than two observable assistant turns")
    if unpaired:
        reasons.append(f"{unpaired} tool calls have no result")
    if tool_count > 96:
        reasons.append(f"tool step count {tool_count} exceeds 96")
    if error_rate > 0.35:
        reasons.append(f"tool error rate {error_rate:.3f} exceeds 0.35")
    if duplicate_rate > 0.40:
        reasons.append(f"duplicate tool-call rate {duplicate_rate:.3f} exceeds 0.40")
    if len(messages) > 200:
        reasons.append(f"message count {len(messages)} exceeds 200")
    score = max(
        0.0,
        1.0
        - min(error_rate, 1.0) * 0.35
        - min(duplicate_rate, 1.0) * 0.25
        - max(tool_count - 48, 0) / 96 * 0.20
        - max(len(messages) - 96, 0) / 200 * 0.20,
    )
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "quality_score": round(score, 6),
        "message_count": len(messages),
        "assistant_turn_count": len(assistant),
        "tool_step_count": tool_count,
        "tool_error_count": tool_errors,
        "tool_error_rate": round(error_rate, 6),
        "duplicate_tool_call_count": duplicate_calls,
        "duplicate_tool_call_rate": round(duplicate_rate, 6),
        "unpaired_tool_step_count": unpaired,
    }


def _turn_examples(
    row: Mapping[str, Any], quality: Mapping[str, Any], runtime: Mapping[str, Any]
) -> list[dict[str, Any]]:
    source_messages = row["messages"]
    if any(item.get("role") == "system" for item in source_messages if isinstance(item, Mapping)):
        raise ValueError("Teacher trajectory unexpectedly contains a system message")
    system = runtime.get("system_message")
    tools = runtime.get("tools")
    if not isinstance(system, Mapping) or system.get("role") != "system":
        raise ValueError(f"missing Pi system message for task {row.get('task_id')}")
    if not isinstance(tools, Sequence) or not tools:
        raise ValueError(f"missing Pi tool contract for task {row.get('task_id')}")
    messages = _qwen_runtime_messages([dict(system), *source_messages])
    seen_signatures: set[str] = set()
    result: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        calls = message.get("tool_calls", ())
        useful_text = isinstance(content, str) and bool(content.strip())
        call_rows = [item for item in calls if isinstance(item, Mapping)] if isinstance(calls, Sequence) else []
        signatures = _tool_signature(message)
        duplicate_only = bool(signatures) and all(item in seen_signatures for item in signatures)
        seen_signatures.update(signatures)
        if not useful_text and not call_rows:
            continue
        if duplicate_only and not useful_text:
            continue
        result.append(
            {
                "schema_version": "teacher-sft-turn/v2",
                "turn_id": f"{row['episode_id']}:{index}",
                "task_id": row["task_id"],
                "source_episode_id": row["episode_id"],
                "source_episode_checksum": row["episode_checksum"],
                "quality_score": quality["quality_score"],
                "messages": messages[: index + 1],
                "tools": tools,
                "tools_checksum": sha256_json(tools),
                "target_message_index": index,
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--runtime-contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    rows = [json.loads(line) for line in args.examples.read_text().splitlines() if line]
    splits = json.loads(args.splits.read_text())
    selected = json.loads(args.selected.read_text())
    runtime_contexts = json.loads(args.runtime_contexts.read_text())
    tools = runtime_contexts.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("runtime context contains no tools")
    tools_checksum = sha256_json(tools)
    if runtime_contexts.get("tools_checksum") != tools_checksum:
        raise ValueError("runtime context tool checksum mismatch")
    split_sets = {name: set(splits[name]) for name in ("train", "development", "test")}
    if any(split_sets[a] & split_sets[b] for a, b in (("train", "development"), ("train", "test"), ("development", "test"))):
        raise ValueError("task split leakage detected")
    unknown = set().union(*split_sets.values()) - set(selected)
    if unknown:
        raise ValueError(f"split contains unknown tasks: {sorted(unknown)}")

    quality_rows = []
    turns = []
    for row in rows:
        task_id = row.get("task_id")
        if task_id not in split_sets["train"]:
            raise ValueError(f"Teacher example outside TRAIN split: {task_id}")
        quality = _trajectory_quality(row)
        quality_row = {"task_id": task_id, "episode_id": row.get("episode_id"), **quality}
        quality_rows.append(quality_row)
        if quality["accepted"]:
            task_runtime = runtime_contexts.get("task_contexts", {}).get(task_id)
            if not isinstance(task_runtime, Mapping):
                raise ValueError(f"no Pi runtime context for training task: {task_id}")
            row_turns = _turn_examples(
                row, quality, {"system_message": task_runtime.get("system_message"), "tools": tools}
            )
            first_assistant = next(
                (index + 1 for index, message in enumerate(row["messages"])
                 if isinstance(message, Mapping) and message.get("role") == "assistant"),
                None,
            )
            included_indices = {item["target_message_index"] for item in row_turns}
            if first_assistant is None or first_assistant not in included_indices:
                raise ValueError(
                    f"quality selection removed the first assistant target for {task_id}"
                )
            quality_row["first_assistant_target_index"] = first_assistant
            quality_row["first_assistant_target_included"] = True
            quality_row["admitted_turn_count"] = len(row_turns)
            turns.extend(row_turns)
    accepted_tasks = {item["task_id"] for item in quality_rows if item["accepted"]}
    if not accepted_tasks or not turns:
        raise ValueError("quality gate admitted no training data")

    turns.sort(key=lambda item: (
        item["task_id"], item["source_episode_id"], item["target_message_index"]
    ))
    train_bytes = b"".join(canonical_json_bytes(item) + b"\n" for item in turns)
    (args.output / "train-turns.jsonl").write_bytes(train_bytes)
    _write_json(args.output / "quality-report.json", {
        "policy_version": QUALITY_POLICY_VERSION,
        "turn_selection_policy_version": TURN_SELECTION_POLICY_VERSION,
        "accepted_trajectory_count": sum(item["accepted"] for item in quality_rows),
        "rejected_trajectory_count": sum(not item["accepted"] for item in quality_rows),
        "turn_example_count": len(turns),
        "trajectories": quality_rows,
    })
    for split in ("development", "test"):
        _write_json(args.output / f"{split}-tasks.json", {
            "split": split.upper(),
            "teacher_unseen": True,
            "tasks": [
                {"instance_id": task_id, "base_commit": selected[task_id]["base_commit"]}
                for task_id in sorted(split_sets[split])
            ],
        })
    gradient_accumulation = _choose_gradient_accumulation(len(turns))
    lora = {
        "schema_version": "qwen-lora-sft-config/v1",
        "student_model": args.student_model,
        "dataset": "train-turns.jsonl",
        "max_length": args.max_length,
        "epochs": 2,
        "learning_rate": 0.0001,
        "batch_size": 1,
        "gradient_accumulation_steps": gradient_accumulation,
        "bf16": True,
        "gradient_checkpointing": True,
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                "up_proj", "down_proj", "lm_head",
            ]},
    }
    _write_json(args.output / "lora-config.json", lora)
    manifest = {
        "schema_version": PACKAGE_VERSION,
        "quality_policy_version": QUALITY_POLICY_VERSION,
        "turn_selection_policy_version": TURN_SELECTION_POLICY_VERSION,
        "runtime_format_policy_version": RUNTIME_FORMAT_POLICY_VERSION,
        "source_examples": str(args.examples.resolve()),
        "source_examples_sha256": _bytes_sha(args.examples.read_bytes()),
        "runtime_contexts": str(args.runtime_contexts.resolve()),
        "runtime_contexts_sha256": _bytes_sha(args.runtime_contexts.read_bytes()),
        "tools_checksum": tools_checksum,
        "runtime_format_policy": {
            "message_char_cap": MESSAGE_CHAR_CAP,
            "assistant_tool_arguments": "json-object",
            "assistant_tool_target_content": "omit-prose",
            "paired_tool_error_targets": "retain",
        },
        "output_head_policy": {
            "name": "lora-lm-head/v1",
            "reason": (
                "Native Qwen tool-call markers are added special tokens; "
                "the output head needs a trainable low-rank path."
            ),
        },
        "split_policy": splits,
        "accepted_train_tasks": sorted(accepted_tasks),
        "turn_example_count": len(turns),
        "train_turns_sha256": _bytes_sha(train_bytes),
        "student_model": args.student_model,
        "lora_config_checksum": sha256_json(lora),
        "optimizer_step_policy": {
            "name": "exact-epoch-divisor/v1",
            "gradient_accumulation_steps": gradient_accumulation,
            "optimizer_steps_per_epoch": len(turns) // gradient_accumulation,
        },
    }
    _write_json(args.output / "training-package-manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
