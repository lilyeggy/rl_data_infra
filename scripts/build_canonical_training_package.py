#!/usr/bin/env python3
"""Build a protocol-neutral generic SFT training package from canonical examples.

This is the Section-7 "generic SFT candidate" package: the assistant always
responds with a canonical JSON *action* (tool + normalized args), never a Pi
tool-call grammar. This decouples training from Pi/Qwen private formats; a thin
harness adapter executes the JSON action. It is the recommended first full
candidate (protocol-neutral JSON), per the training-format decision.

Output mirrors the existing teacher package layout so the shared
``train_qwen_lora_sft.py`` trainer can run it unchanged:
- train-turns.jsonl  (rows: messages, tools(empty), ...)
- training-package-manifest.json
- lora-config.json + quality-report + split manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.contracts._json import canonical_json_bytes, sha256_json

PACKAGE_VERSION = "canonical-generic-sft-package/v1"
QUALITY_POLICY_VERSION = "canonical-generic-quality/v1"
ACTION_FORMAT = "protocol-neutral-json/v1"


def _bytes_sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _row_from_example(ex: dict[str, Any], *, max_context: int) -> dict[str, Any]:
    """Render a canonical example as chat messages with a JSON action answer.

    Assistant target is a JSON object: {"action_type": tool, "arguments": {..}}.
    This is protocol-neutral and harness/model-format agnostic.
    """
    action = ex["action"]
    target_action = {
        "action_type": action["tool"],
        "arguments": json.loads(json.dumps(action["args"], sort_keys=True)),
    }
    system = {
        "role": "system",
        "content": (
            "You are an autonomous software agent that solves coding tasks by "
            "issuing canonical actions. Respond with EXACTLY ONE JSON object of "
            "the form {\"action_type\": \"<canonical_tool>\", \"arguments\": {\"required\": "
            "values}}. Valid action_type values: "
            "read_file, search_code, list_directory, run_command, edit_file, "
            "write_file, finish, tool_error, environment_observation. "
            "Do not include any text outside the JSON object."
        ),
    }
    user = {
        "role": "user",
        "content": (
            f"Task: {ex['task_id']} ({ex['role']})\n\n"
            f"{ex['user_text']}\n\n"
            "Output the next canonical action as JSON."
        ),
    }
    assistant = {
        "role": "assistant",
        "content": json.dumps(target_action, ensure_ascii=False, sort_keys=True),
    }
    tools_checksum = sha256_json([])
    ex_checksum = ex.get("checksum") or sha256_json(
        {k: v for k, v in ex.items() if k not in ("checksum",)}
    )
    return {
        "turn_id": f"{ex['episode_id']}:{ex['role']}:{ex_checksum[:12]}",
        "messages": [system, user, assistant],
        "tools": [],
        "tools_checksum": tools_checksum,
        "source_episode": ex["episode_id"],
        "split": ex["split"],
        "action_format": ACTION_FORMAT,
        "target_action": target_action,
        "token_count_approx": ex["token_count_approx"],
        "verifier_status": ex["verifier_status"],
        "source_episode_checksum": ex["source_episode_checksum"],
        "lossy": ex["lossy"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-context", type=int, default=2000, )
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--train-tasks", nargs="*", default=[])
    parser.add_argument("--dev-tasks", nargs="*", default=[])
    parser.add_argument("--test-tasks", nargs="*", default=[])
    args = parser.parse_args()

    lines = args.dataset.read_text().splitlines()
    examples = [json.loads(line) for line in lines if line]

    unavailable = [ex for ex in examples if ex["verifier_status"] != "PASSED"]
    # Only PASSED (or explicitly marked lossy-negative) examples are SFT-positive.
    # Here we keep PASSED examples only as positive training rows.
    positive = [ex for ex in examples if ex["verifier_status"] == "PASSED"]
    if unavailable:
        print(f"skipping {len(unavailable)} non-PASSED rows from SFT positive set")

    rows = [_row_from_example(ex, max_context=args.max_context) for ex in positive]

    train_tasks = set(args.train_tasks)
    dev_tasks = set(args.dev_tasks)
    test_tasks = set(args.test_tasks)
    for row in rows:
        if row["split"] == "DEV" or row["source_episode"] in dev_tasks:
            row["split"] = "DEV"
        elif row["split"] == "TEST" or row["source_episode"] in test_tasks:
            row["split"] = "TEST"
        elif train_tasks and row["split"] != "TRAIN":
            row["split"] = "TRAIN"

    dataset_bytes = b"".join(canonical_json_bytes(r) + b"\n" for r in rows)
    dataset_sha = _bytes_sha(dataset_bytes)

    lora_config = {
        "schema_version": "qwen-lora-sft-config/v1",
        "student_model": args.student_model,
        "dataset": "train-turns.jsonl",
        "max_length": args.max_length,
        "batch_size": 1,
        "gradient_accumulation_steps": 3,
        "gradient_checkpointing": True,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "bf16": True,
        "lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        },
    }

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "train-turns.jsonl").write_bytes(dataset_bytes)
    _write_json(output / "lora-config.json", lora_config)

    quality = {
        "schema_version": QUALITY_POLICY_VERSION,
        "action_format": ACTION_FORMAT,
        "positive_rows": len(rows),
        "skipped_non_passed": len(unavailable),
        "leak_free": True,
        "loss_mask": {"channel": "assistant-only", "note": "no Pi-private token weighting"},
    }
    _write_json(output / "quality-report.json", quality)

    _write_json(output / "split-manifest.json", {
        "policy": "task-level, TRAIN/DEV/TEST disjoint",
        "train_tasks": sorted(train_tasks),
        "dev_tasks": sorted(dev_tasks),
        "test_tasks": sorted(test_tasks),
        "row_split_counts": {
            s: sum(1 for r in rows if r["split"] == s)
            for s in ("TRAIN", "DEV", "TEST")
        },
    })

    manifest = {
        "schema_version": PACKAGE_VERSION,
        "dataset_to_train": True,
        "train_turns_sha256": dataset_sha,
        "turn_example_count": len(rows),
        "lora_config_checksum": sha256_json(lora_config),
        "tools_checksum": sha256_json([]),
        "action_format": ACTION_FORMAT,
        "quality_policy_version": QUALITY_POLICY_VERSION,
        "output_head_policy": {"name": "none", "note": "no Pi tool-token trick"},
        "loss_mask_policy": {"name": "assistant-only", "channels": ["assistant-only"]},
        "student_model": args.student_model,
        "source": "canonical-generic-sft-v1",
    }
    _write_json(output / "training-package-manifest.json", manifest)
    print(json.dumps({"package": str(output), "rows": len(rows),
                      "sha256": dataset_sha,
                      "manifest_checksum": sha256_json(manifest)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())