"""ARCHIVED: baseline + SFT evaluation for local-model harness experiments.

Usage:
  python3 baseline_local_harness.py --model <path> [--adapter <lora-path>]
"""

from __future__ import annotations

import argparse
import json

from experiments.local_model.harness import (
    EVAL_TASKS,
    device_and_dtype,
    load_model,
    rollout,
    verify,
)


def evaluate(model, tokenizer, ws_root, device, label: str):
    results = []
    for task_index in EVAL_TASKS:
        messages, reward, status = rollout(
            model, tokenizer, ws_root, task_index, sample=False, device=device
        )
        ok = reward == 1.0
        results.append(ok)
        print(json.dumps({"model": label, "task": task_index, "success": ok, "status": status}))
    print("SUMMARY", label, "success=", sum(results), "/", len(results), results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--workspace", default="/tmp/local-harness-workspace")
    args = parser.parse_args()

    device, dtype = device_and_dtype()
    model, tokenizer = load_model(args.model, adapter_path=args.adapter, device=device, dtype=dtype)
    from pathlib import Path
    ws_root = Path(args.workspace)
    evaluate(model, tokenizer, ws_root, device, "adapter" if args.adapter else "base")


if __name__ == "__main__":
    main()
