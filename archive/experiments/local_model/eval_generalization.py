"""ARCHIVED: generalization validation for the retired local-model prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.local_model.harness import (
    EVAL_TASKS,
    device_and_dtype,
    load_model,
    rollout,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--workspace", default="/tmp/local-harness-workspace")
    args = parser.parse_args()

    device, dtype = device_and_dtype()
    ws_root = Path(args.workspace)
    results = {}
    for label, adapter in (("base", None), ("adapter", args.adapter)):
        if adapter is None and args.adapter is None:
            pass
        model, tokenizer = load_model(args.model, adapter_path=adapter, device=device, dtype=dtype)
        row = []
        for task_index in EVAL_TASKS:
            messages, reward, status = rollout(model, tokenizer, ws_root, task_index, sample=False, device=device)
            ok = reward == 1.0
            row.append(ok)
            print(json.dumps({"model": label, "task": task_index, "success": ok, "status": status}))
        results[label] = row
        print("SUMMARY", label, "success=", sum(row), "/", len(row), row)
        del model, tokenizer
    if args.adapter is not None:
        print("DIFF base=", results["base"], "adapter=", results["adapter"])


if __name__ == "__main__":
    main()
