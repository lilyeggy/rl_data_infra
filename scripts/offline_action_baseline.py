#!/usr/bin/env python3
"""Gate-1 offline evaluation: format + action-selection baseline over models.

For each candidate model (base / v6 / v7 / canonical-generic), load it on the
GPU, run greedy inference on canonical SFT "context -> choose next action"
prompts, parse the response as a canonical action, and score form-legal exports
and tool-accuracy against the ground-truth next action from the dataset.

This is an OFFLINE baseline; it isolates decision ability (Gate 1) from
harness-loop execution (Gates 2-4).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

VALID_TOOLS = {
    "read_file", "search_code", "list_directory", "run_command", "edit_file",
    "write_file", "finish", "tool_error", "environment_observation",
}


def _parse_action_text(text: str) -> Any:
    """Best-effort parse of model text into a canonical action object."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
    m = re.search(r"(?:\"?(action_type|name|tool)\"?)\s*[:=]\s*\"?([a-z_]+)\"?", text)
    if m:
        return {"name": m.group(2), "arguments": {}}
    return {"raw_text": text}


def main() -> int:  # pragma: no cover - server-invoked
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-and-adapter", nargs="+", required=True,
                        help="label=/path/to/model[+adapter] repeatable")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line]
    if args.limit > 0:
        rows = rows[: args.limit]

    conversations: list[list[dict[str, str]]] = []
    expected_tools: list[str] = []
    for row in rows:
        conversations.append([
            {"role": m["role"], "content": m["content"]}
            for m in row["messages"] if m["role"] != "assistant"
        ])
        expected_tools.append(row.get("target_action", {}).get("action_type", ""))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    results: dict[str, dict[str, Any]] = {}
    for entry in args.model_and_adapter:
        if "=" in entry:
            label, path = entry.split("=", 1)
        else:
            label, path = entry, entry
        base, sep, adapter = path.partition("+")
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager"
        )
        if sep:  # has adapter
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        model.eval()

        submissions: list[Any] = []
        with torch.no_grad():
            for messages in conversations:
                p = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                enc = tokenizer(p, return_tensors="pt").to(model.device)
                out = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                gen_text = tokenizer.decode(
                    out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
                )
                submissions.append(_parse_action_text(gen_text))

        legal = parse_ok = correct_tool = invalid = 0
        for sub, exp_tool in zip(submissions, expected_tools):
            obj = sub
            if not isinstance(obj, Mapping):
                invalid += 1
                continue
            name = obj.get("name") or obj.get("action_type") or obj.get("tool")
            if not name:
                invalid += 1
                continue
            parse_ok += 1
            if name in VALID_TOOLS:
                legal += 1
            if name == exp_tool:
                correct_tool += 1
        total = len(submissions)
        results[label] = {
            "total": total,
            "parse_ok_count": parse_ok,
            "legal_count": legal,
            "invalid_count": invalid,
            "correct_tool_count": correct_tool,
            "legal_ratio": round(legal / total, 4) if total else 0,
            "action_accuracy": round(correct_tool / total, 4) if total else 0,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"version": "gate1-offline-baseline/v1", "models": results}
    (args.output_dir / "offline-baseline-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
