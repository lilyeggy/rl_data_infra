"""Diagnose why a turn produced no usable tool call, from recorded evidence.

Read-only. For every recorded model response it reports:

* how far into the generation the first extractable call appears;
* whether the text ends inside an unterminated JSON object, i.e. whether a call
  was cut off by the token budget rather than never attempted;
* the token/character ratio, cross-checked against ``model-evidence.jsonl`` via
  the real tokenizer, so "1024 tokens" is verified rather than assumed.

Usage: PYTHONPATH=<repo> python3 probe_turn_yield.py <attempt-dir> <tokenizer-dir>
"""

from __future__ import annotations

import json
import os
import sys

from src.capture.tool_calls import _balanced_json_objects, extract_tool_calls

TOOLS = [
    {"type": "function", "function": {"name": name}}
    for name in ("read", "bash", "write", "edit", "ls")
]


def response_text(event: dict) -> str:
    response = (event.get("attributes") or {}).get("response")
    if isinstance(response, str):
        response = json.loads(response)
    choices = (response or {}).get("choices") or [{}]
    return (choices[0].get("message") or {}).get("content") or ""


def open_json_at_end(text: str) -> bool:
    """True when the text ends mid-object: a call the budget cut off."""
    spans = _balanced_json_objects(text)
    if not spans:
        return text.rfind("{") > text.rfind("}")
    last_start, span = spans[-1]
    return last_start + len(span) < len(text) and text.rfind("{") > text.rfind("}")


def main() -> int:
    base = sys.argv[1]
    tokenizer_dir = sys.argv[2]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    attempted = cut_off = acted = 0
    for episode in sorted(os.listdir(base)):
        directory = os.path.join(base, episode)
        events_path = os.path.join(directory, "raw-events.jsonl")
        evidence_path = os.path.join(directory, "model-evidence.jsonl")
        if not os.path.exists(events_path):
            continue
        rows = [json.loads(line) for line in open(events_path) if line.strip()]
        responses = [row for row in rows if row.get("event_type") == "MODEL_RESPONSE"]
        evidence = []
        if os.path.exists(evidence_path):
            evidence = [json.loads(line) for line in open(evidence_path) if line.strip()]
        for index, row in enumerate(responses):
            text = response_text(row)
            calls = extract_tool_calls(text, TOOLS)
            spans = _balanced_json_objects(text)
            offset = spans[0][0] if spans else -1
            open_end = open_json_at_end(text)
            acted += 1 if calls else 0
            attempted += 1 if spans else 0
            cut_off += 1 if (not calls and open_end) else 0
            ids = []
            if index < len(evidence):
                ids = evidence[index]["backend"]["response_token_ids"] or []
            decoded = tokenizer.decode(ids) if ids else ""
            print(
                f"{episode} call={index + 1} tokens={len(ids)} chars={len(text)} "
                f"chars_per_token={len(text) / max(len(ids), 1):.2f} "
                f"json_spans={len(spans)} first_at={offset} open_at_end={open_end} "
                f"calls={len(calls)}"
            )
            if ids and decoded[:80] != text[:80]:
                print("   TOKEN/TEXT MISMATCH: decoded[:, :80] =", repr(decoded[:80]))
    print(
        f"TOTAL responses={acted + (0)} acted={acted} with_json={attempted} "
        f"cut_off={cut_off}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
