"""Measure, from a finished round's evidence, whether turns carry tool calls.

Read-only probe for the smoke23 failure (`real tool observation and subsequent
model call required`): it decodes every recorded model response and runs the
repository's own extractor over it, so the answer is a measurement rather than a
guess about why the policy produced no action.

Usage: PYTHONPATH=<repo> python3 probe_smoke23_tool_calls.py <attempt-dir>
"""

from __future__ import annotations

import json
import os
import sys

from src.capture.tool_calls import extract_tool_calls

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


def main() -> int:
    base = sys.argv[1]
    total = with_call = 0
    for episode in sorted(os.listdir(base)):
        path = os.path.join(base, episode, "raw-events.jsonl")
        if not os.path.exists(path):
            print(f"{episode}: no raw events")
            continue
        rows = [json.loads(line) for line in open(path) if line.strip()]
        responses = [row for row in rows if row.get("event_type") == "MODEL_RESPONSE"]
        for index, row in enumerate(responses):
            text = response_text(row)
            calls = extract_tool_calls(text, TOOLS)
            total += 1
            with_call += 1 if calls else 0
            tags = text.count("<tool_call>")
            print(
                f"{episode} call={index + 1} chars={len(text)} tags={tags} "
                f"extracted={len(calls)} names={[c['function']['name'] for c in calls]}"
            )
    print(f"TOTAL model responses={total} carrying a usable tool call={with_call}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
