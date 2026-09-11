"""Show the JSON spans the strict extractor rejected, from recorded evidence.

Read-only. The extractor refuses a call whose ``name`` is undeclared, whose
``arguments`` is not an object, or whose JSON is malformed -- all by design. But
OpenAI's own wire format carries ``arguments`` as a JSON *string*, and the
repository's tagged path passes the parser's string straight through, so a
rejection here is only correct if the span really is not a tool call. This prints
the spans so that judgement rests on the text.

Usage: PYTHONPATH=<repo> python3 probe_rejected_spans.py <attempt-dir>
"""

from __future__ import annotations

import json
import os
import sys

from src.capture.tool_calls import _balanced_json_objects, allowed_tool_names

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
    allowed = allowed_tool_names(TOOLS)
    for episode in sorted(os.listdir(base)):
        path = os.path.join(base, episode, "raw-events.jsonl")
        if not os.path.exists(path):
            continue
        rows = [json.loads(line) for line in open(path) if line.strip()]
        responses = [row for row in rows if row.get("event_type") == "MODEL_RESPONSE"]
        if not responses:
            continue
        text = response_text(responses[0])
        spans = _balanced_json_objects(text)
        if not spans:
            print(f"{episode}: no JSON span at all")
            continue
        print(f"--- {episode}")
        for offset, span in spans[:6]:
            try:
                parsed = json.loads(span)
                kind = type(parsed).__name__
                if isinstance(parsed, dict):
                    keys = sorted(parsed)
                    name = parsed.get("name")
                    args = parsed.get("arguments")
                    verdict = (
                        "ACCEPTED" if name in allowed and isinstance(args, dict) else
                        "rejected: " + (
                            "undeclared name" if name not in allowed else
                            "arguments is " + type(args).__name__
                        )
                    )
                    print(f"  @{offset} keys={keys} name={name!r} "
                          f"args_type={type(args).__name__} -> {verdict}")
                    print(f"     span[:220]={span[:220]!r}")
                else:
                    print(f"  @{offset} parsed as {kind}; span[:160]={span[:160]!r}")
            except json.JSONDecodeError:
                print(f"  @{offset} UNPARSEABLE span[:160]={span[:160]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
