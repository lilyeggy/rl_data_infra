"""Thin Harness Adapter: canonical JSON action <-> Pi tool protocol.

This is the "Thin Harness Adapter" in the layered architecture:
``Canonical Agent Semantics -> Model-specific Training View -> Thin Harness Adapter``.

It has exactly two responsibilities and knows nothing about training data:
1. **canonical -> Pi**: parse the canonical SFT model's JSON output
   (``{"action_type": "read_file", "arguments": {...}}``) and emit the matching
   Pi tool call (``{"name": "read", "arguments": {...}}``) so the real Pi
   harness can execute it.
2. **Pi result -> canonical observation**: wrap a Pi tool result back into the
   canonical observation shape the model expects.

Rule: Pi tool names (read/bash/edit/grep/find/ls/...) exist ONLY here, never in
the canonical episode or training views. Unknown canonical action_types and
irreversible mappings fail closed (raise) — never silently dropped.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

# canonical action_type -> Pi tool name
CANONICAL_TO_PI_TOOL: dict[str, str] = {
    "read_file": "read",
    "search_code": "grep",
    "list_directory": "ls",
    "run_command": "bash",
    "edit_file": "edit",
    "write_file": "write",
    "finish": "finish",
    "tool_error": "tool_error",
    "environment_observation": "environment_observation",
}

# How to rewrite arguments from canonical shape to Pi tool arguments.
# Pi tools use: read(path, limit, offset), grep(pattern, path, context),
# ls(path), bash(command), edit(path, content, replace?), write(path, content).
_CANONICAL_TO_PI_ARGS: dict[str, dict[str, str]] = {
    "read_file": {"path": "path", "limit": "limit", "offset": "offset"},
    "search_code": {"pattern": "pattern", "path": "path", "context": "context"},
    "list_directory": {"path": "path"},
    "run_command": {"command": "command"},
    "edit_file": {
        "path": "path",
        "content": "content",
        "old_string": "replace",
        "new_string": "replacement",
    },
    "write_file": {"path": "path", "content": "content"},
    "finish": {},
    "tool_error": {},
    "environment_observation": {"cwd": "cwd"},
}

# Pi tool result -> minimal canonical observation (only what the model consumes).
_OBSERVATION_TOOL_MAP = {
    "read": "read_file",
    "grep": "search_code",
    "ls": "list_directory",
    "bash": "run_command",
    "edit": "edit_file",
    "write": "write_file",
    "finish": "finish",
}


def parse_canonical_action(text: str) -> tuple[bool, str, dict]:
    """Parse one canonical JSON action from generated text.

    Returns (ok, error, payload). Accepts a bare ``{"action_type":..., "arguments":...}``
    object (with or without code fences / <tool_call> wrappers). Fails closed on
    anything else.
    """

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned.strip())
    # strip optional <tool_call> wrappers
    m_wrap = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", cleaned, re.S)
    if m_wrap:
        cleaned = m_wrap.group(1)
    obj = None
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        # find first balanced JSON object
        depth = 0
        start = None
        for i, ch in enumerate(cleaned):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = cleaned[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                    except json.JSONDecodeError:
                        continue
                    break
    if not isinstance(obj, Mapping):
        return False, "no parseable JSON action object", {}
    action_type = obj.get("action_type") or obj.get("name")
    if not isinstance(action_type, str) or not action_type.strip():
        return False, f"missing action_type in {json.dumps(obj)[:120]}", {}
    if action_type not in CANONICAL_TO_PI_TOOL:
        return False, f"unknown canonical action_type {action_type!r}", {}
    arguments = obj.get("arguments")
    if not isinstance(arguments, Mapping):
        return False, "arguments must be an object", {}
    return True, "", {"action_type": action_type, "arguments": arguments}


def canonical_to_pi_call(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a parsed canonical action into a Pi tool call.

    Raises ValueError on unknown action_type or an irreversible argument mapping
    (fail closed).
    """

    action_type = payload["action_type"]
    pi_tool = CANONICAL_TO_PI_TOOL.get(action_type)
    if pi_tool is None:
        raise ValueError(f"cannot map canonical action_type {action_type!r} to Pi tool")
    args = payload.get("arguments") or {}
    pi_args: dict[str, Any] = {}
    for key, value in args.items():
        mapped = _CANONICAL_TO_PI_ARGS.get(action_type, {}).get(key, key)
        pi_args[mapped] = value
    return {"name": pi_tool, "arguments": pi_args}


def pi_result_to_observation(pi_name: str, result: Any) -> dict[str, Any]:
    """Wrap a Pi tool result into the canonical observation shape."""
    return {"tool": pi_name, "observation": result}


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


if __name__ == "__main__":
    # quick self-check
    test = '{"action_type": "read_file", "arguments": {"path": "sympy/core/a.py", "limit": 100}}'
    ok, err, payload = parse_canonical_action(test)
    assert ok, err
    print("parsed:", payload)
    print("pi call:", canonical_to_pi_call(payload))
    for bad in (
        '{"action_type": "teleport"}',
        "hello",
        '{"action_type":"read_file","arguments":"x"}',
    ):
        ok2, err2, _ = parse_canonical_action(bad)
        assert not ok2, bad
        print("fail-closed:", err2)