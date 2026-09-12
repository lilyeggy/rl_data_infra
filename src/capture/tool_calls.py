"""Recover tool calls from a policy's generated text.

The policy this project trains (Qwen2.5-Coder-14B plus an APPS/SWE SFT adapter)
emits tool calls as bare JSON objects -- ``{"name": ..., "arguments": {...}}`` --
and only sometimes wraps them in the ``<tool_call>`` tags its chat template asks
for. Both the stage D and stage E rollout servers handle that by doing their own
extraction after generation, which is why those stages produced executable
episodes.

Stock vLLM is stricter: its tool parsers (``hermes``, ``qwen3_coder``, ...) match
the tagged form only, so a bare-JSON generation comes back with no ``tool_calls``
at all, Pi executes nothing, and every episode ends on its first turn. The
framework path therefore needs the same tolerance the earlier stages had, applied
where the protocol conversion already happens: the data-plane proxy.

Text decoding is used only to build the OpenAI-shaped response Pi consumes. The
training tokens are never rewritten -- they still come from the engine's native
``response_token_ids``/``response_logprobs``.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from typing import Any

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def allowed_tool_names(tools: Any) -> set[str]:
    """Declared tool names; anything undeclared is never accepted.

    Tolerant about the container and the entries: Pi's request body arrives as
    a list of dicts, while the proxy's request evidence is a tuple of frozen
    ``mappingproxy`` objects. Anything that is not a mapping, or that carries
    no function name, contributes nothing.
    """
    names: set[str] = set()
    if tools is None or isinstance(tools, (str, bytes)):
        return names
    try:
        entries = list(tools)
    except TypeError:
        return names
    for tool in entries:
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        if isinstance(function, Mapping) and function.get("name"):
            names.add(str(function["name"]))
    return names


def _balanced_json_objects(text: str) -> list[tuple[int, str]]:
    """Scan top-level ``{...}`` spans, honouring strings and escapes.

    A regex cannot span the nested ``arguments`` object, so brace depth is
    tracked by hand. Each span carries its start offset so a span already
    covered by a ``<tool_call>`` block can be skipped: the same call would
    otherwise be counted twice, once per extraction path.
    """
    spans: list[tuple[int, str]] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            current = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
            elif current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    spans.append((start, text[start : end + 1]))
                    break
    return spans


def _tag_block_ranges(text: str) -> list[tuple[int, int]]:
    """Character ranges covered by ``<tool_call>...</tool_call>`` blocks."""
    ranges: list[tuple[int, int]] = []
    for start, end in ((m.start(), m.end()) for m in _TOOL_CALL_BLOCK.finditer(text)):
        ranges.append((start, end))
    return ranges


def extract_tool_calls(text: str, tools: Any) -> list[dict[str, Any]]:
    """Parse tagged or bare tool-call JSON into OpenAI ``tool_calls``.

    Mirrors the stage D/E extraction: an undeclared name, malformed JSON, a
    non-object ``arguments``, or an empty ``arguments`` object yields nothing,
    so a bad generation produces no call rather than a fabricated one.
    """
    allowed = allowed_tool_names(tools)
    if not allowed or not isinstance(text, str) or not text:
        return []

    tagged = _tag_block_ranges(text)
    candidates = [match.group(1) for match in _TOOL_CALL_BLOCK.finditer(text)]
    candidates.extend(
        span
        for start, span in _balanced_json_objects(text)
        if not any(begin <= start < end for begin, end in tagged)
    )

    calls: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        name = parsed.get("name")
        arguments = parsed.get("arguments", {})
        if name not in allowed or not isinstance(arguments, dict):
            continue
        # Empty-argument calls make Pi execute a no-op and retry in a loop;
        # returning none lets it re-ask instead.
        if not arguments:
            continue
        calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        arguments, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            }
        )
    return calls
