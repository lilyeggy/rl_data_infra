"""Serving-side canonical tool-call extraction for the harness server.

The openai_server parser only understands Pi-format payloads
(``{"name":"read","arguments":{...}}``). A canonical SFT model emits
``{"action_type":"read_file","arguments":{...}}``. This module bridges the two
at serving time so the real Pi harness can drive the canonical model.

It is intentionally a thin, fail-closed shim: it contains NO training data,
NO absolute paths, and its only job is canonical-action -> Pi-tool translation.
A model that emits neither a recognized Pi tool call nor a recognized canonical
action yields NO tool call (the harness sees a plain stop/empty response) and an
explicit diagnostic, never a fabricated tool call.
"""

from __future__ import annotations

from typing import Any, Mapping

from src.capture.canonical_harness_adapter import canonical_to_pi_call, parse_canonical_action


def extract_tool_calls(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract Pi tool-call payloads from generated text.

    Supports both Pi-native payloads (``{"name":..., "arguments":...}``) and
    canonical payloads (``{"action_type":..., "arguments":...}``). Canonical
    payloads are translated through the harness adapter to Pi tool names.
    Returns (tool_call_payloads, diagnostics).
    """

    # 1) Pi-native format first (used by base Qwen via <tool_call> wrapper).
    pi_payloads = _pi_native_payloads(text)
    if pi_payloads:
        return pi_payloads, []

    # 2) Canonical action format (used by the canonical generic SFT model).
    canonical = parse_canonical_action(text)
    if canonical[0]:
        ok, err, payload = canonical
        try:
            pi_call = canonical_to_pi_call(payload)
        except ValueError as exc:
            return [], [f"canonical->pi mapping failed: {exc}"]
        return [pi_call], ["canonical_action"]

    return [], [f"no tool call parsed: {canonical[1]}"]


def _pi_native_payloads(text: str) -> list[dict[str, Any]]:
    """Best effort: recognize <tool_call>{name,arguments}</tool_call>."""
    import re

    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    if not m:
        return []
    try:
        import json

        payload = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, Mapping) or "name" not in payload:
        return []
    return [{"name": payload["name"], "arguments": payload.get("arguments", {})}]