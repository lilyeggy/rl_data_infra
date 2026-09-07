"""Fail-closed structural checks for already certified SFT examples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def classify_sft_example(example: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    """Check the final serialized SFT example after episode certification.

    This is deliberately only a structural gate. A caller must also possess an
    ELIGIBLE decision for the SFT consumer profile; a boolean ``verified`` field
    in arbitrary JSON is never sufficient on its own.
    """

    reasons: list[str] = []
    final = example.get("final_answer")
    if not isinstance(final, str) or not final.strip():
        reasons.append("empty final answer")
    if example.get("certification_verdict") != "ELIGIBLE":
        reasons.append("missing ELIGIBLE SFT certification decision")
    messages = example.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or not messages:
        reasons.append("no chat messages")
    steps = example.get("steps", ())
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        reasons.append("steps must be an array")
    else:
        missing = 0
        for step in steps:
            if not isinstance(step, Mapping):
                missing += 1
                continue
            if step.get("content") is not None and step.get("result") is None:
                missing += 1
        if missing:
            reasons.append(f"tool result pairing missing for {missing} step(s)")
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        pending: list[str] = []
        seen_calls: set[str] = set()
        seen_results: set[str] = set()
        for message in messages:
            if not isinstance(message, Mapping):
                reasons.append("chat message is not an object")
                continue
            role = message.get("role")
            if role == "assistant":
                if pending:
                    reasons.append("assistant decision occurs before prior tool observations")
                    pending.clear()
                calls = message.get("tool_calls", ())
                if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
                    continue
                for call in calls:
                    if not isinstance(call, Mapping) or not isinstance(call.get("id"), str):
                        reasons.append("assistant tool call lacks a stable id")
                        continue
                    call_id = call["id"]
                    if call_id in seen_calls:
                        reasons.append(f"duplicate assistant tool call id {call_id!r}")
                    seen_calls.add(call_id)
                    pending.append(call_id)
            elif role == "tool":
                call_id = message.get("tool_call_id")
                if not isinstance(call_id, str) or call_id not in pending:
                    reasons.append("tool observation has no pending matching call")
                    continue
                if call_id in seen_results:
                    reasons.append(f"duplicate tool observation for {call_id!r}")
                    continue
                pending.remove(call_id)
                seen_results.add(call_id)
        if pending:
            reasons.append(f"tool observations missing for {len(pending)} call(s)")
        if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)):
            step_ids = {
                step.get("tool_call_id") for step in steps
                if isinstance(step, Mapping) and isinstance(step.get("tool_call_id"), str)
            }
            if seen_calls != step_ids:
                reasons.append("chat tool calls and canonical steps disagree")
            if seen_results != step_ids:
                reasons.append("chat tool observations and canonical steps disagree")
    return not reasons, tuple(reasons)
