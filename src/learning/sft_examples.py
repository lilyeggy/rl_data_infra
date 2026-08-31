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
    return not reasons, tuple(reasons)
