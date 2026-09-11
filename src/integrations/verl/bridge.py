"""Model bridge between Pi and the verl-managed vLLM endpoint.

The bridge never starts its own model service and never changes rewards. It
forwards Pi's OpenAI-protocol requests to the verl endpoint, captures native
per-call token evidence, verifies generation-history contiguity, and renders
admitted sequences into the pinned verl ``AgentLoopOutput`` field mapping.

Text decoding is used only to return responses to Pi. Decoded text is never
re-encoded to replace native training tokens.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.errors import ContractValidationError
from src.integrations.verl.admission import AdmittedVerlSequence

BRIDGE_VERSION = "verl-model-bridge/v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class BridgeCallRecord:
    """Native evidence for one model call, in execution order."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    response_logprobs: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt_token_ids": list(self.prompt_token_ids),
            "response_token_ids": list(self.response_token_ids),
            "response_logprobs": list(self.response_logprobs),
        }


def _int_tuple(value: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContractValidationError(f"{field_name} must be an array")
    output: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ContractValidationError(f"{field_name} must contain integers")
        output.append(item)
    return tuple(output)


def build_per_call_segments(
    records: Sequence[BridgeCallRecord],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[int, ...]]:
    """Verify history contiguity and split per-call native evidence.

    Returns ``(segments, frozen_prompt_ids)`` for
    :func:`assemble_episode_sequence`. The suffix of call N is the exact token
    block appended to the context before call N+1, derived arithmetically from
    observed prompt tokens — never by re-tokenizing text.

    Any history rewrite (non-prefix continuation) raises instead of repairing.
    """
    items = tuple(records)
    if not items:
        raise ContractValidationError("bridge captured no model calls")
    prompts: list[tuple[int, ...]] = []
    for index, record in enumerate(items):
        if not isinstance(record, BridgeCallRecord):
            raise ContractValidationError(f"records[{index}] must be a BridgeCallRecord")
        if not record.request_id:
            raise ContractValidationError(f"records[{index}] lacks a request_id")
        prompt = _int_tuple(record.prompt_token_ids, f"records[{index}].prompt_token_ids")
        response = _int_tuple(record.response_token_ids, f"records[{index}].response_token_ids")
        if not prompt:
            raise ContractValidationError(f"records[{index}] has empty prompt tokens")
        if not response:
            raise ContractValidationError(f"records[{index}] has empty response tokens")
        if len(record.response_logprobs) != len(response):
            raise ContractValidationError(f"records[{index}] response/logprob misalignment")
        prompts.append(prompt)
    frozen_prompt = prompts[0]
    segments: list[Mapping[str, Any]] = []
    for index, record in enumerate(items):
        response = _int_tuple(record.response_token_ids, f"records[{index}].response")
        if index < len(items) - 1:
            expected_prefix = prompts[index] + response
            following = prompts[index + 1]
            if tuple(following[: len(expected_prefix)]) != tuple(expected_prefix):
                raise ContractValidationError(
                    f"records[{index + 1}] rewrote generation history; "
                    "episode is not trainable"
                )
            suffix = tuple(following[len(expected_prefix):])
        else:
            suffix = ()
        segments.append(
            {
                "native_response_ids": list(response),
                "native_response_logprobs": list(record.response_logprobs),
                "context_suffix_ids": list(suffix),
                "history_rewritten": False,
            }
        )
    return tuple(segments), frozen_prompt


def to_agent_loop_output_dict(
    sequence: AdmittedVerlSequence,
    *,
    num_turns: int,
    generate_seconds: float = 0.0,
    tool_calls: float = 0.0,
) -> dict[str, Any]:
    """Render an admitted sequence into the pinned verl output field mapping.

    Field names and semantics match verl v0.7.1 ``AgentLoopOutput``
    (``response_mask`` 1 for LLM tokens / 0 for observation tokens;
    observation logprob slots are explicit 0.0 placeholders excluded from
    losses via the mask). The caller converts this mapping into the real verl
    type at the GPU boundary; this module never imports verl.
    """
    if not isinstance(sequence, AdmittedVerlSequence):
        raise ContractValidationError("sequence must be an AdmittedVerlSequence")
    if isinstance(num_turns, bool) or not isinstance(num_turns, int) or num_turns < 0:
        raise ContractValidationError("num_turns must be a non-negative integer")
    return {
        "prompt_ids": list(sequence.prompt_ids),
        "response_ids": list(sequence.response_ids),
        "response_mask": list(sequence.response_mask),
        "response_logprobs": list(sequence.response_logprobs),
        "reward_score": sequence.reward,
        "num_turns": num_turns,
        "metrics": {
            "generate_sequences": float(generate_seconds),
            "tool_calls": float(tool_calls),
            "num_preempted": -1,
        },
        "extra_fields": {
            "episode_id": sequence.episode_id,
            "group_id": sequence.group_id,
            "policy_fingerprint": sequence.policy_fingerprint,
            "member_id": sequence.member_id,
            "execution_bundle_checksum": sequence.execution_bundle_checksum,
            "policy_artifact_checksum": sequence.policy_artifact_checksum,
            "num_model_calls": sequence.num_model_calls,
            "num_tool_rounds": sequence.num_tool_rounds,
            "bridge_version": BRIDGE_VERSION,
        },
    }
