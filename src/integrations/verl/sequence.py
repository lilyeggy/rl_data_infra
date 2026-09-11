"""Assemble one Pi episode into a single verl training sequence.

Single-episode-to-single-sequence contract (closeout §3.4):

- The first model request's frozen prompt tokens are the sequence prompt.
- Native generated tokens are kept verbatim; decode/re-encode drift must not
  rewrite them.
- Tool observations and template separators are appended to the context only;
  their mask is 0.
- Observation logprob slots use 0.0 as an explicit non-probability placeholder;
  all losses and statistics must exclude them via the mask.
- Any history rewrite (compaction, truncation, branch, rollback) rejects the
  episode instead of re-tokenizing final messages.

This module never tokenizes text. It assembles token evidence that the model
bridge captured during the live Pi run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.errors import ContractValidationError

SEQUENCE_ASSEMBLER_VERSION = "verl-sequence-assembler/v1"

_OBSERVATION_LOGPROB_PLACEHOLDER = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class AssembledSequence:
    """One episode rendered as a single verl response sequence."""

    episode_id: str
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    response_logprobs: tuple[float, ...]
    num_model_calls: int
    num_tool_rounds: int
    schema_version: str = SEQUENCE_ASSEMBLER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "prompt_ids": list(self.prompt_ids),
            "response_ids": list(self.response_ids),
            "response_mask": list(self.response_mask),
            "response_logprobs": list(self.response_logprobs),
            "num_model_calls": self.num_model_calls,
            "num_tool_rounds": self.num_tool_rounds,
        }


def _checked_int_list(value: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContractValidationError(f"{field_name} must be an array")
    output: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ContractValidationError(f"{field_name} must contain integers")
        output.append(item)
    return tuple(output)


def _checked_float_list(value: Any, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContractValidationError(f"{field_name} must be an array")
    output: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ContractValidationError(f"{field_name} must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ContractValidationError(f"{field_name} must be finite")
        output.append(number)
    return tuple(output)


def _checked_mask(value: Any, field_name: str) -> tuple[int, ...]:
    mask = _checked_int_list(value, field_name)
    if any(item not in (0, 1) for item in mask):
        raise ContractValidationError(f"{field_name} must contain only 0/1")
    return mask


def assemble_episode_sequence(
    *,
    episode_id: str,
    prompt_ids: Sequence[int],
    per_call_segments: Sequence[Mapping[str, Any]],
) -> AssembledSequence:
    """Assemble native per-call evidence into one verl response sequence.

    Each segment describes one model call in execution order::

        {
            "native_response_ids": [...],     # required, non-empty
            "native_response_logprobs": [...],# required, aligned, finite
            "context_suffix_ids": [...],      # optional: observation/template
                                             # tokens appended after this call
            "history_rewritten": False,       # True rejects the episode
        }

    The first call additionally provides the frozen prompt via ``prompt_ids``.
    ``context_suffix_ids`` of call N is the exact token block the bridge fed as
    context before call N+1; no re-tokenization is performed here.
    """

    if not episode_id or not isinstance(episode_id, str):
        raise ContractValidationError("episode_id must be a non-empty string")
    prompt = _checked_int_list(prompt_ids, "prompt_ids")
    if not prompt:
        raise ContractValidationError("prompt_ids must be non-empty")
    if (
        not isinstance(per_call_segments, Sequence)
        or isinstance(per_call_segments, (str, bytes))
        or not per_call_segments
    ):
        raise ContractValidationError("per_call_segments must be a non-empty array")

    response_ids: list[int] = []
    response_mask: list[int] = []
    response_logprobs: list[float] = []
    tool_rounds = 0
    for index, segment in enumerate(per_call_segments):
        field = f"per_call_segments[{index}]"
        if not isinstance(segment, Mapping):
            raise ContractValidationError(f"{field} must be an object")
        if segment.get("history_rewritten", False):
            raise ContractValidationError(
                f"{field} rewrote generation history; episode is not trainable"
            )
        native_ids = _checked_int_list(
            segment.get("native_response_ids"), f"{field}.native_response_ids"
        )
        if not native_ids:
            raise ContractValidationError(
                f"{field}.native_response_ids must be non-empty"
            )
        native_logprobs = _checked_float_list(
            segment.get("native_response_logprobs"),
            f"{field}.native_response_logprobs",
        )
        if len(native_logprobs) != len(native_ids):
            raise ContractValidationError(
                f"{field} response/logprob alignment invalid"
            )
        context_suffix = _checked_int_list(
            segment.get("context_suffix_ids", ()), f"{field}.context_suffix_ids"
        )
        if len(segment.get("context_suffix_ids", ())) != len(context_suffix):
            raise ContractValidationError(f"{field}.context_suffix_ids invalid")

        response_ids.extend(native_ids)
        response_mask.extend([1] * len(native_ids))
        response_logprobs.extend(native_logprobs)
        if context_suffix:
            response_ids.extend(context_suffix)
            response_mask.extend([0] * len(context_suffix))
            response_logprobs.extend(
                [_OBSERVATION_LOGPROB_PLACEHOLDER] * len(context_suffix)
            )
            tool_rounds += 1

    mask = _checked_mask(response_mask, "response_mask")
    if not any(mask):
        raise ContractValidationError("assembled sequence has no trainable mask")
    if not (len(response_ids) == len(mask) == len(response_logprobs)):
        raise ContractValidationError("assembled sequence alignment invalid")
    return AssembledSequence(
        episode_id=episode_id,
        prompt_ids=prompt,
        response_ids=tuple(response_ids),
        response_mask=mask,
        response_logprobs=tuple(response_logprobs),
        num_model_calls=len(per_call_segments),
        num_tool_rounds=tool_rounds,
    )
