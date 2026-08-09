"""Canonical producer-agnostic rollout record v1."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

from src.contracts._json import freeze_json, sha256_json, thaw_json, validate_sha256
from src.errors import ContractValidationError


SCHEMA_VERSION = "rollout-record/v1"


class RolloutStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"


class ComponentStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CRASHED = "crashed"
    NOT_RUN = "not_run"
    UNKNOWN = "unknown"


class VerifierStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"
    NOT_RUN = "not_run"
    UNKNOWN = "unknown"


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field_name} must be a non-empty string")
    return value


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _integer_tuple(value: Any, field_name: str, *, binary: bool = False) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ContractValidationError(f"{field_name} must be an array")
    normalized: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ContractValidationError(f"{field_name}[{index}] must be an integer")
        if binary and item not in (0, 1):
            raise ContractValidationError(f"{field_name}[{index}] must be 0 or 1")
        if not binary and item < 0:
            raise ContractValidationError(f"{field_name}[{index}] must be non-negative")
        normalized.append(item)
    return tuple(normalized)


def _float_tuple(value: Any, field_name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ContractValidationError(f"{field_name} must be an array")
    normalized: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ContractValidationError(f"{field_name}[{index}] must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ContractValidationError(f"{field_name}[{index}] must be finite")
        normalized.append(number)
    return tuple(normalized)


def _parse_timestamp(value: str | None, field_name: str) -> datetime | None:
    if value is None:
        return None
    _optional_text(value, field_name)
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ContractValidationError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractValidationError(f"{field_name} must include a UTC offset")
    return parsed


@dataclass(frozen=True, slots=True, kw_only=True)
class RolloutRecord:
    """One canonical trajectory without any Polar or Trainer runtime dependency.

    Missing training fields remain ``None``. They are surfaced through capability
    checks instead of being guessed, retokenized, or filled with zeroes.
    """

    trajectory_id: str
    task_id: str
    source_type: str
    source_record_id: str

    group_id: str | None = None
    policy_version: str | None = None

    token_ids: tuple[int, ...] | None = None
    prompt_token_count: int | None = None
    action_mask: tuple[int, ...] | None = None
    loss_mask: tuple[int, ...] | None = None
    old_logprobs: tuple[float, ...] | None = None
    reward: float | None = None

    rollout_status: RolloutStatus = RolloutStatus.UNKNOWN
    termination_reason: str | None = None
    runtime_status: ComponentStatus = ComponentStatus.UNKNOWN
    harness_status: ComponentStatus = ComponentStatus.UNKNOWN
    model_backend_status: ComponentStatus = ComponentStatus.UNKNOWN
    verifier_status: VerifierStatus = VerifierStatus.UNKNOWN

    verifier_evidence_ref: str | None = None
    source_payload_ref: str | None = None
    source_payload_sha256: str | None = None
    model_id: str | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    tool_events: tuple[Mapping[str, Any], ...] | None = None
    opaque_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("trajectory_id", "task_id", "source_type", "source_record_id"):
            _required_text(getattr(self, name), name)
        for name in (
            "group_id",
            "policy_version",
            "termination_reason",
            "verifier_evidence_ref",
            "source_payload_ref",
            "model_id",
            "model_revision",
            "tokenizer_revision",
        ):
            _optional_text(getattr(self, name), name)

        if self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(
                f"schema_version must be {SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )
        for name, enum_type in (
            ("rollout_status", RolloutStatus),
            ("runtime_status", ComponentStatus),
            ("harness_status", ComponentStatus),
            ("model_backend_status", ComponentStatus),
            ("verifier_status", VerifierStatus),
        ):
            if not isinstance(getattr(self, name), enum_type):
                raise ContractValidationError(f"{name} must be a {enum_type.__name__}")

        token_ids = _integer_tuple(self.token_ids, "token_ids")
        action_mask = _integer_tuple(self.action_mask, "action_mask", binary=True)
        loss_mask = _integer_tuple(self.loss_mask, "loss_mask", binary=True)
        old_logprobs = _float_tuple(self.old_logprobs, "old_logprobs")
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "action_mask", action_mask)
        object.__setattr__(self, "loss_mask", loss_mask)
        object.__setattr__(self, "old_logprobs", old_logprobs)

        if action_mask is not None and loss_mask is not None:
            raise ContractValidationError("provide action_mask or loss_mask, not both")
        mask = action_mask if action_mask is not None else loss_mask
        if mask is not None and token_ids is None:
            raise ContractValidationError("a mask cannot be present without token_ids")
        if token_ids is not None and mask is not None and len(token_ids) != len(mask):
            raise ContractValidationError("token_ids and mask must have the same length")
        if self.prompt_token_count is not None:
            if isinstance(self.prompt_token_count, bool) or not isinstance(
                self.prompt_token_count, int
            ):
                raise ContractValidationError("prompt_token_count must be an integer")
            if token_ids is None:
                raise ContractValidationError(
                    "prompt_token_count cannot be present without token_ids"
                )
            if not 0 <= self.prompt_token_count <= len(token_ids):
                raise ContractValidationError(
                    "prompt_token_count must be between zero and token_ids length"
                )
        if old_logprobs is not None:
            if token_ids is None:
                raise ContractValidationError("old_logprobs cannot be present without token_ids")
            valid_lengths = {len(token_ids)}
            if mask is not None:
                valid_lengths.add(sum(mask))
            if self.prompt_token_count is not None:
                valid_lengths.add(len(token_ids) - self.prompt_token_count)
            if len(old_logprobs) not in valid_lengths:
                raise ContractValidationError(
                    "old_logprobs length must match all tokens or trainable tokens"
                )

        if self.reward is not None:
            if isinstance(self.reward, bool) or not isinstance(self.reward, (int, float)):
                raise ContractValidationError("reward must be numeric")
            reward = float(self.reward)
            if not math.isfinite(reward):
                raise ContractValidationError("reward must be finite")
            object.__setattr__(self, "reward", reward)

        validate_sha256(self.source_payload_sha256, "source_payload_sha256")
        started = _parse_timestamp(self.started_at, "started_at")
        ended = _parse_timestamp(self.ended_at, "ended_at")
        if started is not None and ended is not None and ended < started:
            raise ContractValidationError("ended_at cannot be earlier than started_at")

        if self.tool_events is not None:
            if not isinstance(self.tool_events, (list, tuple)):
                raise ContractValidationError("tool_events must be an array")
            frozen_events = tuple(
                freeze_json(event, f"$.tool_events[{index}]")
                for index, event in enumerate(self.tool_events)
            )
            if any(not isinstance(event, Mapping) for event in frozen_events):
                raise ContractValidationError("each tool event must be an object")
            object.__setattr__(self, "tool_events", frozen_events)
        object.__setattr__(self, "opaque_metadata", freeze_json(self.opaque_metadata))
        if not isinstance(self.opaque_metadata, Mapping):
            raise ContractValidationError("opaque_metadata must be an object")

    @property
    def trainable_token_count(self) -> int:
        mask = self.action_mask if self.action_mask is not None else self.loss_mask
        return sum(mask) if mask is not None else 0

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "group_id": self.group_id,
            "policy_version": self.policy_version,
            "source_type": self.source_type,
            "source_record_id": self.source_record_id,
            "token_ids": list(self.token_ids) if self.token_ids is not None else None,
            "prompt_token_count": self.prompt_token_count,
            "action_mask": list(self.action_mask) if self.action_mask is not None else None,
            "loss_mask": list(self.loss_mask) if self.loss_mask is not None else None,
            "old_logprobs": list(self.old_logprobs) if self.old_logprobs is not None else None,
            "reward": self.reward,
            "rollout_status": self.rollout_status.value,
            "termination_reason": self.termination_reason,
            "runtime_status": self.runtime_status.value,
            "harness_status": self.harness_status.value,
            "model_backend_status": self.model_backend_status.value,
            "verifier_status": self.verifier_status.value,
            "verifier_evidence_ref": self.verifier_evidence_ref,
            "source_payload_ref": self.source_payload_ref,
            "source_payload_sha256": self.source_payload_sha256,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "tool_events": thaw_json(self.tool_events),
            "opaque_metadata": thaw_json(self.opaque_metadata),
        }

    def semantic_dict(self) -> dict[str, Any]:
        """Return the record content excluding the current source envelope."""

        value = self.to_dict()
        for key in (
            "source_type",
            "source_record_id",
            "source_payload_ref",
            "source_payload_sha256",
        ):
            value.pop(key)
        return value

    def with_source_envelope(
        self,
        *,
        source_type: str,
        source_record_id: str,
        source_payload_ref: str | None,
        source_payload_sha256: str,
    ) -> "RolloutRecord":
        return replace(
            self,
            source_type=source_type,
            source_record_id=source_record_id,
            source_payload_ref=source_payload_ref,
            source_payload_sha256=source_payload_sha256,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RolloutRecord":
        if not isinstance(value, Mapping):
            raise ContractValidationError("rollout record must be an object")
        known = {
            "schema_version",
            "trajectory_id",
            "task_id",
            "group_id",
            "policy_version",
            "source_type",
            "source_record_id",
            "token_ids",
            "prompt_token_count",
            "action_mask",
            "loss_mask",
            "old_logprobs",
            "reward",
            "rollout_status",
            "termination_reason",
            "runtime_status",
            "harness_status",
            "model_backend_status",
            "verifier_status",
            "verifier_evidence_ref",
            "source_payload_ref",
            "source_payload_sha256",
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "started_at",
            "ended_at",
            "tool_events",
            "opaque_metadata",
        }
        unknown = sorted(set(value) - known)
        if unknown:
            raise ContractValidationError(
                "unknown rollout-record fields must be placed in opaque_metadata: "
                + ", ".join(unknown)
            )
        required = ("trajectory_id", "task_id", "source_type", "source_record_id")
        missing = [name for name in required if name not in value]
        if missing:
            raise ContractValidationError("missing required fields: " + ", ".join(missing))

        def enum_value(name: str, enum_type: type[Enum], default: Enum) -> Enum:
            raw = value.get(name, default.value)
            try:
                return enum_type(raw)
            except (TypeError, ValueError) as exc:
                raise ContractValidationError(f"invalid {name}: {raw!r}") from exc

        return cls(
            trajectory_id=value["trajectory_id"],
            task_id=value["task_id"],
            source_type=value["source_type"],
            source_record_id=value["source_record_id"],
            group_id=value.get("group_id"),
            policy_version=value.get("policy_version"),
            token_ids=value.get("token_ids"),
            prompt_token_count=value.get("prompt_token_count"),
            action_mask=value.get("action_mask"),
            loss_mask=value.get("loss_mask"),
            old_logprobs=value.get("old_logprobs"),
            reward=value.get("reward"),
            rollout_status=enum_value("rollout_status", RolloutStatus, RolloutStatus.UNKNOWN),
            termination_reason=value.get("termination_reason"),
            runtime_status=enum_value("runtime_status", ComponentStatus, ComponentStatus.UNKNOWN),
            harness_status=enum_value("harness_status", ComponentStatus, ComponentStatus.UNKNOWN),
            model_backend_status=enum_value(
                "model_backend_status", ComponentStatus, ComponentStatus.UNKNOWN
            ),
            verifier_status=enum_value(
                "verifier_status", VerifierStatus, VerifierStatus.UNKNOWN
            ),
            verifier_evidence_ref=value.get("verifier_evidence_ref"),
            source_payload_ref=value.get("source_payload_ref"),
            source_payload_sha256=value.get("source_payload_sha256"),
            model_id=value.get("model_id"),
            model_revision=value.get("model_revision"),
            tokenizer_revision=value.get("tokenizer_revision"),
            started_at=value.get("started_at"),
            ended_at=value.get("ended_at"),
            tool_events=value.get("tool_events"),
            opaque_metadata=value.get("opaque_metadata", {}),
            schema_version=value.get("schema_version", SCHEMA_VERSION),
        )
