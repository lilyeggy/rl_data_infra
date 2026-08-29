"""Canonical observable execution event contract.

This module intentionally contains only the public shape for the first Day 5
exercise.  The contract invariants and deterministic serialization are the
learner-owned core implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Mapping

from src.contracts._json import freeze_json, sha256_json, thaw_json
from src.errors import ContractValidationError


SCHEMA_VERSION = "trace-event/v1"


class EventType(str, Enum):
    MODEL_REQUEST = "MODEL_REQUEST"
    MODEL_RESPONSE = "MODEL_RESPONSE"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    SANDBOX_STARTED = "SANDBOX_STARTED"
    SANDBOX_COMMAND = "SANDBOX_COMMAND"
    SANDBOX_FINISHED = "SANDBOX_FINISHED"
    VERIFICATION_STARTED = "VERIFICATION_STARTED"
    VERIFICATION_FINISHED = "VERIFICATION_FINISHED"
    EPISODE_FINISHED = "EPISODE_FINISHED"
    HARNESS_DECISION = "HARNESS_DECISION"
    CONTEXT_SELECTED = "CONTEXT_SELECTED"
    CONTEXT_COMPACTED = "CONTEXT_COMPACTED"
    RETRY_SCHEDULED = "RETRY_SCHEDULED"
    TERMINATION_DECIDED = "TERMINATION_DECIDED"


class EventComponent(str, Enum):
    MODEL = "MODEL"
    HARNESS = "HARNESS"
    TOOL = "TOOL"
    SANDBOX = "SANDBOX"
    MODEL_BACKEND = "MODEL_BACKEND"
    EVALUATOR = "EVALUATOR"
    EXTERNAL_SERVICE = "EXTERNAL_SERVICE"


class EventStatus(str, Enum):
    STARTED = "STARTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True, kw_only=True)
class TraceEvent:
    """One immutable observable fact in an Agent execution.

    Implement the validation and serialization behavior specified by
    ``docs/data-contract.md`` and ``tests/contracts/test_trace_event.py``.
    """

    event_id: str
    run_id: str
    episode_id: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    sequence: int
    timestamp: str
    event_type: EventType
    component: EventComponent
    status: EventStatus
    attempt: int
    attributes: Mapping[str, Any] = field(default_factory=dict)
    artifact_refs: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("event_id", "run_id", "episode_id", "trace_id", "span_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractValidationError(f"{name} must be a non-empty string")

        if self.parent_span_id is not None:
            if not isinstance(self.parent_span_id, str) or not self.parent_span_id.strip():
                raise ContractValidationError(
                    "parent_span_id must be null or a non-empty string"
                )
            if self.parent_span_id == self.span_id:
                raise ContractValidationError("parent_span_id cannot equal span_id")

        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ContractValidationError("sequence must be a non-negative integer")
        if (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
            or self.attempt < 1
        ):
            raise ContractValidationError("attempt must be a positive integer")

        if not isinstance(self.timestamp, str) or not self.timestamp:
            raise ContractValidationError("timestamp must be a UTC ISO-8601 instant")
        candidate = (
            self.timestamp[:-1] + "+00:00"
            if self.timestamp.endswith("Z")
            else self.timestamp
        )
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise ContractValidationError(
                "timestamp must be a UTC ISO-8601 instant"
            ) from exc
        if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
            raise ContractValidationError("timestamp must use the UTC offset")

        for name, enum_type in (
            ("event_type", EventType),
            ("component", EventComponent),
            ("status", EventStatus),
        ):
            if not isinstance(getattr(self, name), enum_type):
                raise ContractValidationError(f"{name} must be a {enum_type.__name__}")

        if not isinstance(self.attributes, Mapping):
            raise ContractValidationError("attributes must be an object")
        object.__setattr__(self, "attributes", freeze_json(self.attributes, "$.attributes"))

        if not isinstance(self.artifact_refs, (tuple, list)):
            raise ContractValidationError("artifact_refs must be an array")
        normalized_refs: list[str] = []
        for index, value in enumerate(self.artifact_refs):
            if not isinstance(value, str) or not value.strip():
                raise ContractValidationError(
                    f"artifact_refs[{index}] must be a non-empty string"
                )
            normalized_refs.append(value)
        if len(set(normalized_refs)) != len(normalized_refs):
            raise ContractValidationError("artifact_refs must not contain duplicates")
        object.__setattr__(self, "artifact_refs", tuple(normalized_refs))

        if self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(
                f"schema_version must be {SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "component": self.component.value,
            "status": self.status.value,
            "attempt": self.attempt,
            "attributes": thaw_json(self.attributes),
            "artifact_refs": list(self.artifact_refs),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TraceEvent":
        if not isinstance(value, Mapping):
            raise ContractValidationError("TraceEvent payload must be an object")
        allowed = {
            "schema_version",
            "event_id",
            "run_id",
            "episode_id",
            "trace_id",
            "span_id",
            "parent_span_id",
            "sequence",
            "timestamp",
            "event_type",
            "component",
            "status",
            "attempt",
            "attributes",
            "artifact_refs",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ContractValidationError(
                "unknown TraceEvent fields "
                f"{unknown}; producer-specific data belongs in attributes"
            )
        missing = sorted(allowed - set(value))
        if missing:
            raise ContractValidationError(f"missing TraceEvent fields: {missing}")
        if value["schema_version"] != SCHEMA_VERSION:
            raise ContractValidationError(
                f"schema_version must be {SCHEMA_VERSION!r}, "
                f"got {value['schema_version']!r}"
            )
        try:
            event_type = EventType(value["event_type"])
            component = EventComponent(value["component"])
            status = EventStatus(value["status"])
        except (TypeError, ValueError) as exc:
            raise ContractValidationError(f"invalid TraceEvent enum value: {exc}") from exc
        return cls(
            schema_version=value["schema_version"],
            event_id=value["event_id"],
            run_id=value["run_id"],
            episode_id=value["episode_id"],
            trace_id=value["trace_id"],
            span_id=value["span_id"],
            parent_span_id=value["parent_span_id"],
            sequence=value["sequence"],
            timestamp=value["timestamp"],
            event_type=event_type,
            component=component,
            status=status,
            attempt=value["attempt"],
            attributes=value["attributes"],
            artifact_refs=tuple(value["artifact_refs"]),
        )

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())
