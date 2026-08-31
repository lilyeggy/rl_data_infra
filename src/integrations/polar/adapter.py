"""Version-pinned adapter for Polar stable TaskResult/TaskStatus payloads.

Polar expands one task into ``num_samples`` sessions.  This module therefore
allocates one :class:`ExecutionIdentity` per terminal session and never treats
the Polar task id as an episode id.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.contracts._json import freeze_json, validate_sha256
from src.contracts._validation import optional_text, required_text
from src.contracts.execution_identity import ExecutionIdentity
from src.errors import ContractValidationError
from src.producers.base import (
    ProducerArtifact,
    ProducerCapability,
    ProducerExecutionStatus,
)

POLAR_ADAPTER_VERSION = "polar-stable-task-result/v1"
IDENTITY_METADATA_KEY = "agent_data_plane"
_TERMINAL_SESSION_STATUSES = frozenset({"COMPLETED", "ERROR", "TIMEOUT"})
_TERMINAL_TASK_STATUSES = frozenset({"completed", "failed"})


class PolarSchemaError(ContractValidationError):
    """Polar returned a terminal payload that violates the pinned schema."""


class PolarResultNotReady(RuntimeError):
    """The requested Polar task has not reached a terminal state."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PolarBatchContext:
    """Immutable identity/provenance allocated before a Polar task is submitted."""

    run_id: str
    logical_task_id: str
    polar_task_id: str
    expected_samples: int
    producer_version: str
    group_id: str
    policy_fingerprint: str | None = None
    sampling_fingerprint: str | None = None
    evaluator_fingerprint: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "logical_task_id",
            "polar_task_id",
            "producer_version",
            "group_id",
        ):
            required_text(getattr(self, name), name)
        if (
            isinstance(self.expected_samples, bool)
            or not isinstance(self.expected_samples, int)
            or self.expected_samples < 1
        ):
            raise ContractValidationError("expected_samples must be a positive integer")
        for name in (
            "policy_fingerprint",
            "sampling_fingerprint",
            "evaluator_fingerprint",
        ):
            optional_text(getattr(self, name), name)
            validate_sha256(getattr(self, name), name)

    def identity_metadata(self) -> dict[str, Any]:
        """Metadata injected into Polar and expected back on every trajectory."""

        return {
            "schema_version": POLAR_ADAPTER_VERSION,
            "run_id": self.run_id,
            "logical_task_id": self.logical_task_id,
            "polar_task_id": self.polar_task_id,
            "group_id": self.group_id,
            "producer_version": self.producer_version,
            "policy_fingerprint": self.policy_fingerprint,
            "sampling_fingerprint": self.sampling_fingerprint,
            "evaluator_fingerprint": self.evaluator_fingerprint,
        }


def adapt_polar_task_result(
    value: Mapping[str, Any],
    *,
    context: PolarBatchContext,
) -> tuple[ProducerArtifact, ...]:
    """Convert a terminal Polar task result into one artifact per session.

    The adapter accepts both the documented ``TaskStatus`` polling shape and
    terminal ``TaskResult`` shape.  Counts and identity metadata must agree
    exactly with the pre-submission context.
    """

    payload = _mapping(value, "Polar task result")
    if payload.get("task_id") != context.polar_task_id:
        raise PolarSchemaError("Polar task result task_id mismatch")
    task_status = payload.get("status")
    if not isinstance(task_status, str):
        raise PolarSchemaError("Polar task result status must be a string")
    if task_status not in _TERMINAL_TASK_STATUSES:
        raise PolarResultNotReady(
            f"Polar task {context.polar_task_id!r} is not terminal: {task_status!r}"
        )

    total_sessions = payload.get("total_sessions")
    if total_sessions is not None and total_sessions != context.expected_samples:
        raise PolarSchemaError("Polar total_sessions does not match expected_samples")
    completed_sessions = payload.get("completed_sessions")
    if completed_sessions is not None and completed_sessions != context.expected_samples:
        raise PolarSchemaError("terminal Polar task has incomplete sessions")

    results = _sequence(payload.get("results"), "Polar task result.results")
    if len(results) != context.expected_samples:
        raise PolarSchemaError("Polar result count does not match expected_samples")

    sessions: list[Mapping[str, Any]] = []
    seen_session_ids: set[str] = set()
    for index, raw_session in enumerate(results):
        session = _mapping(raw_session, f"results[{index}]")
        session_id = required_text(session.get("session_id"), f"results[{index}].session_id")
        if session_id in seen_session_ids:
            raise PolarSchemaError(f"duplicate Polar session_id {session_id!r}")
        seen_session_ids.add(session_id)
        sessions.append(session)

    artifacts = []
    for attempt_id, session in enumerate(
        sorted(sessions, key=lambda item: str(item["session_id"])), start=1
    ):
        artifacts.append(_adapt_session(session, context=context, attempt_id=attempt_id))
    return tuple(artifacts)


def _adapt_session(
    session: Mapping[str, Any],
    *,
    context: PolarBatchContext,
    attempt_id: int,
) -> ProducerArtifact:
    session_id = required_text(session.get("session_id"), "session_id")
    if session.get("task_id") != context.polar_task_id:
        raise PolarSchemaError(f"session {session_id!r} task_id mismatch")
    status = session.get("status")
    if status not in _TERMINAL_SESSION_STATUSES:
        raise PolarSchemaError(f"session {session_id!r} has non-terminal status")

    trajectory = _mapping(session.get("trajectory"), f"session {session_id}.trajectory")
    trajectory_status = trajectory.get("status")
    if trajectory_status != status:
        raise PolarSchemaError(
            f"session {session_id!r} status disagrees with trajectory status"
        )
    metadata = _mapping(
        trajectory.get("metadata", {}), f"session {session_id}.trajectory.metadata"
    )
    identity_metadata = _mapping(
        metadata.get(IDENTITY_METADATA_KEY),
        f"session {session_id}.trajectory.metadata.{IDENTITY_METADATA_KEY}",
    )
    if dict(identity_metadata) != context.identity_metadata():
        raise PolarSchemaError(f"session {session_id!r} data-plane identity mismatch")

    traces = _sequence(trajectory.get("traces", []), f"session {session_id}.trajectory.traces")
    trace_facts = [
        _validate_trace(_mapping(trace, f"session {session_id}.traces[{index}]"), index=index)
        for index, trace in enumerate(traces)
    ]

    capabilities = {ProducerCapability.ASYNC_SUBMISSION}
    issues: list[str] = []
    if not trace_facts:
        issues.append("trajectory has no traces")
    else:
        if all(fact["has_token_ids"] for fact in trace_facts):
            capabilities.add(ProducerCapability.TOKEN_IDS)
        else:
            issues.append("one or more traces have no response token ids")
        if all(fact["has_loss_mask"] for fact in trace_facts):
            capabilities.add(ProducerCapability.ACTION_MASK)
        else:
            issues.append("one or more traces have no aligned non-empty loss mask")
        if all(fact["has_logprobs"] for fact in trace_facts):
            capabilities.add(ProducerCapability.BEHAVIOR_LOGPROBS)
        else:
            issues.append("one or more traces have no aligned behavior logprobs")
        if (
            status == "COMPLETED"
            and context.evaluator_fingerprint is not None
            and all(
            fact["has_reward"] for fact in trace_facts
            )
        ):
            capabilities.add(ProducerCapability.VERIFIER_EVIDENCE)
        elif status == "COMPLETED" and context.evaluator_fingerprint is not None:
            issues.append("configured evaluator produced incomplete reward evidence")

    if context.policy_fingerprint is not None:
        capabilities.add(ProducerCapability.POLICY_VERSION)

    execution_status = {
        "COMPLETED": ProducerExecutionStatus.COMPLETED,
        "ERROR": ProducerExecutionStatus.INFRA_INVALID,
        "TIMEOUT": ProducerExecutionStatus.TIMEOUT,
    }[status]
    raw_payload = freeze_json(
        {
            "adapter_version": POLAR_ADAPTER_VERSION,
            "polar_task_id": context.polar_task_id,
            "polar_session_id": session_id,
            "status": status,
            "trajectory": trajectory,
            "timing": session.get("timing", {}),
            "node_id": session.get("node_id"),
            "error": session.get("error"),
            "metadata": session.get("metadata", {}),
            "evaluator_fingerprint": context.evaluator_fingerprint,
        }
    )
    assert isinstance(raw_payload, Mapping)
    return ProducerArtifact(
        identity=ExecutionIdentity(
            run_id=context.run_id,
            task_id=context.logical_task_id,
            episode_id=f"polar/{context.polar_task_id}/{session_id}",
            attempt_id=attempt_id,
            producer_id="polar",
            producer_version=context.producer_version,
            group_id=context.group_id,
            policy_fingerprint=context.policy_fingerprint,
            sampling_fingerprint=context.sampling_fingerprint,
        ),
        status=execution_status,
        capabilities=frozenset(capabilities),
        payload=raw_payload,
        issues=tuple(issues),
    )


def _validate_trace(trace: Mapping[str, Any], *, index: int) -> dict[str, bool]:
    response_ids = _integer_sequence(trace.get("response_ids", []), f"traces[{index}].response_ids")
    _integer_sequence(trace.get("prompt_ids", []), f"traces[{index}].prompt_ids")
    loss_mask = _integer_sequence(trace.get("loss_mask", []), f"traces[{index}].loss_mask")
    if any(item not in (0, 1) for item in loss_mask):
        raise PolarSchemaError(f"traces[{index}].loss_mask must contain only 0 or 1")
    if loss_mask and len(loss_mask) != len(response_ids):
        raise PolarSchemaError(f"traces[{index}] loss_mask/response_ids length mismatch")

    raw_logprobs = trace.get("response_logprobs")
    logprobs: tuple[float, ...] | None = None
    if raw_logprobs is not None:
        values = _sequence(raw_logprobs, f"traces[{index}].response_logprobs")
        parsed = []
        for position, item in enumerate(values):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise PolarSchemaError(
                    f"traces[{index}].response_logprobs[{position}] must be numeric"
                )
            number = float(item)
            if not math.isfinite(number):
                raise PolarSchemaError(f"traces[{index}] has non-finite logprob")
            parsed.append(number)
        logprobs = tuple(parsed)
        if len(logprobs) != len(response_ids):
            raise PolarSchemaError(f"traces[{index}] logprob/response_ids length mismatch")

    reward = trace.get("reward")
    has_reward = reward is not None
    if has_reward:
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise PolarSchemaError(f"traces[{index}].reward must be numeric or null")
        if not math.isfinite(float(reward)):
            raise PolarSchemaError(f"traces[{index}].reward must be finite")

    return {
        "has_token_ids": bool(response_ids),
        "has_loss_mask": bool(response_ids) and len(loss_mask) == len(response_ids),
        "has_logprobs": bool(response_ids)
        and logprobs is not None
        and len(logprobs) == len(response_ids),
        "has_reward": has_reward,
    }


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolarSchemaError(f"{field_name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise PolarSchemaError(f"{field_name} contains a non-string key")
    return value


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise PolarSchemaError(f"{field_name} must be an array")
    return value


def _integer_sequence(value: Any, field_name: str) -> tuple[int, ...]:
    items = _sequence(value, field_name)
    parsed = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, int):
            raise PolarSchemaError(f"{field_name}[{index}] must be an integer")
        parsed.append(item)
    return tuple(parsed)
