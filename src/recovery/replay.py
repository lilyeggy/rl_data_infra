"""Derived replay evidence for bounded FILE_NOT_FOUND recovery.

This module applies the policy to canonical events without appending synthetic
Harness decisions.  A replay report can say whether observed tool actions are
consistent with the policy plan, but only an actual Hook event can prove that a
Harness emitted the decision at runtime.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.contracts._json import sha256_json
from src.contracts.trace_event import EventStatus, EventType, TraceEvent
from src.recovery.file_not_found_policy import (
    BoundedFileNotFoundRecoveryPolicy,
    RecoveryAction,
    RecoveryDecision,
    RecoveryInput,
)


RECOVERY_REPLAY_VERSION = "file-not-found-replay/v1"


@dataclass(frozen=True, slots=True)
class RecoveryReplayStep:
    decision: RecoveryDecision
    observed_event_ids: tuple[str, ...]
    observed_tool_name: str | None
    action_alignment: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "observed_event_ids": list(self.observed_event_ids),
            "observed_tool_name": self.observed_tool_name,
            "action_alignment": self.action_alignment,
        }


@dataclass(frozen=True, slots=True)
class RecoveryReplayReport:
    episode_id: str
    source_event_ids: tuple[str, ...]
    trigger_event_id: str | None
    steps: tuple[RecoveryReplayStep, ...]
    warnings: tuple[str, ...]
    replay_version: str = RECOVERY_REPLAY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay_version": self.replay_version,
            "episode_id": self.episode_id,
            "source_event_ids": list(self.source_event_ids),
            "trigger_event_id": self.trigger_event_id,
            "steps": [step.to_dict() for step in self.steps],
            "warnings": list(self.warnings),
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


def replay_file_not_found_recovery(
    events: Iterable[TraceEvent],
    *,
    policy: BoundedFileNotFoundRecoveryPolicy | None = None,
    initial_budget: int = 1,
) -> RecoveryReplayReport:
    """Replay one policy plan against canonical events without mutating them."""

    ordered = tuple(
        sorted(events, key=lambda event: (event.sequence, event.timestamp, event.event_id))
    )
    episode_ids = {event.episode_id for event in ordered}
    if len(episode_ids) != 1:
        raise ValueError("recovery replay requires events from exactly one Episode")
    episode_id = next(iter(episode_ids), "unknown")
    source_event_ids = tuple(event.event_id for event in ordered)
    policy = policy or BoundedFileNotFoundRecoveryPolicy()
    warnings: list[str] = []
    errors = [
        event
        for event in ordered
        if event.event_type is EventType.TOOL_RESULT
        and event.attributes.get("error_code") == "FILE_NOT_FOUND"
    ]
    if not errors:
        return RecoveryReplayReport(
            episode_id=episode_id,
            source_event_ids=source_event_ids,
            trigger_event_id=None,
            steps=(),
            warnings=("no structured FILE_NOT_FOUND TOOL_RESULT was observed",),
        )

    error = errors[0]
    error_path = error.attributes.get("error_path")
    scope_key = _scope_key(error_path)
    first = policy.decide(
        RecoveryInput(
            error_code="FILE_NOT_FOUND",
            error_path=error_path if isinstance(error_path, str) else None,
            scope_key=scope_key,
            remaining_discovery_budget=initial_budget,
            trigger_event_ids=(error.event_id,),
        )
    )
    steps: list[RecoveryReplayStep] = []
    find_call, find_result = _next_find_pair(ordered, error.sequence)
    first_observed = (find_call.event_id, find_result.event_id) if find_call and find_result else ()
    first_tool = _tool_name(find_call)
    steps.append(
        RecoveryReplayStep(
            decision=first,
            observed_event_ids=first_observed,
            observed_tool_name=first_tool,
            action_alignment=_alignment(first.action, first_tool),
        )
    )

    if first.action is not RecoveryAction.DISCOVER:
        return RecoveryReplayReport(
            episode_id=episode_id,
            source_event_ids=source_event_ids,
            trigger_event_id=error.event_id,
            steps=tuple(steps),
            warnings=tuple(warnings),
        )
    if find_result is None:
        warnings.append("policy requested discovery but no successful find result was observed")
        return RecoveryReplayReport(
            episode_id=episode_id,
            source_event_ids=source_event_ids,
            trigger_event_id=error.event_id,
            steps=tuple(steps),
            warnings=tuple(warnings),
        )

    candidates = _find_candidates(find_result)
    second = policy.decide(
        RecoveryInput(
            error_code="FILE_NOT_FOUND",
            error_path=error_path if isinstance(error_path, str) else None,
            scope_key=scope_key,
            discovery_cache={scope_key: candidates},
            remaining_discovery_budget=first.budget_after or 0,
            trigger_event_ids=(error.event_id, find_result.event_id),
        )
    )
    read_call = _next_read_call(ordered, find_result.sequence)
    read_event_ids = (read_call.event_id,) if read_call else ()
    read_tool = _tool_name(read_call)
    steps.append(
        RecoveryReplayStep(
            decision=second,
            observed_event_ids=read_event_ids,
            observed_tool_name=read_tool,
            action_alignment=_alignment(second.action, read_tool),
        )
    )
    if second.action is RecoveryAction.READ and read_call is None:
        warnings.append("policy selected a unique candidate but no subsequent read was observed")
    return RecoveryReplayReport(
        episode_id=episode_id,
        source_event_ids=source_event_ids,
        trigger_event_id=error.event_id,
        steps=tuple(steps),
        warnings=tuple(warnings),
    )


def _scope_key(error_path: Any) -> str:
    if not isinstance(error_path, str) or not error_path.strip():
        return "."
    return posixpath.dirname(error_path.replace("\\", "/")) or "."


def _tool_name(event: TraceEvent | None) -> str | None:
    if event is None:
        return None
    value = event.attributes.get("tool_name")
    return value if isinstance(value, str) else None


def _next_find_pair(
    events: tuple[TraceEvent, ...], after_sequence: int
) -> tuple[TraceEvent | None, TraceEvent | None]:
    call = next(
        (
            event
            for event in events
            if event.sequence > after_sequence
            and event.event_type is EventType.TOOL_CALL
            and _tool_name(event) == "find"
        ),
        None,
    )
    if call is None:
        return None, None
    result = next(
        (
            event
            for event in events
            if event.sequence > call.sequence
            and event.event_type is EventType.TOOL_RESULT
            and event.span_id == call.span_id
            and event.status is EventStatus.SUCCEEDED
        ),
        None,
    )
    return call, result


def _next_read_call(events: tuple[TraceEvent, ...], after_sequence: int) -> TraceEvent | None:
    return next(
        (
            event
            for event in events
            if event.sequence > after_sequence
            and event.event_type is EventType.TOOL_CALL
            and _tool_name(event) == "read"
        ),
        None,
    )


def _find_candidates(event: TraceEvent) -> tuple[str, ...]:
    result = event.attributes.get("result")
    texts: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            text = value.get("text")
            if isinstance(text, str):
                texts.extend(line.strip() for line in text.splitlines() if line.strip())
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(result)
    return tuple(sorted(set(texts)))


def _alignment(action: RecoveryAction, observed_tool_name: str | None) -> str:
    expected = {
        RecoveryAction.DISCOVER: "find",
        RecoveryAction.READ: "read",
        RecoveryAction.TERMINATE: None,
        RecoveryAction.NO_ACTION: None,
    }[action]
    if expected == observed_tool_name:
        return "MATCH"
    if expected is None and observed_tool_name is None:
        return "MATCH"
    if observed_tool_name is None:
        return "NO_OBSERVED_ACTION"
    return "OBSERVED_ACTION_DIFF"
