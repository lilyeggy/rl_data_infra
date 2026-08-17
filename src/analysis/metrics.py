"""Deterministic Episode metrics with explicit missing-observability semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from src.contracts._json import canonical_json_bytes, sha256_json
from src.contracts.agent_episode import AgentEpisode, CaptureCapability
from src.contracts.trace_event import EventComponent, EventStatus, EventType, TraceEvent


METRIC_VERSION = "episode-metrics/v1"


def _timestamp(value: str) -> datetime:
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(candidate)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    episode_id: str
    task_status: str
    execution_validity: str
    integrity_state: str
    duration_ms: float
    turn_count: int
    tool_call_count: int | None
    tool_error_count: int | None
    duplicate_action_count: int | None
    duplicate_action_rate: float | None
    loop_count: int | None
    tool_error_recovery_rate: float | None
    verification_attempts: int
    input_tokens: int | None
    output_tokens: int | None
    model_latency_ms: float | None
    tool_latency_ms: float | None
    verifier_latency_ms: float | None
    estimated_cost: float | None
    not_observable: tuple[str, ...]
    input_episode_checksum: str
    metric_version: str = METRIC_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_version": self.metric_version,
            "episode_id": self.episode_id,
            "task_status": self.task_status,
            "execution_validity": self.execution_validity,
            "integrity_state": self.integrity_state,
            "duration_ms": self.duration_ms,
            "turn_count": self.turn_count,
            "tool_call_count": self.tool_call_count,
            "tool_error_count": self.tool_error_count,
            "duplicate_action_count": self.duplicate_action_count,
            "duplicate_action_rate": self.duplicate_action_rate,
            "loop_count": self.loop_count,
            "tool_error_recovery_rate": self.tool_error_recovery_rate,
            "verification_attempts": self.verification_attempts,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_latency_ms": self.model_latency_ms,
            "tool_latency_ms": self.tool_latency_ms,
            "verifier_latency_ms": self.verifier_latency_ms,
            "estimated_cost": self.estimated_cost,
            "not_observable": list(self.not_observable),
            "input_episode_checksum": self.input_episode_checksum,
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


def _tool_action_key(event: TraceEvent) -> bytes:
    return canonical_json_bytes(
        {
            "tool_name": event.attributes.get("tool_name"),
            "arguments": event.attributes.get("arguments"),
        }
    )


def _latency_sum(events: tuple[TraceEvent, ...], components: set[EventComponent]) -> float | None:
    values = [
        value
        for event in events
        if event.component in components
        for value in [_number(event.attributes.get("latency_ms"))]
        if value is not None
    ]
    return sum(values) if values else None


def _usage(events: tuple[TraceEvent, ...]) -> tuple[int | None, int | None]:
    input_tokens = 0
    output_tokens = 0
    observed = False
    for event in events:
        if event.event_type is not EventType.MODEL_RESPONSE:
            continue
        usage = event.attributes.get("usage")
        if not isinstance(usage, Mapping):
            continue
        input_value = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_value = usage.get("output_tokens", usage.get("completion_tokens"))
        if isinstance(input_value, int) and not isinstance(input_value, bool):
            input_tokens += input_value
            observed = True
        if isinstance(output_value, int) and not isinstance(output_value, bool):
            output_tokens += output_value
            observed = True
    return (input_tokens, output_tokens) if observed else (None, None)


def compute_episode_metrics(episode: AgentEpisode) -> EpisodeMetrics:
    events = episode.events
    not_observable: list[str] = ["estimated_cost: pricing configuration absent"]
    duration_ms = (_timestamp(episode.ended_at) - _timestamp(episode.started_at)).total_seconds() * 1000
    turn_count = sum(event.event_type is EventType.MODEL_RESPONSE for event in events)
    verification_attempts = sum(
        event.event_type is EventType.VERIFICATION_STARTED for event in events
    )

    if CaptureCapability.TOOL_IO in episode.capabilities:
        calls = [event for event in events if event.event_type is EventType.TOOL_CALL]
        results = [event for event in events if event.event_type is EventType.TOOL_RESULT]
        failed_result_indices = [
            index
            for index, event in enumerate(events)
            if event.event_type is EventType.TOOL_RESULT
            and event.status in {EventStatus.FAILED, EventStatus.ERROR, EventStatus.TIMEOUT}
        ]
        keys = [_tool_action_key(event) for event in calls]
        duplicate_action_count = sum(
            current == previous for previous, current in zip(keys, keys[1:])
        )
        loop_count = 0
        in_loop = False
        for previous, current in zip(keys, keys[1:]):
            if current == previous and not in_loop:
                loop_count += 1
                in_loop = True
            elif current != previous:
                in_loop = False
        recovered = 0
        for failed_index in failed_result_indices:
            if any(
                later.event_type is EventType.TOOL_RESULT
                and later.status is EventStatus.SUCCEEDED
                for later in events[failed_index + 1 :]
            ):
                recovered += 1
        tool_call_count: int | None = len(calls)
        tool_error_count: int | None = sum(
            event.status in {EventStatus.FAILED, EventStatus.ERROR, EventStatus.TIMEOUT}
            for event in results
        )
        duplicate_rate: float | None = (
            duplicate_action_count / len(calls) if calls else 0.0
        )
        recovery_rate: float | None = (
            recovered / len(failed_result_indices) if failed_result_indices else 0.0
        )
    else:
        tool_call_count = None
        tool_error_count = None
        duplicate_action_count = None
        duplicate_rate = None
        loop_count = None
        recovery_rate = None
        not_observable.extend(
            (
                "tool_call_count: TOOL_IO capability missing",
                "tool_error_count: TOOL_IO capability missing",
                "duplicate_action_rate: TOOL_IO capability missing",
                "loop_count: TOOL_IO capability missing",
                "tool_error_recovery_rate: TOOL_IO capability missing",
            )
        )

    if CaptureCapability.MODEL_TOKEN_USAGE in episode.capabilities:
        input_tokens, output_tokens = _usage(events)
        if input_tokens is None:
            not_observable.append("token_usage: declared capability has no observed usage event")
    else:
        input_tokens, output_tokens = None, None
        not_observable.append("token_usage: MODEL_TOKEN_USAGE capability missing")

    return EpisodeMetrics(
        episode_id=episode.episode_id,
        task_status=episode.outcome.task_status.value,
        execution_validity=episode.outcome.execution_validity.value,
        integrity_state=episode.integrity.state.value,
        duration_ms=duration_ms,
        turn_count=turn_count,
        tool_call_count=tool_call_count,
        tool_error_count=tool_error_count,
        duplicate_action_count=duplicate_action_count,
        duplicate_action_rate=duplicate_rate,
        loop_count=loop_count,
        tool_error_recovery_rate=recovery_rate,
        verification_attempts=verification_attempts,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_latency_ms=_latency_sum(events, {EventComponent.MODEL_BACKEND}),
        tool_latency_ms=_latency_sum(events, {EventComponent.TOOL}),
        verifier_latency_ms=_latency_sum(events, {EventComponent.EVALUATOR}),
        estimated_cost=None,
        not_observable=tuple(not_observable),
        input_episode_checksum=episode.checksum,
    )
