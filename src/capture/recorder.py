"""Harness-neutral event emission API used by proxy, tool and hook adapters."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from src.capture.event_writer import EventWriter
from src.contracts.trace_event import (
    EventComponent,
    EventStatus,
    EventType,
    TraceEvent,
)


SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "api_key",
        "apikey",
        "cookie",
        "password",
        "proxy_authorization",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "thinking_signature",
        "text_signature",
        "thinkingsignature",
        "textsignature",
        # Public trace fixtures must not expose chain-of-thought, even when a
        # provider happens to return a plaintext summary beside its signature.
        "thinking",
    }
)


def redact_secrets(value: Any) -> Any:
    """Redact values under known credential keys before they reach raw storage."""

    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]"
            if str(key).lower().replace("-", "_") in SENSITIVE_KEYS
            else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class TraceRecorder:
    """Assign identity/order and persist observable facts from any Harness.

    Convenience methods intentionally keep provider-specific payloads inside
    ``attributes``. A Harness can use only :meth:`emit` and still remain fully
    compatible with the assembler.
    """

    def __init__(
        self,
        writer: EventWriter,
        *,
        run_id: str,
        episode_id: str,
        trace_id: str,
        initial_sequence: int = 0,
        clock: Callable[[], str] = _utc_now,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.writer = writer
        self.run_id = run_id
        self.episode_id = episode_id
        self.trace_id = trace_id
        self._next_sequence = initial_sequence
        self._clock = clock
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)

    def emit(
        self,
        event_type: EventType,
        component: EventComponent,
        status: EventStatus,
        *,
        span_id: str,
        parent_span_id: str | None,
        attributes: Mapping[str, Any] | None = None,
        artifact_refs: tuple[str, ...] = (),
        attempt: int = 1,
        event_id: str | None = None,
        timestamp: str | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            event_id=event_id or f"evt-{self._id_factory()}",
            run_id=self.run_id,
            episode_id=self.episode_id,
            trace_id=self.trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            sequence=self._next_sequence,
            timestamp=timestamp or self._clock(),
            event_type=event_type,
            component=component,
            status=status,
            attempt=attempt,
            attributes=redact_secrets(attributes or {}),
            artifact_refs=artifact_refs,
        )
        self.writer.append(event)
        # An exact replay is still one logical producer step. Advancing the
        # local cursor makes a deterministic capture safe to resume/replay
        # against an existing append-only log.
        self._next_sequence += 1
        return event

    def model_request(
        self,
        *,
        span_id: str,
        parent_span_id: str | None,
        model: str,
        messages: list[Mapping[str, Any]],
        tools: list[Mapping[str, Any]] | None = None,
        attempt: int = 1,
    ) -> TraceEvent:
        return self.emit(
            EventType.MODEL_REQUEST,
            EventComponent.MODEL,
            EventStatus.STARTED,
            span_id=span_id,
            parent_span_id=parent_span_id,
            attempt=attempt,
            attributes={"model": model, "messages": messages, "tools": tools or []},
        )

    def model_response(
        self,
        *,
        span_id: str,
        parent_span_id: str | None,
        response: Mapping[str, Any],
        usage: Mapping[str, Any] | None = None,
        latency_ms: float | None = None,
        status: EventStatus = EventStatus.SUCCEEDED,
        attempt: int = 1,
    ) -> TraceEvent:
        attributes: dict[str, Any] = {"response": response}
        if usage is not None:
            attributes["usage"] = usage
        if latency_ms is not None:
            attributes["latency_ms"] = latency_ms
        return self.emit(
            EventType.MODEL_RESPONSE,
            EventComponent.MODEL_BACKEND,
            status,
            span_id=span_id,
            parent_span_id=parent_span_id,
            attempt=attempt,
            attributes=attributes,
        )

    def tool_call(
        self,
        *,
        span_id: str,
        parent_span_id: str | None,
        tool_name: str,
        arguments: Mapping[str, Any],
        attempt: int = 1,
    ) -> TraceEvent:
        return self.emit(
            EventType.TOOL_CALL,
            EventComponent.TOOL,
            EventStatus.STARTED,
            span_id=span_id,
            parent_span_id=parent_span_id,
            attempt=attempt,
            attributes={"tool_name": tool_name, "arguments": arguments},
        )

    def tool_result(
        self,
        *,
        span_id: str,
        parent_span_id: str | None,
        tool_name: str,
        result: Mapping[str, Any],
        status: EventStatus,
        latency_ms: float | None = None,
        attempt: int = 1,
        artifact_refs: tuple[str, ...] = (),
    ) -> TraceEvent:
        attributes: dict[str, Any] = {"tool_name": tool_name, "result": result}
        if latency_ms is not None:
            attributes["latency_ms"] = latency_ms
        return self.emit(
            EventType.TOOL_RESULT,
            EventComponent.TOOL,
            status,
            span_id=span_id,
            parent_span_id=parent_span_id,
            attempt=attempt,
            attributes=attributes,
            artifact_refs=artifact_refs,
        )

    def finish(
        self,
        *,
        span_id: str,
        task_status: str,
        execution_validity: str,
        verifier_status: str,
        termination_reason: str,
        score: float | None = None,
        evidence_event_ids: tuple[str, ...] = (),
    ) -> TraceEvent:
        return self.emit(
            EventType.EPISODE_FINISHED,
            EventComponent.HARNESS,
            EventStatus.SUCCEEDED,
            span_id=span_id,
            parent_span_id=None,
            attributes={
                "task_status": task_status,
                "execution_validity": execution_validity,
                "verifier_status": verifier_status,
                "termination_reason": termination_reason,
                "score": score,
                "evidence_event_ids": list(evidence_event_ids),
            },
        )


class HarnessHook:
    """Optional observability surface for decisions black-box capture cannot see."""

    def __init__(self, recorder: TraceRecorder) -> None:
        self.recorder = recorder

    def emit_decision(
        self,
        *,
        span_id: str,
        parent_span_id: str | None,
        decision: str,
        reason_code: str,
        details: Mapping[str, Any] | None = None,
    ) -> TraceEvent:
        return self.recorder.emit(
            EventType.HARNESS_DECISION,
            EventComponent.HARNESS,
            EventStatus.SUCCEEDED,
            span_id=span_id,
            parent_span_id=parent_span_id,
            attributes={
                "decision": decision,
                "reason_code": reason_code,
                "details": details or {},
            },
        )

    def emit_retry_scheduled(
        self,
        *,
        span_id: str,
        parent_span_id: str | None,
        attempt: int,
        reason: str,
        delay_ms: int,
    ) -> TraceEvent:
        return self.recorder.emit(
            EventType.RETRY_SCHEDULED,
            EventComponent.HARNESS,
            EventStatus.SUCCEEDED,
            span_id=span_id,
            parent_span_id=parent_span_id,
            attempt=attempt,
            attributes={"reason": reason, "delay_ms": delay_ms},
        )

    def emit_termination_decided(
        self,
        *,
        span_id: str,
        reason: str,
        verifier_observed: bool,
    ) -> TraceEvent:
        return self.recorder.emit(
            EventType.TERMINATION_DECIDED,
            EventComponent.HARNESS,
            EventStatus.SUCCEEDED,
            span_id=span_id,
            parent_span_id=None,
            attributes={"reason": reason, "verifier_observed": verifier_observed},
        )


class EnvironmentCapture:
    """Adapter-facing methods for facts returned by a Sandbox or verifier."""

    def __init__(self, recorder: TraceRecorder) -> None:
        self.recorder = recorder

    def sandbox_started(
        self,
        *,
        span_id: str,
        runtime_id: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> TraceEvent:
        return self.recorder.emit(
            EventType.SANDBOX_STARTED,
            EventComponent.SANDBOX,
            EventStatus.STARTED,
            span_id=span_id,
            parent_span_id=None,
            attributes={"runtime_id": runtime_id, **dict(attributes or {})},
        )

    def command_result(
        self,
        *,
        span_id: str,
        command: str,
        cwd: str,
        exit_code: int | None,
        status: EventStatus,
        latency_ms: float,
        artifact_refs: tuple[str, ...] = (),
    ) -> TraceEvent:
        return self.recorder.emit(
            EventType.SANDBOX_COMMAND,
            EventComponent.SANDBOX,
            status,
            span_id=span_id,
            parent_span_id=None,
            attributes={
                "command": command,
                "cwd": cwd,
                "exit_code": exit_code,
                "latency_ms": latency_ms,
            },
            artifact_refs=artifact_refs,
        )

    def sandbox_finished(
        self,
        *,
        span_id: str,
        runtime_id: str,
        status: EventStatus,
    ) -> TraceEvent:
        return self.recorder.emit(
            EventType.SANDBOX_FINISHED,
            EventComponent.SANDBOX,
            status,
            span_id=span_id,
            parent_span_id=None,
            attributes={"runtime_id": runtime_id},
        )

    def verification_started(self, *, span_id: str, verifier: str) -> TraceEvent:
        return self.recorder.emit(
            EventType.VERIFICATION_STARTED,
            EventComponent.EVALUATOR,
            EventStatus.STARTED,
            span_id=span_id,
            parent_span_id=None,
            attributes={"verifier": verifier},
        )

    def verification_finished(
        self,
        *,
        span_id: str,
        verifier: str,
        passed: bool | None,
        status: EventStatus,
        score: float | None,
        artifact_refs: tuple[str, ...] = (),
    ) -> TraceEvent:
        return self.recorder.emit(
            EventType.VERIFICATION_FINISHED,
            EventComponent.EVALUATOR,
            status,
            span_id=span_id,
            parent_span_id=None,
            attributes={"verifier": verifier, "passed": passed, "score": score},
            artifact_refs=artifact_refs,
        )
