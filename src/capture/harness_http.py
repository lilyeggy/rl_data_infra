"""Authenticated, execution-bound ingress for Harness-native trace facts."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.capture.recorder import TraceRecorder
from src.contracts._validation import strict_fields
from src.contracts.execution_identity import ExecutionIdentity
from src.contracts.trace_event import EventComponent, EventStatus, EventType, TraceEvent
from src.errors import ContractValidationError

HARNESS_EVENT_INGRESS_VERSION = "harness-event-ingress/v1"

_HARNESS_EVENT_COMPONENTS = {
    EventType.TOOL_CALL: EventComponent.TOOL,
    EventType.TOOL_RESULT: EventComponent.TOOL,
    EventType.SANDBOX_COMMAND: EventComponent.SANDBOX,
    EventType.HARNESS_DECISION: EventComponent.HARNESS,
    EventType.CONTEXT_SELECTED: EventComponent.HARNESS,
    EventType.CONTEXT_COMPACTED: EventComponent.HARNESS,
    EventType.RETRY_SCHEDULED: EventComponent.HARNESS,
    EventType.TERMINATION_DECIDED: EventComponent.HARNESS,
}


class HarnessEventIngress:
    """Turn an untrusted event descriptor into a server-owned TraceEvent.

    The Harness may describe facts that only it can observe. Identity,
    sequencing, timestamps and event IDs remain owned by the host recorder.
    Sandbox lifecycle, verifier verdicts and episode completion are reserved for
    the orchestrator, while model I/O is reserved for the model proxy.
    """

    def __init__(
        self,
        *,
        identity: ExecutionIdentity,
        recorder: TraceRecorder,
    ) -> None:
        if not isinstance(identity, ExecutionIdentity):
            raise TypeError("identity must be ExecutionIdentity")
        if not isinstance(recorder, TraceRecorder):
            raise TypeError("recorder must be TraceRecorder")
        self.identity = identity
        self.recorder = recorder

    def emit(self, value: Mapping[str, Any]) -> TraceEvent:
        payload = strict_fields(
            value,
            {
                "event_type",
                "component",
                "status",
                "span_id",
                "parent_span_id",
                "attributes",
                "artifact_refs",
                "attempt",
            },
            "HarnessEventDescriptor",
        )
        try:
            event_type = EventType(payload["event_type"])
            component = EventComponent(payload["component"])
            status = EventStatus(payload["status"])
        except (TypeError, ValueError) as exc:
            raise ContractValidationError("invalid Harness event enum") from exc
        expected_component = _HARNESS_EVENT_COMPONENTS.get(event_type)
        if expected_component is None:
            raise ContractValidationError(
                f"Harness cannot emit orchestrator-owned event type {event_type.value}"
            )
        if component is not expected_component:
            raise ContractValidationError(
                f"{event_type.value} must use component {expected_component.value}"
            )
        if payload["attempt"] != self.identity.attempt_id:
            raise ContractValidationError("Harness event attempt does not match execution")
        if not isinstance(payload["attributes"], Mapping):
            raise ContractValidationError("Harness event attributes must be an object")
        if not isinstance(payload["artifact_refs"], list):
            raise ContractValidationError("Harness event artifact_refs must be an array")
        return self.recorder.emit(
            event_type,
            component,
            status,
            span_id=payload["span_id"],
            parent_span_id=payload["parent_span_id"],
            attributes=payload["attributes"],
            artifact_refs=tuple(payload["artifact_refs"]),
            attempt=payload["attempt"],
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class HarnessTraceClient:
    """Tiny dependency-free SDK usable by a Python Harness adapter."""

    endpoint: str
    access_token: str
    attempt: int = 1
    timeout_seconds: float = 10

    def emit(
        self,
        *,
        event_type: EventType,
        component: EventComponent,
        status: EventStatus,
        span_id: str,
        parent_span_id: str | None = None,
        attributes: Mapping[str, Any] | None = None,
        artifact_refs: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        body = json.dumps(
            {
                "event_type": event_type.value,
                "component": component.value,
                "status": status.value,
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "attributes": dict(attributes or {}),
                "artifact_refs": list(artifact_refs),
                "attempt": self.attempt,
            },
            ensure_ascii=False,
        ).encode()
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            message = exc.read().decode(errors="replace")
            raise ContractValidationError(
                f"Harness event ingress rejected the event: HTTP {exc.code}: {message}"
            ) from exc
        if not isinstance(result, dict):
            raise ContractValidationError("Harness event ingress returned invalid JSON")
        return result
