"""Convert Pi ``--mode json`` NDJSON into canonical observable facts.

The adapter deliberately treats Pi as a black-box Harness.  It records only
fields present in the protocol and never infers hidden context/retry reasons
from assistant prose.  A zero process exit code is not a success signal: the
terminal semantic state comes from Pi message events and an external verifier.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

from src.capture.recorder import redact_secrets
from src.contracts._json import freeze_json, sha256_json, thaw_json
from src.contracts.agent_episode import (
    CaptureCapability,
    EpisodeVerifierStatus,
    ExecutionValidity,
    TaskStatus,
)
from src.contracts.trace_event import EventComponent, EventStatus, EventType, TraceEvent
from src.errors import AdapterIssue, ErrorCode

PI_ADAPTER_VERSION = "pi-json-adapter/v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class PiRunConfig:
    provider: str = "opencode-go"
    model: str = "gpt-5.6-luna"
    pi_version: str = "0.84.2"
    thinking: str = "minimal"
    tools: tuple[str, ...] = ("read", "grep", "find", "ls")
    api: str = "openai-responses"

    def command(self, prompt: str) -> tuple[str, ...]:
        """Return a non-shell command with no silent model fallback."""

        return (
            "pi",
            "--provider",
            self.provider,
            "--model",
            self.model,
            "--mode",
            "json",
            "--print",
            "--no-session",
            "--no-context-files",
            "--no-extensions",
            "--no-skills",
            "--tools",
            ",".join(self.tools),
            "--thinking",
            self.thinking,
            prompt,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "pi_version": self.pi_version,
            "thinking": self.thinking,
            "tools": list(self.tools),
            "api": self.api,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class PiOutcomeDeclaration:
    """Outcome supplied by a verifier, never guessed from model prose."""

    task_status: TaskStatus
    execution_validity: ExecutionValidity
    verifier_status: EpisodeVerifierStatus
    score: float | None = None
    termination_reason: str = "PI_AGENT_SETTLED"


@dataclass(frozen=True, slots=True)
class PiAdapterResult:
    session_id: str
    events: tuple[TraceEvent, ...]
    capabilities: frozenset[CaptureCapability]
    issues: tuple[AdapterIssue, ...]
    backend_error_messages: tuple[str, ...]
    source_checksum: str
    adapter_version: str = PI_ADAPTER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_version": self.adapter_version,
            "session_id": self.session_id,
            "events": [event.to_dict() for event in self.events],
            "capabilities": sorted(item.value for item in self.capabilities),
            "issues": [
                {
                    "code": issue.code.value,
                    "message": issue.message,
                    "source_record_id": issue.source_record_id,
                    "field": issue.field,
                    "details": thaw_json(issue.details),
                }
                for issue in self.issues
            ],
            "backend_error_messages": list(self.backend_error_messages),
            "source_checksum": self.source_checksum,
        }


def read_pi_ndjson(text: str) -> tuple[tuple[Mapping[str, Any], ...], tuple[AdapterIssue, ...]]:
    records: list[Mapping[str, Any]] = []
    issues: list[AdapterIssue] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(
                AdapterIssue(
                    code=ErrorCode.SOURCE_WARNING,
                    message=f"invalid Pi JSON at line {line_number}: {exc.msg}",
                    field="raw_ndjson",
                    details={"line_number": line_number},
                )
            )
            continue
        if not isinstance(value, Mapping):
            issues.append(
                AdapterIssue(
                    code=ErrorCode.SOURCE_WARNING,
                    message=f"Pi record at line {line_number} is not an object",
                    field="raw_ndjson",
                    details={"line_number": line_number},
                )
            )
            continue
        records.append(freeze_json(redact_secrets(value), f"$.line[{line_number}]"))
    return tuple(records), tuple(issues)


def dump_pi_ndjson(records: Iterable[Mapping[str, Any]]) -> str:
    """Serialize already validated/redacted Pi records for a shareable fixture."""

    return "".join(
        json.dumps(thaw_json(record), ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )


class PiJsonAdapter:
    def convert(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        run_id: str,
        episode_id: str,
        trace_id: str,
        config: PiRunConfig,
        declared_outcome: PiOutcomeDeclaration,
        source_issues: Iterable[AdapterIssue] = (),
        normalize_tool_errors: bool = False,
    ) -> PiAdapterResult:
        raw_records = tuple(records)
        session = next((item for item in raw_records if item.get("type") == "session"), None)
        session_id = (
            str(session.get("id"))
            if session and session.get("id")
            else "pi-session-unknown"
        )
        base = self._base_time(session)
        tool_result_times = self._tool_result_timestamps(raw_records)
        events: list[TraceEvent] = []
        issues = list(source_issues)
        pending_tools: dict[str, tuple[str, str, str]] = {}
        backend_errors: list[str] = []
        assistant_index = 0
        final_response_seen = False
        retry_observed = False
        tool_io_observed = False
        usage_observed = False
        termination_observed = False
        observed_messages: list[Mapping[str, Any]] = []
        last_event_time = base - timedelta(milliseconds=1)

        def emit(
            event_type: EventType,
            component: EventComponent,
            status: EventStatus,
            *,
            span_id: str,
            parent_span_id: str | None,
            attributes: Mapping[str, Any],
            source_index: int,
            attempt: int = 1,
            source_timestamp_ms: int | float | None = None,
        ) -> TraceEvent:
            nonlocal last_event_time
            sequence = len(events)
            identity = sha256_json(
                {
                    "session_id": session_id,
                    "source_index": source_index,
                    "sequence": sequence,
                    "event_type": event_type.value,
                }
            )[:24]
            observed_time = self._timestamp_from_milliseconds(source_timestamp_ms)
            candidate_time = observed_time or (last_event_time + timedelta(milliseconds=1))
            if candidate_time <= last_event_time:
                candidate_time = last_event_time + timedelta(milliseconds=1)
            last_event_time = candidate_time
            event = TraceEvent(
                event_id=f"evt-pi-{identity}",
                run_id=run_id,
                episode_id=episode_id,
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                sequence=sequence,
                timestamp=candidate_time.isoformat().replace("+00:00", "Z"),
                event_type=event_type,
                component=component,
                status=status,
                attempt=attempt,
                attributes=redact_secrets(attributes),
            )
            events.append(event)
            return event

        for source_index, record in enumerate(raw_records):
            record_type = record.get("type")
            if record_type == "message_end":
                message = record.get("message")
                if not isinstance(message, Mapping):
                    continue
                if message.get("role") != "assistant":
                    if message.get("role") in {"user", "toolResult"}:
                        observed_messages.append(
                            {
                                "role": message.get("role"),
                                "content": message.get("content", ()),
                                "tool_name": message.get("toolName"),
                                "tool_call_id": message.get("toolCallId"),
                                "is_error": message.get("isError"),
                            }
                        )
                    continue
                assistant_index += 1
                message_timestamp = message.get("timestamp")
                model_span = f"span-pi-model-{assistant_index}"
                emit(
                    EventType.MODEL_REQUEST,
                    EventComponent.MODEL,
                    EventStatus.STARTED,
                    span_id=model_span,
                    parent_span_id=None,
                    attributes={
                        "provider": config.provider,
                        "model": config.model,
                        "api": message.get("api", config.api),
                        "source": "pi-message-end",
                        "messages": observed_messages,
                        "tools": list(config.tools),
                        "system_prompt": "NOT_OBSERVABLE",
                        "request_payload": "PARTIAL_RECONSTRUCTION_FROM_MESSAGE_ENDS",
                    },
                    source_index=source_index,
                    attempt=assistant_index,
                    source_timestamp_ms=(
                        message_timestamp - 1
                        if isinstance(message_timestamp, (int, float))
                        else None
                    ),
                )
                stop_reason = str(message.get("stopReason", "unknown"))
                error_message = message.get("errorMessage")
                response_status = (
                    EventStatus.ERROR if stop_reason == "error" else EventStatus.SUCCEEDED
                )
                usage = self._normalize_usage(message.get("usage"))
                usage_observed = usage_observed or usage is not None
                response = emit(
                    EventType.MODEL_RESPONSE,
                    EventComponent.MODEL_BACKEND,
                    response_status,
                    span_id=model_span,
                    parent_span_id=None,
                    attributes={
                        "provider": message.get("provider", config.provider),
                        "model": message.get("model", config.model),
                        "api": message.get("api", config.api),
                        "stop_reason": stop_reason,
                        "response_id": message.get("responseId"),
                        "raw_stop_reason": message.get("rawStopReason"),
                        "content": message.get("content", ()),
                        "usage": usage,
                        "error_message": error_message,
                    },
                    source_index=source_index,
                    attempt=assistant_index,
                    source_timestamp_ms=message_timestamp,
                )
                if stop_reason == "stop":
                    final_response_seen = True
                if stop_reason == "error":
                    backend_errors.append(str(error_message or "Pi model backend error"))
                content = message.get("content", ())
                if isinstance(content, (list, tuple)):
                    for content_index, item in enumerate(content):
                        if not isinstance(item, Mapping) or item.get("type") != "toolCall":
                            continue
                        tool_io_observed = True
                        tool_call_id = str(item.get("id") or f"tool-{source_index}-{content_index}")
                        tool_span = f"span-pi-tool-{sha256_json(tool_call_id)[:18]}"
                        tool_name = str(item.get("name") or "unknown")
                        arguments = item.get("arguments")
                        if not isinstance(arguments, Mapping):
                            arguments = {"raw": arguments}
                        emit(
                            EventType.TOOL_CALL,
                            EventComponent.TOOL,
                            EventStatus.STARTED,
                            span_id=tool_span,
                            parent_span_id=response.span_id,
                            attributes={
                                "tool_name": tool_name,
                                "arguments": arguments,
                                "pi_tool_call_id": tool_call_id,
                            },
                            source_index=source_index,
                            attempt=assistant_index,
                            source_timestamp_ms=message_timestamp,
                        )
                        pending_tools[tool_call_id] = (tool_span, response.span_id, tool_name)
                observed_messages.append(
                    {
                        "role": "assistant",
                        "content": message.get("content", ()),
                        "stop_reason": stop_reason,
                    }
                )
            elif record_type == "tool_execution_end":
                tool_io_observed = True
                tool_call_id = str(record.get("toolCallId") or "unknown")
                pending = pending_tools.pop(tool_call_id, None)
                if pending is None:
                    issues.append(
                        AdapterIssue(
                            code=ErrorCode.SOURCE_WARNING,
                            message="Pi tool result had no observed tool call",
                            source_record_id=tool_call_id,
                            field="toolCallId",
                        )
                    )
                    tool_span = f"span-pi-tool-{sha256_json(tool_call_id)[:18]}"
                    parent_span = None
                    tool_name = str(record.get("toolName") or "unknown")
                else:
                    tool_span, parent_span, tool_name = pending
                result = record.get("result")
                is_error = bool(record.get("isError"))
                if isinstance(result, Mapping):
                    is_error = is_error or bool(result.get("isError"))
                tool_attributes: dict[str, Any] = {
                    "tool_name": record.get("toolName", tool_name),
                    "pi_tool_call_id": tool_call_id,
                    "result": result,
                }
                if normalize_tool_errors:
                    tool_error = self._normalize_tool_error(result, is_error=is_error)
                    if tool_error is not None:
                        tool_attributes.update(tool_error)
                emit(
                    EventType.TOOL_RESULT,
                    EventComponent.TOOL,
                    EventStatus.ERROR if is_error else EventStatus.SUCCEEDED,
                    span_id=tool_span,
                    parent_span_id=parent_span,
                    attributes=tool_attributes,
                    source_index=source_index,
                    source_timestamp_ms=tool_result_times.get(tool_call_id),
                )
            elif record_type == "agent_end" and isinstance(record.get("willRetry"), bool):
                termination_observed = True
                will_retry = bool(record["willRetry"])
                retry_observed = retry_observed or will_retry
                emit(
                    EventType.RETRY_SCHEDULED if will_retry else EventType.TERMINATION_DECIDED,
                    EventComponent.HARNESS,
                    EventStatus.SUCCEEDED,
                    span_id=f"span-pi-harness-{source_index}",
                    parent_span_id=None,
                    attributes={"will_retry": will_retry, "reason": "PI_PROTOCOL_DECLARATION"},
                    source_index=source_index,
                )

        semantic_backend_failure = bool(backend_errors) and not final_response_seen
        if not semantic_backend_failure:
            verifier_span = "span-pi-verifier"
            started = emit(
                EventType.VERIFICATION_STARTED,
                EventComponent.EVALUATOR,
                EventStatus.STARTED,
                span_id=verifier_span,
                parent_span_id=None,
                attributes={"verifier": "external-declared-verifier"},
                source_index=len(raw_records),
            )
            verifier_status = {
                EpisodeVerifierStatus.PASSED: EventStatus.SUCCEEDED,
                EpisodeVerifierStatus.FAILED: EventStatus.FAILED,
                EpisodeVerifierStatus.ERROR: EventStatus.ERROR,
                EpisodeVerifierStatus.TIMEOUT: EventStatus.TIMEOUT,
            }.get(declared_outcome.verifier_status, EventStatus.UNKNOWN)
            verified = emit(
                EventType.VERIFICATION_FINISHED,
                EventComponent.EVALUATOR,
                verifier_status,
                span_id=verifier_span,
                parent_span_id=None,
                attributes={
                    "verifier": "external-declared-verifier",
                    "passed": declared_outcome.verifier_status is EpisodeVerifierStatus.PASSED,
                    "score": declared_outcome.score,
                    "started_event_id": started.event_id,
                },
                source_index=len(raw_records) + 1,
            )
            task_status = declared_outcome.task_status
            validity = declared_outcome.execution_validity
            terminal_verifier = declared_outcome.verifier_status
            evidence = [verified.event_id]
            terminal_status = (
                EventStatus.SUCCEEDED
                if task_status is TaskStatus.SUCCESS
                else EventStatus.FAILED
            )
            termination_reason = declared_outcome.termination_reason
        else:
            task_status = TaskStatus.UNKNOWN
            validity = ExecutionValidity.INFRA_INVALID
            terminal_verifier = EpisodeVerifierStatus.NOT_RUN
            evidence = [
                event.event_id
                for event in events
                if event.event_type is EventType.MODEL_RESPONSE
                and event.status is EventStatus.ERROR
            ]
            terminal_status = EventStatus.ERROR
            termination_reason = "MODEL_BACKEND_ERROR"

        emit(
            EventType.EPISODE_FINISHED,
            EventComponent.HARNESS,
            terminal_status,
            span_id="span-pi-episode",
            parent_span_id=None,
            attributes={
                "task_status": task_status.value,
                "execution_validity": validity.value,
                "verifier_status": terminal_verifier.value,
                "score": None if semantic_backend_failure else declared_outcome.score,
                "termination_reason": termination_reason,
                "evidence_event_ids": evidence,
                "pi_process_exit_code_is_not_semantic_success": True,
            },
            source_index=len(raw_records) + 2,
        )

        capabilities = {CaptureCapability.MODEL_IO}
        if termination_observed:
            capabilities.add(CaptureCapability.TERMINATION_DECISIONS)
        if usage_observed:
            capabilities.add(CaptureCapability.MODEL_TOKEN_USAGE)
        if tool_io_observed:
            capabilities.add(CaptureCapability.TOOL_IO)
        if retry_observed:
            capabilities.add(CaptureCapability.RETRY_DECISIONS)
        return PiAdapterResult(
            session_id=session_id,
            events=tuple(events),
            capabilities=frozenset(capabilities),
            issues=tuple(issues),
            backend_error_messages=tuple(backend_errors),
            source_checksum=sha256_json([thaw_json(item) for item in raw_records]),
        )

    @staticmethod
    def _base_time(session: Mapping[str, Any] | None) -> datetime:
        value = session.get("timestamp") if session else None
        if isinstance(value, str):
            candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
            try:
                parsed = datetime.fromisoformat(candidate)
                if parsed.tzinfo is not None:
                    return parsed.astimezone(timezone.utc)
            except ValueError:
                pass
        return datetime(1970, 1, 1, tzinfo=timezone.utc)

    @staticmethod
    def _timestamp_from_milliseconds(value: int | float | None) -> datetime | None:
        if not isinstance(value, (int, float)):
            return None
        try:
            return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

    @staticmethod
    def _normalize_tool_error(
        result: Any,
        *,
        is_error: bool,
    ) -> Mapping[str, Any] | None:
        """Normalize only structured Pi tool errors into stable error facts.

        The original result remains in the event.  This helper does not inspect
        assistant prose; it only handles a tool protocol result explicitly
        marked as an error and recognizes the provider-independent ENOENT form.
        """

        if not is_error:
            return None
        texts: list[str] = []

        def collect(value: Any) -> None:
            if isinstance(value, Mapping):
                text = value.get("text")
                if isinstance(text, str):
                    texts.append(text)
                for item in value.values():
                    collect(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
            elif isinstance(value, str):
                texts.append(value)

        collect(result)
        message = " ".join(texts)
        if "ENOENT" not in message and "no such file or directory" not in message.lower():
            return None
        path_match = re.search(r"(?:access|path) ['\\\"]([^'\\\"]+)['\\\"]", message)
        normalized: dict[str, Any] = {
            "error_code": "FILE_NOT_FOUND",
            "error_source": "pi_tool_protocol",
        }
        if path_match is not None:
            normalized["error_path"] = path_match.group(1)
        return normalized

    @staticmethod
    def _tool_result_timestamps(
        records: Iterable[Mapping[str, Any]],
    ) -> dict[str, int | float]:
        """Index timestamps Pi places on duplicated tool-result messages.

        ``tool_execution_end`` itself currently has no clock value. Pi repeats
        the same result in message/turn/agent records, so the timestamp can be
        joined by the protocol tool-call id without interpreting prose.
        """

        result: dict[str, int | float] = {}

        def inspect(value: Any) -> None:
            if isinstance(value, Mapping):
                if value.get("role") == "toolResult":
                    call_id = value.get("toolCallId")
                    timestamp = value.get("timestamp")
                    if call_id and isinstance(timestamp, (int, float)):
                        result.setdefault(str(call_id), timestamp)
                for item in value.values():
                    inspect(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    inspect(item)

        for record in records:
            inspect(record)
        return result

    @staticmethod
    def _normalize_usage(value: Any) -> Mapping[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        cost = value.get("cost") if isinstance(value.get("cost"), Mapping) else {}
        return {
            "input_tokens": value.get("input", value.get("input_tokens")),
            "output_tokens": value.get("output", value.get("output_tokens")),
            "cache_read_tokens": value.get("cacheRead", 0),
            "cache_write_tokens": value.get("cacheWrite", 0),
            "reasoning_tokens": value.get("reasoning"),
            "total_tokens": value.get("totalTokens"),
            "cost_total": cost.get("total"),
        }
