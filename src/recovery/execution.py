"""Bounded recovery execution adapter and safe local tool executor."""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from src.capture.recorder import TraceRecorder
from src.contracts._json import sha256_json
from src.contracts.trace_event import EventStatus, TraceEvent
from src.errors import ContractValidationError
from src.recovery.controller import RecoveryController
from src.recovery.dispatch import RecoveryToolDispatcher, ToolDispatchRequest
from src.recovery.file_not_found_policy import RecoveryAction, RecoveryDecision


@dataclass(frozen=True, slots=True)
class RecoveryExecutionResult:
    """Actual tool result returned by a bounded recovery executor."""

    event: TraceEvent
    event_ids: tuple[str, ...]
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecoveryExecutionReport:
    """Derived report for one bounded recovery attempt."""

    trigger_event_id: str
    decisions: tuple[RecoveryDecision, ...]
    executed_event_ids: tuple[str, ...]
    terminal_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_event_id": self.trigger_event_id,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "executed_event_ids": list(self.executed_event_ids),
            "terminal_reason": self.terminal_reason,
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


class RecoveryToolExecutor(Protocol):
    """Runtime-provided executor for actions emitted by the policy."""

    def execute(
        self,
        decision: RecoveryDecision,
        *,
        parent_span_id: str | None,
    ) -> RecoveryExecutionResult:
        """Execute exactly one bounded DISCOVER or READ action."""
        ...


class RecoveryExecutionAdapter:
    """Run at most one discovery and one read for one file-not-found error."""

    def __init__(
        self,
        *,
        controller: RecoveryController,
        executor: RecoveryToolExecutor,
    ) -> None:
        self.controller = controller
        self.executor = executor

    def recover(
        self,
        error_event: TraceEvent,
        *,
        scope_key: str | None = None,
    ) -> RecoveryExecutionReport:
        decisions: list[RecoveryDecision] = []
        executed_event_ids: list[str] = []
        decision = self.controller.handle_tool_result(
            error_event,
            scope_key=scope_key,
        )
        decisions.append(decision)
        terminal_reason: str | None = None

        if decision.action is RecoveryAction.DISCOVER:
            discovery = self.executor.execute(
                decision,
                parent_span_id=error_event.span_id,
            )
            executed_event_ids.extend(discovery.event_ids)
            self.controller.record_discovery_result(
                decision.scope_key or scope_key or ".",
                discovery.candidates,
            )
            decision = self.controller.decide_after_discovery(
                error_event,
                discovery_event_id=discovery.event.event_id,
                scope_key=scope_key,
            )
            decisions.append(decision)

        if decision.action is RecoveryAction.READ:
            read_result = self.executor.execute(
                decision,
                parent_span_id=error_event.span_id,
            )
            executed_event_ids.extend(read_result.event_ids)
        elif decision.action is RecoveryAction.TERMINATE:
            terminal_reason = decision.reason_code

        return RecoveryExecutionReport(
            trigger_event_id=error_event.event_id,
            decisions=tuple(decisions),
            executed_event_ids=tuple(executed_event_ids),
            terminal_reason=terminal_reason,
        )


class DispatchingRecoveryToolExecutor:
    """Adapter from the controller executor API to a Harness dispatcher."""

    def __init__(self, dispatcher: RecoveryToolDispatcher) -> None:
        self.dispatcher = dispatcher

    def execute(
        self,
        decision: RecoveryDecision,
        *,
        parent_span_id: str | None,
    ) -> RecoveryExecutionResult:
        request = ToolDispatchRequest.from_decision(
            decision,
            parent_span_id=parent_span_id,
        )
        result = self.dispatcher.dispatch(request)
        if not isinstance(result, RecoveryExecutionResult):
            raise TypeError("RecoveryToolDispatcher must return RecoveryExecutionResult")
        return result


class LocalRecoveryToolExecutor:
    """Safe deterministic find/read executor for local V2.1 integration tests."""

    def __init__(self, *, workspace: str | Path, recorder: "TraceRecorder") -> None:
        self.workspace = Path(workspace).resolve()
        if not self.workspace.is_dir():
            raise ContractValidationError("workspace must be an existing directory")
        self.recorder = recorder
        self._counter = 0

    def execute(
        self,
        decision: RecoveryDecision,
        *,
        parent_span_id: str | None,
    ) -> RecoveryExecutionResult:
        if decision.action is RecoveryAction.DISCOVER:
            return self._discover(decision, parent_span_id=parent_span_id)
        if decision.action is RecoveryAction.READ:
            return self._read(decision, parent_span_id=parent_span_id)
        raise ContractValidationError(
            f"LocalRecoveryToolExecutor cannot execute {decision.action.value}"
        )

    def _discover(
        self,
        decision: RecoveryDecision,
        *,
        parent_span_id: str | None,
    ) -> RecoveryExecutionResult:
        path = decision.arguments.get("path")
        pattern = decision.arguments.get("pattern")
        limit = decision.arguments.get("limit")
        if not isinstance(path, str) or not isinstance(pattern, str):
            raise ContractValidationError("DISCOVER requires path and pattern strings")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ContractValidationError("DISCOVER limit must be a positive integer")
        scope = self._resolve_inside_workspace(path)
        candidates = self._find(scope, pattern, limit)
        call, result = self._emit_tool_pair(
            parent_span_id=parent_span_id,
            tool_name="find",
            arguments=decision.arguments,
            result={"content": [{"type": "text", "text": "\n".join(candidates)}]},
            status=EventStatus.SUCCEEDED,
        )
        return RecoveryExecutionResult(
            event=result,
            event_ids=(call.event_id, result.event_id),
            candidates=candidates,
        )

    def _read(
        self,
        decision: RecoveryDecision,
        *,
        parent_span_id: str | None,
    ) -> RecoveryExecutionResult:
        path = decision.arguments.get("path")
        if not isinstance(path, str):
            raise ContractValidationError("READ requires a path string")
        try:
            target = self._resolve_inside_workspace(path)
            content = target.read_text()
            result: Mapping[str, Any] = {"content": [{"type": "text", "text": content}]}
            status = EventStatus.SUCCEEDED
        except (OSError, ContractValidationError) as exc:
            result = {
                "isError": True,
                "content": [{"type": "text", "text": str(exc)}],
            }
            status = EventStatus.ERROR
        call, tool_result = self._emit_tool_pair(
            parent_span_id=parent_span_id,
            tool_name="read",
            arguments=decision.arguments,
            result=result,
            status=status,
        )
        return RecoveryExecutionResult(
            event=tool_result,
            event_ids=(call.event_id, tool_result.event_id),
        )

    def _emit_tool_pair(
        self,
        *,
        parent_span_id: str | None,
        tool_name: str,
        arguments: Mapping[str, Any],
        result: Mapping[str, Any],
        status: EventStatus,
    ) -> tuple[TraceEvent, TraceEvent]:
        self._counter += 1
        span_id = f"span-recovery-tool-{self._counter}"
        call = self.recorder.tool_call(
            span_id=span_id,
            parent_span_id=parent_span_id,
            tool_name=tool_name,
            arguments=arguments,
        )
        tool_result = self.recorder.tool_result(
            span_id=span_id,
            parent_span_id=parent_span_id,
            tool_name=tool_name,
            result=result,
            status=status,
        )
        return call, tool_result

    def _find(self, scope: Path, pattern: str, limit: int) -> tuple[str, ...]:
        candidates: list[str] = []
        for path in sorted(scope.rglob("*"), key=lambda item: item.as_posix()):
            if not path.is_file():
                continue
            if not fnmatch.fnmatch(path.name, pattern):
                continue
            relative = path.relative_to(scope).as_posix()
            candidates.append(relative)
            if len(candidates) >= limit:
                break
        return tuple(candidates)

    def _resolve_inside_workspace(self, path: str) -> Path:
        candidate = Path(path)
        resolved = (candidate if candidate.is_absolute() else self.workspace / candidate).resolve()
        try:
            resolved.relative_to(self.workspace)
        except ValueError as exc:
            raise ContractValidationError("recovery path escapes workspace") from exc
        return resolved
