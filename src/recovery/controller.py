"""Episode-scoped runtime coordinator for bounded recovery decisions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.capture.recorder import HarnessHook
from src.contracts.trace_event import EventStatus, EventType, TraceEvent
from src.errors import ContractValidationError
from src.recovery.file_not_found_policy import (
    BoundedFileNotFoundRecoveryPolicy,
    RecoveryAction,
    RecoveryDecision,
    RecoveryInput,
)


@dataclass(frozen=True, slots=True)
class RecoveryControllerSnapshot:
    """Serializable controller state for one Episode replay/checkpoint."""

    discovery_cache: tuple[tuple[str, tuple[str, ...]], ...]
    attempted_scopes: tuple[str, ...]
    remaining_discovery_budget: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "discovery_cache": {
                scope: list(candidates) for scope, candidates in self.discovery_cache
            },
            "attempted_scopes": list(self.attempted_scopes),
            "remaining_discovery_budget": self.remaining_discovery_budget,
        }


class RecoveryController:
    """Connect a stateless policy to one Harness Episode lifecycle.

    The controller owns only derived policy state.  It does not execute tools
    and does not mutate canonical events.  The Harness uses the returned
    ``RecoveryDecision`` to schedule the actual action, while an optional Hook
    records the decision as a raw Harness fact.
    """

    def __init__(
        self,
        *,
        policy: BoundedFileNotFoundRecoveryPolicy | None = None,
        hook: "HarnessHook | None" = None,
        initial_discovery_budget: int = 1,
    ) -> None:
        if (
            isinstance(initial_discovery_budget, bool)
            or not isinstance(initial_discovery_budget, int)
            or initial_discovery_budget < 0
        ):
            raise ContractValidationError(
                "initial_discovery_budget must be a non-negative integer"
            )
        self.policy = policy or BoundedFileNotFoundRecoveryPolicy()
        self.hook = hook
        self._remaining_discovery_budget = initial_discovery_budget
        self._discovery_cache: dict[str, tuple[str, ...]] = {}
        self._attempted_scopes: set[str] = set()

    @property
    def snapshot(self) -> RecoveryControllerSnapshot:
        return RecoveryControllerSnapshot(
            discovery_cache=tuple(
                (scope, self._discovery_cache[scope])
                for scope in sorted(self._discovery_cache)
            ),
            attempted_scopes=tuple(sorted(self._attempted_scopes)),
            remaining_discovery_budget=self._remaining_discovery_budget,
        )

    def handle_tool_result(
        self,
        event: TraceEvent,
        *,
        scope_key: str | None = None,
        decision_span_id: str | None = None,
        decision_parent_span_id: str | None = None,
    ) -> RecoveryDecision:
        """Process one structured tool result and optionally emit its decision."""

        self._validate_tool_result(event)
        error_code = event.attributes.get("error_code")
        error_path = event.attributes.get("error_path")
        resolved_scope = scope_key or self._scope_from_error_path(error_path)
        if not isinstance(error_code, str):
            error_code = "UNKNOWN"
        if not isinstance(error_path, str):
            error_path = None
        decision = self.policy.decide(
            RecoveryInput(
                error_code=error_code,
                error_path=error_path,
                scope_key=resolved_scope,
                discovery_cache=self._discovery_cache,
                remaining_discovery_budget=self._remaining_discovery_budget,
                trigger_event_ids=(event.event_id,),
            )
        )
        if decision.action is RecoveryAction.DISCOVER:
            self._attempted_scopes.add(resolved_scope)
            self._discovery_cache.setdefault(resolved_scope, ())
            self._remaining_discovery_budget = decision.budget_after or 0
        if self.hook is not None:
            self.hook.emit_recovery_decision(
                span_id=decision_span_id or f"span-recovery-{event.event_id}",
                parent_span_id=decision_parent_span_id or event.span_id,
                decision=decision,
            )
        return decision

    def decide_after_discovery(
        self,
        error_event: TraceEvent,
        *,
        discovery_event_id: str,
        scope_key: str | None = None,
    ) -> RecoveryDecision:
        """Continue one recovery plan after its bounded discovery result."""

        self._validate_tool_result(error_event)
        if not isinstance(discovery_event_id, str) or not discovery_event_id.strip():
            raise ContractValidationError(
                "discovery_event_id must be a non-empty string"
            )
        resolved_scope = scope_key or self._scope_from_error_path(
            error_event.attributes.get("error_path")
        )
        if resolved_scope not in self._attempted_scopes:
            raise ContractValidationError(
                "cannot continue recovery before a discovery decision for this scope"
            )
        error_path = error_event.attributes.get("error_path")
        decision = self.policy.decide(
            RecoveryInput(
                error_code="FILE_NOT_FOUND",
                error_path=error_path if isinstance(error_path, str) else None,
                scope_key=resolved_scope,
                discovery_cache=self._discovery_cache,
                remaining_discovery_budget=self._remaining_discovery_budget,
                trigger_event_ids=(error_event.event_id, discovery_event_id),
            )
        )
        if self.hook is not None:
            self.hook.emit_recovery_decision(
                span_id=f"span-recovery-after-{discovery_event_id}",
                parent_span_id=error_event.span_id,
                decision=decision,
            )
        return decision

    def record_discovery_result(
        self,
        scope_key: str,
        candidates: Iterable[str],
    ) -> RecoveryControllerSnapshot:
        """Commit one bounded discovery result into the Episode-local cache."""

        if not isinstance(scope_key, str) or not scope_key.strip():
            raise ContractValidationError("scope_key must be a non-empty string")
        if scope_key not in self._attempted_scopes:
            raise ContractValidationError(
                "discovery result arrived for a scope without a discovery decision"
            )
        normalized = tuple(sorted(set(self._validate_candidates(candidates))))
        self._discovery_cache[scope_key] = normalized
        return self.snapshot

    @staticmethod
    def _validate_tool_result(event: TraceEvent) -> None:
        if not isinstance(event, TraceEvent):
            raise TypeError("RecoveryController accepts TraceEvent values")
        if event.event_type is not EventType.TOOL_RESULT:
            raise ContractValidationError("recovery input must be a TOOL_RESULT event")
        if event.status not in {
            EventStatus.FAILED,
            EventStatus.ERROR,
            EventStatus.TIMEOUT,
        }:
            raise ContractValidationError(
                "recovery input must be a failed, error, or timeout TOOL_RESULT"
            )

    @staticmethod
    def _validate_candidates(candidates: Iterable[str]) -> tuple[str, ...]:
        values = tuple(candidates)
        if any(not isinstance(item, str) or not item.strip() for item in values):
            raise ContractValidationError(
                "discovery candidates must contain non-empty strings"
            )
        return values

    @staticmethod
    def _scope_from_error_path(error_path: object) -> str:
        if not isinstance(error_path, str) or not error_path.strip():
            return "."
        normalized = error_path.replace("\\", "/")
        scope = normalized.rsplit("/", 1)[0] if "/" in normalized else "."
        return scope or "."
