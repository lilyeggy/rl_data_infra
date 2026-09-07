"""Harness-neutral dispatch contract for executing recovery actions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from src.contracts._json import sha256_json, thaw_json
from src.errors import ContractValidationError
from src.recovery.file_not_found_policy import RecoveryAction, RecoveryDecision

if TYPE_CHECKING:
    from src.recovery.execution import RecoveryExecutionResult


DISPATCH_REQUEST_VERSION = "recovery-tool-dispatch/v1"


@dataclass(frozen=True, slots=True)
class ToolDispatchRequest:
    """Immutable request passed from the recovery plane to a Harness executor."""

    request_id: str
    action: RecoveryAction
    tool_name: str
    arguments: Mapping[str, Any]
    trigger_event_ids: tuple[str, ...]
    decision_checksum: str
    policy_version: str
    parent_span_id: str | None
    schema_version: str = DISPATCH_REQUEST_VERSION

    @classmethod
    def from_decision(
        cls,
        decision: RecoveryDecision,
        *,
        parent_span_id: str | None,
    ) -> "ToolDispatchRequest":
        if decision.action not in {RecoveryAction.DISCOVER, RecoveryAction.READ}:
            raise ContractValidationError(
                "only DISCOVER and READ decisions can become tool dispatch requests"
            )
        if not decision.tool_name:
            raise ContractValidationError("tool dispatch requires a tool_name")
        payload = {
            "action": decision.action.value,
            "tool_name": decision.tool_name,
            "arguments": thaw_json(decision.arguments),
            "trigger_event_ids": list(decision.trigger_event_ids),
            "decision_checksum": decision.checksum,
            "policy_version": decision.policy_version,
            "parent_span_id": parent_span_id,
        }
        request_id = f"recovery-request-{sha256_json(payload)[:20]}"
        return cls(
            request_id=request_id,
            action=decision.action,
            tool_name=decision.tool_name,
            arguments=decision.arguments,
            trigger_event_ids=decision.trigger_event_ids,
            decision_checksum=decision.checksum,
            policy_version=decision.policy_version,
            parent_span_id=parent_span_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "action": self.action.value,
            "tool_name": self.tool_name,
            "arguments": thaw_json(self.arguments),
            "trigger_event_ids": list(self.trigger_event_ids),
            "decision_checksum": self.decision_checksum,
            "policy_version": self.policy_version,
            "parent_span_id": self.parent_span_id,
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


class RecoveryToolDispatcher(Protocol):
    """Interface implemented by a real Harness tool dispatch layer."""

    def dispatch(self, request: ToolDispatchRequest) -> "RecoveryExecutionResult":
        """Execute one request and return a canonical tool result."""
        ...
