from __future__ import annotations

import unittest

from src.contracts.trace_event import EventComponent, EventStatus, EventType
from src.recovery.dispatch import ToolDispatchRequest
from src.recovery.execution import (
    DispatchingRecoveryToolExecutor,
    RecoveryExecutionResult,
)
from src.recovery.file_not_found_policy import (
    BoundedFileNotFoundRecoveryPolicy,
    RecoveryAction,
    RecoveryDecision,
    RecoveryInput,
)
from tests.execution_fixtures import make_trace_event


class FakeDispatcher:
    def __init__(self) -> None:
        self.requests: list[ToolDispatchRequest] = []

    def dispatch(self, request: ToolDispatchRequest) -> RecoveryExecutionResult:
        self.requests.append(request)
        event = make_trace_event(
            event_id="evt-dispatched-result",
            sequence=10,
            span_id="span-dispatched",
            parent_span_id=request.parent_span_id,
            event_type=EventType.TOOL_RESULT,
            component=EventComponent.TOOL,
            status=EventStatus.SUCCEEDED,
            attributes={"tool_name": request.tool_name, "result": {"content": []}},
            artifact_refs=(),
        )
        return RecoveryExecutionResult(
            event=event,
            event_ids=("evt-dispatched-call", event.event_id),
        )


class RecoveryDispatchTest(unittest.TestCase):
    def _decision(self) -> RecoveryDecision:
        return BoundedFileNotFoundRecoveryPolicy().decide(
            RecoveryInput(
                error_code="FILE_NOT_FOUND",
                error_path="workspace/missing-1.json",
                scope_key="workspace",
                remaining_discovery_budget=1,
                trigger_event_ids=("evt-error",),
            )
        )

    def test_request_preserves_decision_lineage_and_is_deterministic(self) -> None:
        decision = self._decision()
        first = ToolDispatchRequest.from_decision(decision, parent_span_id="span-error")
        second = ToolDispatchRequest.from_decision(decision, parent_span_id="span-error")

        self.assertEqual(first.request_id, second.request_id)
        self.assertEqual(first.action, RecoveryAction.DISCOVER)
        self.assertEqual(first.tool_name, "find")
        self.assertEqual(first.trigger_event_ids, ("evt-error",))
        self.assertEqual(first.decision_checksum, decision.checksum)
        self.assertEqual(first.parent_span_id, "span-error")
        self.assertEqual(first.checksum, second.checksum)

    def test_terminate_cannot_become_a_tool_request(self) -> None:
        decision = RecoveryDecision(
            action=RecoveryAction.TERMINATE,
            reason_code="RECOVERY_BUDGET_EXHAUSTED",
        )

        with self.assertRaisesRegex(ValueError, "only DISCOVER and READ"):
            ToolDispatchRequest.from_decision(decision, parent_span_id=None)

    def test_dispatching_executor_returns_canonical_tool_result(self) -> None:
        dispatcher = FakeDispatcher()
        executor = DispatchingRecoveryToolExecutor(dispatcher)
        result = executor.execute(self._decision(), parent_span_id="span-error")

        self.assertEqual(len(dispatcher.requests), 1)
        self.assertEqual(dispatcher.requests[0].tool_name, "find")
        self.assertEqual(result.event.event_type, EventType.TOOL_RESULT)
        self.assertEqual(result.event.status, EventStatus.SUCCEEDED)
        self.assertEqual(result.event_ids, ("evt-dispatched-call", "evt-dispatched-result"))


if __name__ == "__main__":
    unittest.main()
