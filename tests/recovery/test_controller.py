from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.capture.event_writer import EventJsonlReader, EventWriter
from src.capture.recorder import HarnessHook, TraceRecorder
from src.contracts.trace_event import EventComponent, EventStatus, EventType
from src.errors import ContractValidationError
from src.recovery.controller import RecoveryController
from src.recovery.file_not_found_policy import RecoveryAction
from tests.execution_fixtures import make_trace_event


class RecoveryControllerTest(unittest.TestCase):
    @staticmethod
    def _error(event_id: str, *, path: str = "workspace/missing.json"):
        return make_trace_event(
            event_id=event_id,
            span_id=f"span-{event_id}",
            parent_span_id=None,
            event_type=EventType.TOOL_RESULT,
            component=EventComponent.TOOL,
            status=EventStatus.ERROR,
            attributes={
                "tool_name": "read",
                "error_code": "FILE_NOT_FOUND",
                "error_path": path,
            },
            artifact_refs=(),
        )

    def test_discovery_is_scoped_and_result_enables_one_read(self) -> None:
        controller = RecoveryController(initial_discovery_budget=1)

        first = controller.handle_tool_result(self._error("evt-error-1"))
        self.assertEqual(first.action, RecoveryAction.DISCOVER)
        self.assertEqual(controller.snapshot.remaining_discovery_budget, 0)
        self.assertEqual(controller.snapshot.attempted_scopes, ("workspace",))

        controller.record_discovery_result("workspace", ("workspace/task.json",))
        second = controller.handle_tool_result(self._error("evt-error-2"))

        self.assertEqual(second.action, RecoveryAction.READ)
        self.assertEqual(second.arguments["path"], "workspace/task.json")
        self.assertEqual(controller.snapshot.discovery_cache[0][1], ("workspace/task.json",))

    def test_scope_is_not_searched_again_before_result_arrives(self) -> None:
        controller = RecoveryController(initial_discovery_budget=1)

        first = controller.handle_tool_result(self._error("evt-error-1"))
        second = controller.handle_tool_result(self._error("evt-error-2"))

        self.assertEqual(first.action, RecoveryAction.DISCOVER)
        self.assertEqual(second.action, RecoveryAction.TERMINATE)
        self.assertEqual(second.reason_code, "NO_DISCOVERY_CANDIDATE")
        self.assertEqual(controller.snapshot.attempted_scopes, ("workspace",))

    def test_runtime_decision_is_emitted_only_when_hook_is_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            writer = EventWriter(Path(temporary) / "raw-events.jsonl")
            recorder = TraceRecorder(
                writer,
                run_id="run-controller",
                episode_id="episode-controller",
                trace_id="trace-controller",
                clock=lambda: "2026-08-18T00:00:00Z",
                id_factory=iter(("decision",)).__next__,
            )
            controller = RecoveryController(
                hook=HarnessHook(recorder),
                initial_discovery_budget=1,
            )
            decision = controller.handle_tool_result(self._error("evt-error-1"))

            events = EventJsonlReader.read(Path(temporary) / "raw-events.jsonl").events
            self.assertEqual(decision.action, RecoveryAction.DISCOVER)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].event_type, EventType.HARNESS_DECISION)
            self.assertEqual(events[0].attributes["decision"], "DISCOVER")
            self.assertEqual(events[0].attributes["trigger_event_ids"], ("evt-error-1",))

    def test_invalid_runtime_inputs_are_rejected(self) -> None:
        controller = RecoveryController()
        success = replace(
            self._error("evt-success"),
            status=EventStatus.SUCCEEDED,
        )
        with self.assertRaises(ContractValidationError):
            controller.handle_tool_result(success)
        with self.assertRaises(ContractValidationError):
            controller.record_discovery_result("workspace", ("task.json",))


if __name__ == "__main__":
    unittest.main()
