from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.capture.event_writer import EventJsonlReader, EventWriter
from src.capture.recorder import HarnessHook, TraceRecorder
from src.contracts.trace_event import EventComponent, EventStatus, EventType
from src.recovery.controller import RecoveryController
from src.recovery.execution import (
    LocalRecoveryToolExecutor,
    RecoveryExecutionAdapter,
)
from src.recovery.file_not_found_policy import RecoveryAction, RecoveryDecision
from tests.execution_fixtures import make_trace_event


class RecoveryExecutionTest(unittest.TestCase):
    @staticmethod
    def _error(event_id: str = "evt-file-error"):
        return make_trace_event(
            event_id=event_id,
            span_id="span-error",
            parent_span_id=None,
            event_type=EventType.TOOL_RESULT,
            component=EventComponent.TOOL,
            status=EventStatus.ERROR,
            attributes={
                "tool_name": "read",
                "error_code": "FILE_NOT_FOUND",
                "error_path": "workspace/missing-1.json",
            },
            artifact_refs=(),
        )

    def _adapter(self, temporary: str) -> tuple[RecoveryExecutionAdapter, Path]:
        root = Path(temporary)
        workspace = root / "workspace"
        workspace.mkdir()
        recorder = TraceRecorder(
            EventWriter(root / "raw-events.jsonl"),
            run_id="run-execution",
            episode_id="episode-execution",
            trace_id="trace-execution",
            clock=lambda: "2026-08-18T00:00:00Z",
            id_factory=iter(str(index) for index in range(20)).__next__,
        )
        controller = RecoveryController(
            hook=HarnessHook(recorder),
            initial_discovery_budget=1,
        )
        executor = LocalRecoveryToolExecutor(workspace=root, recorder=recorder)
        return RecoveryExecutionAdapter(controller=controller, executor=executor), workspace

    def test_unique_file_executes_discover_then_read_with_decision_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter, workspace = self._adapter(temporary)
            target = workspace / "task-1.json"
            target.write_text('{"task_id":"task-1"}\n')

            report = adapter.recover(self._error())
            events = EventJsonlReader.read(Path(temporary) / "raw-events.jsonl").events

            self.assertEqual(
                [decision.action for decision in report.decisions],
                [RecoveryAction.DISCOVER, RecoveryAction.READ],
            )
            self.assertEqual(len(report.executed_event_ids), 4)
            self.assertIsNone(report.terminal_reason)
            self.assertEqual(events[0].event_type, EventType.HARNESS_DECISION)
            self.assertEqual(events[0].attributes["decision"], "DISCOVER")
            self.assertEqual(events[3].event_type, EventType.HARNESS_DECISION)
            self.assertEqual(events[3].attributes["decision"], "READ")
            self.assertEqual(events[-1].event_type, EventType.TOOL_RESULT)
            self.assertIn("task-1", str(events[-1].attributes["result"]))

    def test_ambiguous_discovery_terminates_without_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter, workspace = self._adapter(temporary)
            (workspace / "a-1.json").write_text("a")
            (workspace / "b-1.json").write_text("b")

            report = adapter.recover(self._error("evt-ambiguous"))

            self.assertEqual(
                [decision.action for decision in report.decisions],
                [RecoveryAction.DISCOVER, RecoveryAction.TERMINATE],
            )
            self.assertEqual(report.terminal_reason, "AMBIGUOUS_DISCOVERY_CANDIDATES")
            self.assertEqual(len(report.executed_event_ids), 2)

    def test_local_executor_rejects_workspace_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter, workspace = self._adapter(temporary)
            (workspace / "task-1.json").write_text("safe")
            outside = Path(temporary).parent / "outside-recovery-test.json"
            try:
                outside.write_text("outside")
                report = adapter.recover(self._error("evt-safe"))
                self.assertEqual(report.decisions[0].action, RecoveryAction.DISCOVER)
                self.assertEqual(report.decisions[-1].action, RecoveryAction.READ)
                self.assertNotIn("outside", str(report.to_dict()))
                escaped = adapter.executor.execute(
                    RecoveryDecision(
                        action=RecoveryAction.READ,
                        reason_code="TEST_ESCAPE",
                        tool_name="read",
                        arguments={"path": "../outside-recovery-test.json"},
                    ),
                    parent_span_id=None,
                )
                self.assertEqual(escaped.event.status, EventStatus.ERROR)
                self.assertNotIn("outside", str(escaped.event.attributes["result"]))
            finally:
                outside.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
