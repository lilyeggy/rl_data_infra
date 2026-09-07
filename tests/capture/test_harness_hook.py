from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.capture.event_writer import EventJsonlReader, EventWriter
from src.capture.recorder import HarnessHook, TraceRecorder
from src.contracts.trace_event import EventType
from src.recovery.file_not_found_policy import (
    BoundedFileNotFoundRecoveryPolicy,
    RecoveryInput,
)


class HarnessHookTest(unittest.TestCase):
    def _hook(self, temporary: str) -> HarnessHook:
        recorder = TraceRecorder(
            EventWriter(Path(temporary) / "raw-events.jsonl"),
            run_id="run-hook",
            episode_id="episode-hook",
            trace_id="trace-hook",
            clock=lambda: "2026-08-18T00:00:00Z",
            id_factory=iter(("decision", "retry", "terminate", "context", "compact")).__next__,
        )
        return HarnessHook(recorder)

    def test_decision_preserves_trigger_and_policy_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            event = self._hook(temporary).emit_decision(
                span_id="span-decision",
                parent_span_id=None,
                decision="DISCOVER",
                reason_code="FILE_NOT_FOUND",
                details={"pattern": "task-*.json"},
                trigger_event_ids=("evt-file-error",),
                policy_name="bounded-file-recovery",
                policy_version="file-not-found-recovery/v2.1",
                scope_key="workspace:/task-1",
                budget_before=1,
                budget_after=0,
                cache_hit=False,
            )

            self.assertEqual(event.event_type, EventType.HARNESS_DECISION)
            self.assertEqual(event.attributes["trigger_event_ids"], ("evt-file-error",))
            self.assertEqual(event.attributes["policy_version"], "file-not-found-recovery/v2.1")
            self.assertEqual(event.attributes["budget_before"], 1)
            self.assertEqual(event.attributes["budget_after"], 0)
            self.assertFalse(event.attributes["cache_hit"])

    def test_recovery_decision_is_emitted_without_recomputing_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            policy = BoundedFileNotFoundRecoveryPolicy()
            decision = policy.decide(
                RecoveryInput(
                    error_code="FILE_NOT_FOUND",
                    error_path="workspace/missing-1.json",
                    scope_key="workspace",
                    remaining_discovery_budget=1,
                    trigger_event_ids=("evt-file-error",),
                )
            )
            event = self._hook(temporary).emit_recovery_decision(
                span_id="span-recovery",
                parent_span_id=None,
                decision=decision,
            )

            self.assertEqual(event.attributes["decision"], "DISCOVER")
            self.assertEqual(event.attributes["reason_code"], "FILE_NOT_FOUND_DISCOVERY")
            self.assertEqual(event.attributes["trigger_event_ids"], ("evt-file-error",))
            self.assertEqual(
                event.attributes["details"]["decision_checksum"], decision.checksum
            )
            self.assertEqual(event.attributes["budget_after"], 0)

    def test_hook_events_are_append_only_and_replayable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            hook = self._hook(temporary)
            hook.emit_retry_scheduled(
                span_id="span-retry",
                parent_span_id="span-error",
                attempt=1,
                reason="FILE_NOT_FOUND",
                delay_ms=0,
                trigger_event_ids=("evt-file-error",),
                policy_name="bounded-file-recovery",
                policy_version="file-not-found-recovery/v2.1",
                budget_before=1,
                budget_after=0,
            )
            hook.emit_termination_decided(
                span_id="span-terminate",
                reason="RECOVERY_BUDGET_EXHAUSTED",
                verifier_observed=False,
                trigger_event_ids=("evt-file-error",),
                policy_name="bounded-file-recovery",
                policy_version="file-not-found-recovery/v2.1",
            )

            result = EventJsonlReader.read(Path(temporary) / "raw-events.jsonl")
            self.assertEqual(len(result.events), 2)
            self.assertEqual(
                [event.event_type for event in result.events],
                [EventType.RETRY_SCHEDULED, EventType.TERMINATION_DECIDED],
            )
            self.assertEqual(result.events[0].attributes["trigger_event_ids"], ("evt-file-error",))
            self.assertEqual(result.events[1].attributes["reason"], "RECOVERY_BUDGET_EXHAUSTED")

    def test_context_events_keep_decisions_separate_from_diagnoses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            hook = self._hook(temporary)
            selected = hook.emit_context_selected(
                span_id="span-context",
                parent_span_id=None,
                selection="preserve-tool-error",
                trigger_event_ids=("evt-file-error",),
                policy_name="bounded-file-recovery",
                policy_version="file-not-found-recovery/v2.1",
                details={"source": "harness-state"},
            )
            compacted = hook.emit_context_compacted(
                span_id="span-compact",
                parent_span_id=None,
                before_tokens=100,
                after_tokens=60,
                trigger_event_ids=(selected.event_id,),
                policy_name="context-policy",
                policy_version="context/v1",
            )

            self.assertEqual(selected.event_type, EventType.CONTEXT_SELECTED)
            self.assertEqual(compacted.event_type, EventType.CONTEXT_COMPACTED)
            self.assertEqual(compacted.attributes["trigger_event_ids"], (selected.event_id,))
            self.assertNotIn("reason_code", selected.attributes)


if __name__ == "__main__":
    unittest.main()
