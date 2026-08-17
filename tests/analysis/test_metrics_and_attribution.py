from __future__ import annotations

import unittest
from dataclasses import replace

from src.analysis.attribution import AttributionEngine, FailureLayer
from src.analysis.metrics import compute_episode_metrics
from src.assembly.episode_assembler import EpisodeAssembler
from src.contracts.agent_episode import CaptureCapability
from src.contracts.trace_event import EventComponent, EventStatus, EventType
from tests.execution_fixtures import (
    make_complete_event_stream,
    make_episode_context,
)


class MetricsAndAttributionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assembler = EpisodeAssembler()
        self.context = make_episode_context()

    def _assemble(self, events, context=None):
        return self.assembler.assemble(
            events,
            contexts={"episode-001": context or self.context},
        ).episodes[0]

    def test_metrics_are_derived_from_events_and_preserve_lineage(self) -> None:
        episode = self._assemble(make_complete_event_stream())

        metrics = compute_episode_metrics(episode)

        self.assertEqual(metrics.duration_ms, 6000)
        self.assertEqual(metrics.turn_count, 1)
        self.assertEqual(metrics.tool_call_count, 1)
        self.assertEqual(metrics.input_tokens, 10)
        self.assertEqual(metrics.output_tokens, 4)
        self.assertEqual(metrics.model_latency_ms, 100)
        self.assertEqual(metrics.tool_latency_ms, 25)
        self.assertEqual(metrics.verifier_latency_ms, 50)
        self.assertEqual(metrics.input_episode_checksum, episode.checksum)
        self.assertIsNone(metrics.estimated_cost)

    def test_missing_tool_capability_is_not_fabricated_as_zero(self) -> None:
        context = make_episode_context(
            capabilities=frozenset(
                {
                    CaptureCapability.MODEL_IO,
                    CaptureCapability.MODEL_TOKEN_USAGE,
                }
            )
        )
        metrics = compute_episode_metrics(
            self._assemble(make_complete_event_stream(), context)
        )

        self.assertIsNone(metrics.tool_call_count)
        self.assertIsNone(metrics.duplicate_action_rate)
        self.assertTrue(any("TOOL_IO" in item for item in metrics.not_observable))

    def test_success_has_no_failure_diagnosis(self) -> None:
        report = AttributionEngine().analyze(
            self._assemble(make_complete_event_stream())
        )

        self.assertFalse(report.diagnoses)

    def test_infra_invalid_is_attributed_to_observable_component(self) -> None:
        events = list(make_complete_event_stream())
        terminal = events.pop()
        events.append(
            replace(
                events[-1],
                event_id="evt-sandbox-timeout",
                sequence=6,
                timestamp="2026-08-14T00:00:06Z",
                span_id="span-sandbox",
                parent_span_id=None,
                event_type=EventType.SANDBOX_COMMAND,
                component=EventComponent.SANDBOX,
                status=EventStatus.TIMEOUT,
                attributes={"command": "tests", "timeout_seconds": 30},
            )
        )
        events.append(
            replace(
                terminal,
                sequence=7,
                timestamp="2026-08-14T00:00:07Z",
                attributes={
                    "task_status": "UNKNOWN",
                    "execution_validity": "INFRA_INVALID",
                    "verifier_status": "TIMEOUT",
                    "score": None,
                    "termination_reason": "SANDBOX_TIMEOUT",
                    "evidence_event_ids": ["evt-sandbox-timeout"],
                },
            )
        )

        report = AttributionEngine().analyze(self._assemble(tuple(events)))

        diagnosis = next(item for item in report.diagnoses if item.layer is FailureLayer.SANDBOX)
        self.assertEqual(diagnosis.reason_code, "SANDBOX_OPERATION_TIMEOUT")
        self.assertEqual(diagnosis.evidence_event_ids, ("evt-sandbox-timeout",))

    def test_tool_error_loop_is_harness_only_when_decisions_are_observable(self) -> None:
        events = list(make_complete_event_stream())
        events[3] = replace(events[3], status=EventStatus.FAILED, attributes={"exit_code": 1})
        decision = replace(
            events[0],
            event_id="evt-retry-decision",
            sequence=4,
            timestamp="2026-08-14T00:00:04Z",
            span_id="span-retry-decision",
            parent_span_id="span-model",
            event_type=EventType.HARNESS_DECISION,
            component=EventComponent.HARNESS,
            status=EventStatus.SUCCEEDED,
            attributes={"decision": "RETRY_UNCHANGED"},
        )
        repeated_call = replace(
            events[2],
            event_id="evt-tool-call-repeated",
            sequence=5,
            timestamp="2026-08-14T00:00:05Z",
            span_id="span-tool-repeated",
        )
        repeated_result = replace(
            events[3],
            event_id="evt-tool-result-repeated",
            sequence=6,
            timestamp="2026-08-14T00:00:06Z",
            span_id="span-tool-repeated",
        )
        verification_start = replace(events[4], sequence=7, timestamp="2026-08-14T00:00:07Z")
        verification_finish = replace(
            events[5],
            sequence=8,
            timestamp="2026-08-14T00:00:08Z",
            status=EventStatus.FAILED,
            attributes={"passed": False},
        )
        terminal = replace(
            events[6],
            sequence=9,
            timestamp="2026-08-14T00:00:09Z",
            attributes={
                "task_status": "FAILURE",
                "execution_validity": "VALID",
                "verifier_status": "FAILED",
                "score": 0,
                "termination_reason": "VERIFIER_FAILED",
                "evidence_event_ids": ["evt-verification-finished"],
            },
        )
        trace = tuple(events[:4]) + (
            decision,
            repeated_call,
            repeated_result,
            verification_start,
            verification_finish,
            terminal,
        )

        observed = AttributionEngine().analyze(self._assemble(trace))
        black_box_context = make_episode_context(
            capabilities=frozenset(
                item
                for item in self.context.capabilities
                if item is not CaptureCapability.HARNESS_DECISIONS
            )
        )
        black_box = AttributionEngine().analyze(
            self._assemble(trace, black_box_context)
        )

        self.assertIs(observed.diagnoses[0].layer, FailureLayer.HARNESS)
        self.assertIn("evt-retry-decision", observed.diagnoses[0].evidence_event_ids)
        self.assertIs(black_box.diagnoses[0].layer, FailureLayer.UNKNOWN)
        self.assertTrue(black_box.insufficient_evidence_rules)


if __name__ == "__main__":
    unittest.main()
