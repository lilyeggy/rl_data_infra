from __future__ import annotations

import unittest
from pathlib import Path

from src.capture.pi_adapter import (
    PiJsonAdapter,
    PiOutcomeDeclaration,
    PiRunConfig,
    read_pi_ndjson,
)
from src.contracts.agent_episode import (
    EpisodeVerifierStatus,
    ExecutionValidity,
    TaskStatus,
)
from src.recovery.file_not_found_policy import RecoveryAction
from src.recovery.replay import replay_file_not_found_recovery


FIXTURES = Path(__file__).parents[1] / "fixtures" / "pi" / "v2-real"


class RecoveryReplayTest(unittest.TestCase):
    def _events(self, fixture_name: str):
        records, issues = read_pi_ndjson((FIXTURES / fixture_name).read_text())
        self.assertFalse(issues)
        result = PiJsonAdapter().convert(
            records,
            run_id="run-replay",
            episode_id=f"episode-{fixture_name}",
            trace_id=f"trace-{fixture_name}",
            config=PiRunConfig(),
            declared_outcome=PiOutcomeDeclaration(
                task_status=TaskStatus.UNKNOWN,
                execution_validity=ExecutionValidity.UNKNOWN,
                verifier_status=EpisodeVerifierStatus.UNKNOWN,
            ),
            normalize_tool_errors=True,
        )
        return result.events

    def test_candidate_fixture_matches_bounded_recovery_plan(self) -> None:
        report = replay_file_not_found_recovery(self._events("task-1-candidate.ndjson"))

        self.assertEqual(len(report.steps), 2)
        self.assertEqual(report.steps[0].decision.action, RecoveryAction.DISCOVER)
        self.assertEqual(report.steps[0].action_alignment, "MATCH")
        self.assertEqual(report.steps[1].decision.action, RecoveryAction.READ)
        self.assertEqual(report.steps[1].action_alignment, "MATCH")
        self.assertFalse(report.warnings)
        self.assertEqual(report.steps[1].decision.selected_candidates, (
            "/private/tmp/v2-pi-real/task-1/task-1.json",
        ))

    def test_control_fixture_does_not_claim_a_missing_harness_decision(self) -> None:
        report = replay_file_not_found_recovery(self._events("task-1-control.ndjson"))

        self.assertEqual(len(report.steps), 1)
        self.assertEqual(report.steps[0].decision.action, RecoveryAction.DISCOVER)
        self.assertEqual(report.steps[0].action_alignment, "NO_OBSERVED_ACTION")
        self.assertTrue(report.warnings)
        self.assertNotIn("HARNESS_DECISION", report.to_dict())

    def test_replay_is_deterministic(self) -> None:
        events = self._events("task-2-candidate.ndjson")
        first = replay_file_not_found_recovery(events)
        second = replay_file_not_found_recovery(tuple(reversed(events)))

        self.assertEqual(first.checksum, second.checksum)
        self.assertEqual(first.to_dict(), second.to_dict())


if __name__ == "__main__":
    unittest.main()
