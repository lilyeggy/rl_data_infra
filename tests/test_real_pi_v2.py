from __future__ import annotations

import unittest
from pathlib import Path

from src.capture.pi_adapter import read_pi_ndjson
from src.contracts.agent_episode import TaskStatus
from examples.legacy_scenarios.real_pi_v2 import verify_reference_answer


FIXTURES = Path(__file__).parent / "fixtures" / "pi" / "v2-real"


class RealPiV2VerifierTest(unittest.TestCase):
    def test_verifier_accepts_candidate_without_trusting_claim_alone(self) -> None:
        records, _ = read_pi_ndjson((FIXTURES / "task-1-candidate.ndjson").read_text())
        outcome, evidence = verify_reference_answer(records, task_index=1)
        self.assertEqual(outcome.task_status, TaskStatus.SUCCESS)
        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["observed"]["failure_count"], 2)

    def test_verifier_rejects_control_final_answer(self) -> None:
        records, _ = read_pi_ndjson((FIXTURES / "task-1-control.ndjson").read_text())
        outcome, evidence = verify_reference_answer(records, task_index=1)
        self.assertEqual(outcome.task_status, TaskStatus.FAILURE)
        self.assertFalse(evidence["passed"])


if __name__ == "__main__":
    unittest.main()
