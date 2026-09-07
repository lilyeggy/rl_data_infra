from __future__ import annotations

import unittest
from pathlib import Path

from src.capture.pi_adapter import (
    PiJsonAdapter,
    PiOutcomeDeclaration,
    read_pi_ndjson,
)
from src.contracts.agent_episode import (
    EpisodeVerifierStatus,
    ExecutionValidity,
    TaskStatus,
)
from examples.legacy_scenarios.real_pi_v2 import verify_reference_answer


FIXTURES = Path(__file__).parent / "fixtures" / "pi" / "v2.1-live"


class RealPiV21LiveFixtureTest(unittest.TestCase):
    def test_live_candidate_fixtures_replay_to_success(self) -> None:
        for task_index in (1, 2, 3):
            path = FIXTURES / f"candidate-{task_index}.ndjson"
            records, issues = read_pi_ndjson(path.read_text())
            self.assertFalse(issues)
            declaration, _ = verify_reference_answer(records, task_index=task_index)
            self.assertEqual(
                declaration.task_status, TaskStatus.SUCCESS, msg=f"candidate-{task_index}"
            )
            self.assertEqual(declaration.execution_validity, ExecutionValidity.VALID)

    def test_live_control_fixtures_replay_to_valid_failure(self) -> None:
        for task_index in (1, 2, 3):
            path = FIXTURES / f"control-{task_index}.ndjson"
            records, issues = read_pi_ndjson(path.read_text())
            self.assertFalse(issues)
            declaration, _ = verify_reference_answer(records, task_index=task_index)
            self.assertEqual(
                declaration.task_status, TaskStatus.FAILURE, msg=f"control-{task_index}"
            )
            self.assertEqual(declaration.execution_validity, ExecutionValidity.VALID)

    def test_live_fixtures_redact_content_thinking_blocks(self) -> None:
        for path in sorted(FIXTURES.glob("*.ndjson")):
            records = read_pi_ndjson(path.read_text())[0]
            for record in records:
                if record.get("type") != "message_end":
                    continue
                message = record.get("message") or {}
                for item in message.get("content") or []:
                    if isinstance(item, dict) and item.get("type") == "thinking":
                        self.assertEqual(
                            item.get("thinking"),
                            "[REDACTED]",
                            msg=f"plaintext thinking block in {path.name}",
                        )
                        self.assertEqual(item.get("thinkingSignature"), "[REDACTED]")


if __name__ == "__main__":
    unittest.main()
