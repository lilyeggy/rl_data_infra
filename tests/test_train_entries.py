"""Fail-closed dataset entrypoint tests.

These use minimal synthetic fixtures and never require GPU, a model server, a
network, or a real SWE-bench checkout.
"""

from __future__ import annotations

import unittest

from src.learning.sft_examples import classify_sft_example

TC = "test-temperature"


class BuildSftDatasetTest(unittest.TestCase):
    def test_verified_false_never_trains(self) -> None:
        trainable, reasons = classify_sft_example(
            {
                "messages": [{"role": "assistant", "content": "x"}],
                "final_answer": "42",
                "certification_verdict": "REJECTED",
                "steps": [],
            }
        )
        self.assertFalse(trainable)
        self.assertTrue(any("certification" in r for r in reasons))

    def test_empty_final_answer_never_trains(self) -> None:
        trainable, reasons = classify_sft_example(
            {
                "messages": [],
                "final_answer": "   ",
                "certification_verdict": "ELIGIBLE",
                "steps": [],
            }
        )
        self.assertFalse(trainable)
        self.assertTrue(any("empty final answer" in r for r in reasons))

    def test_missing_tool_result_pairing_never_trains(self) -> None:
        trainable, reasons = classify_sft_example(
            {
                "messages": [],
                "final_answer": "42",
                "certification_verdict": "ELIGIBLE",
                "steps": [{"content": '{"action":"read"}', "result": None}],
            }
        )
        self.assertFalse(trainable)
        self.assertTrue(any("tool result pairing missing" in r for r in reasons))

    def test_verified_and_complete_trains(self) -> None:
        trainable, reasons = classify_sft_example(
            {
                "messages": [{"role": "assistant", "content": "ok"}],
                "final_answer": '{"answer": 42}',
                "certification_verdict": "ELIGIBLE",
                "steps": [],
            }
        )
        self.assertTrue(trainable)
        self.assertEqual(reasons, ())

    def test_missing_observation_in_messages_never_trains(self) -> None:
        trainable, reasons = classify_sft_example(
            {
                "messages": [
                    {"role": "user", "content": "fix it"},
                    {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
                    {"role": "assistant", "content": "done"},
                ],
                "final_answer": "done",
                "certification_verdict": "ELIGIBLE",
                "steps": [{"tool_call_id": "call-1", "content": "x", "result": "y"}],
            }
        )
        self.assertFalse(trainable)
        self.assertTrue(any("before prior tool observations" in reason for reason in reasons))


if __name__ == "__main__":
    unittest.main()
