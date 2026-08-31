"""Fail-closed dataset and SWE-bench verifier entrypoint tests.

These use minimal synthetic fixtures and never require GPU, a model server, a
network, or a real SWE-bench checkout.
"""

from __future__ import annotations

import unittest

from src.evaluation.swebench import decide_swebench_result, patch_apply_status
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
                "steps": [{"content": '{"action":"read"}', "result": "file-content"}],
            }
        )
        self.assertTrue(trainable)
        self.assertEqual(reasons, ())


class SwebenchTest(unittest.TestCase):
    def test_patch_apply_failure_is_infra_invalid(self) -> None:
        result = patch_apply_status(
            "inst-1", agent_applied=False, test_applied=True, eval_wt="/tmp/eval"
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "INFRA_INVALID")
        self.assertFalse(result["resolved"])
        self.assertIn("agent patch", result["error"])
        result2 = patch_apply_status("inst-1", True, False, "/tmp/eval")
        self.assertEqual(result2["status"], "INFRA_INVALID")
        self.assertIn("test patch", result2["error"])

    def test_pass_to_pass_regression_unresolved(self) -> None:
        ftp = {"t1": {"passed": True, "infra_invalid": False}}
        ptp = {
            "p1": {"passed": True, "infra_invalid": False},
            "p2": {"passed": False, "infra_invalid": False},  # regression
        }
        result = decide_swebench_result("inst-1", ftp, ptp, "/tmp/eval")
        self.assertFalse(result["resolved"])
        self.assertEqual(result["status"], "UNRESOLVED")

    def test_resolved_requires_ftp_and_ptp_all_pass(self) -> None:
        ftp = {"t1": {"passed": True, "infra_invalid": False}}
        ptp = {"p1": {"passed": True, "infra_invalid": False}}
        result = decide_swebench_result("inst-1", ftp, ptp, "/tmp/eval")
        self.assertTrue(result["resolved"])
        self.assertEqual(result["status"], "RESOLVED")

    def test_verifier_timeout_is_infra_invalid_not_reward_zero(self) -> None:
        ftp = {"t1": {"passed": False, "infra_invalid": True, "error": "timeout"}}
        result = decide_swebench_result("inst-1", ftp, {}, "/tmp/eval")
        self.assertEqual(result["status"], "INFRA_INVALID")
        self.assertTrue(any("timeout" in r for r in [result["error"]]))


if __name__ == "__main__":
    unittest.main()
