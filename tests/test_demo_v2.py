from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.demo_v2 import generate_v2_demo


class DemoV2Test(unittest.TestCase):
    def test_generates_replayable_real_pi_decision_and_teacher_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            obsolete = Path(directory) / "pi-real-probe-conversion.json"
            obsolete.write_text("stale pre-release evidence")
            summary = generate_v2_demo(directory)
            root = Path(directory)
            self.assertFalse(obsolete.exists())
            self.assertEqual(summary["gate_decision"], "REJECT")
            self.assertEqual(summary["paired_coverage"], 1.0)
            self.assertEqual(summary["observed_outcomes"], {
                "control_successes": 0,
                "candidate_successes": 3,
            })
            self.assertEqual(summary["target_slice"], {
                "reason_code": "OBSERVED_TOOL_ERROR_LOOP",
                "control": 3,
                "candidate": 0,
            })
            self.assertEqual(summary["training_semantics"]["sft_candidates"], 3)
            self.assertEqual(summary["training_semantics"]["on_policy_rl_candidates"], 0)
            for name in (
                "raw-events.jsonl",
                "comparison.json",
                "gate-result.json",
                "capture-evidence.json",
                "training-candidates.json",
                "observatory.html",
                "summary.json",
                "artifact-manifest.json",
            ):
                self.assertTrue((root / name).is_file(), name)
            gate = json.loads((root / "gate-result.json").read_text())
            failed = {item["rule"] for item in gate["checks"] if item["passed"] is False}
            self.assertEqual(
                failed, {"max_token_increase_ratio", "max_latency_increase_ratio"}
            )
            first_checksum = json.loads((root / "summary.json").read_text())["evidence_checksums"]
            generate_v2_demo(directory)
            second_checksum = json.loads((root / "summary.json").read_text())["evidence_checksums"]
            self.assertEqual(first_checksum, second_checksum)


if __name__ == "__main__":
    unittest.main()
