from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from examples.legacy_scenarios.demo_v1 import generate_v1_demo


class DemoV1Test(unittest.TestCase):
    def test_demo_is_idempotent_and_emits_three_outcome_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact"

            first = generate_v1_demo(output)
            raw_before = (output / "raw-events.jsonl").read_bytes()
            second = generate_v1_demo(output)

            self.assertEqual(first, second)
            self.assertEqual((output / "raw-events.jsonl").read_bytes(), raw_before)
            self.assertEqual(first["episode_count"], 3)
            outcomes = {
                (item["task_status"], item["execution_validity"])
                for item in first["episodes"]
            }
            self.assertEqual(
                outcomes,
                {
                    ("SUCCESS", "VALID"),
                    ("FAILURE", "VALID"),
                    ("UNKNOWN", "INFRA_INVALID"),
                },
            )
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["assembly_output_checksum"], first["assembly_output_checksum"])


if __name__ == "__main__":
    unittest.main()
