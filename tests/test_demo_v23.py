from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from examples.legacy_scenarios.demo_v23 import generate_v23_storage_smoke


class DemoV23Test(unittest.TestCase):
    def test_generates_partitioned_storage_smoke_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summary = generate_v23_storage_smoke(temporary)

            self.assertEqual(summary["release"], "v2.3")
            self.assertEqual(summary["first_append_count"], summary["input_event_count"])
            self.assertEqual(summary["replay_duplicate_count"], summary["input_event_count"])
            self.assertEqual(summary["quarantine_count"], 0)
            self.assertEqual(
                summary["read_back_event_count"], summary["input_event_count"]
            )
            self.assertFalse(summary["distributed_exactly_once"])
            self.assertEqual(summary["compaction_count"], 3)
            self.assertEqual(summary["schema_validation"], "explicit-supported-version-only")
            self.assertTrue((Path(temporary) / "storage").is_dir())
            self.assertTrue((Path(temporary) / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
