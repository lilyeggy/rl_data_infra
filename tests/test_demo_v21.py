from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from examples.legacy_scenarios.demo_v21 import generate_v21_recovery_replay


class DemoV21Test(unittest.TestCase):
    def test_generates_derived_recovery_replay_without_mutating_v2_fixtures(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "pi" / "v2-real" / "task-1-candidate.ndjson"
        before = fixture.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            summary = generate_v21_recovery_replay(temporary)
            self.assertEqual(summary["release"], "v2.1")
            self.assertTrue(summary["candidate_replay_steps_match"])
            self.assertTrue(summary["control_replay_has_no_claimed_decision"])
            self.assertEqual(summary["decision_capture"], "NOT_OBSERVABLE")
            self.assertFalse(summary["raw_events_modified"])
            self.assertTrue((Path(temporary) / "recovery-replay.json").is_file())
            evidence = json.loads((Path(temporary) / "capture-evidence.json").read_text())
            self.assertEqual(evidence["decision_capture"], "NOT_OBSERVABLE")
        self.assertEqual(fixture.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
