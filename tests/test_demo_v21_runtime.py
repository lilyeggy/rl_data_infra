from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from examples.legacy_scenarios.demo_v21_runtime import generate_v21_runtime_evidence


class DemoV21RuntimeTest(unittest.TestCase):
    def test_generates_complete_decision_aware_local_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summary = generate_v21_runtime_evidence(temporary)
            output = Path(temporary)

            self.assertEqual(summary["runtime_type"], "local-recovery-runtime")
            self.assertTrue(summary["not_real_pi"])
            self.assertEqual(summary["episode_count"], 3)
            self.assertEqual(summary["complete_episode_count"], 3)
            self.assertEqual(summary["success_count"], 3)
            self.assertEqual(summary["harness_decision_event_count"], 6)
            self.assertEqual(
                summary["recovery_action_sequences"],
                [["DISCOVER", "READ"]] * 3,
            )

            episodes = [
                json.loads(line)
                for line in (output / "episodes.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(episodes), 3)
            for episode in episodes:
                self.assertEqual(episode["integrity"]["state"], "COMPLETE")
                event_types = [event["event_type"] for event in episode["events"]]
                self.assertEqual(event_types.count("HARNESS_DECISION"), 2)
                self.assertIn("TOOL_CALL", event_types)
                self.assertIn("TOOL_RESULT", event_types)
            self.assertIn(
                "local runtime validates controller/executor/Hook integration",
                summary["claim_boundary"],
            )


if __name__ == "__main__":
    unittest.main()
