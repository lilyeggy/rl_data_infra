from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from examples.legacy_scenarios.demo_v22 import generate_v22_multi_agent_evidence


class DemoV22Test(unittest.TestCase):
    def test_generates_explicit_multi_agent_graph_and_training_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summary = generate_v22_multi_agent_evidence(temporary)
            output = Path(temporary)

            self.assertTrue(summary["synthetic_fixture"])
            self.assertEqual(summary["graph_status"], "OBSERVED")
            self.assertEqual(summary["agent_count"], 2)
            self.assertEqual(summary["sft_candidates"], 2)
            self.assertEqual(summary["on_policy_rl_candidates"], 0)
            self.assertTrue((output / "multi-agent-graph.json").is_file())
            self.assertTrue((output / "training-views.json").is_file())

            graph = json.loads((output / "multi-agent-graph.json").read_text())
            self.assertEqual(graph["graph_status"], "OBSERVED")
            self.assertEqual(len(graph["agents"]), 2)
            self.assertTrue(
                any(
                    relation["relation_type"] == "REWARD_OWNERSHIP"
                    for relation in graph["relations"]
                )
            )


if __name__ == "__main__":
    unittest.main()
