from __future__ import annotations

import tempfile
import unittest

from examples.legacy_scenarios.demo_v24 import generate_v24_offline_training_view


class DemoV24Test(unittest.TestCase):
    def test_generates_offline_training_view_from_live_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summary = generate_v24_offline_training_view(temporary)

            self.assertEqual(summary["release"], "v2.4-offline-training-view")
            self.assertEqual(summary["episode_count"], 6)
            self.assertEqual(summary["rollout_count"], 6)
            self.assertEqual(summary["sft_eligible_rollouts"], 3)
            self.assertEqual(summary["on_policy_rl_eligible"], 0)
            self.assertIn("target-policy sampled token ids", summary["missing_for_on_policy"])
            # control/candidate arms differ in harness policy (tool schema): the
            # legacy "same task and model" DPO claim is rejected fail-closed.
            self.assertEqual(summary["preference_pair_count"], 0)
            self.assertEqual(len(summary["preference_pair_rejections"]), 3)
            for rejection in summary["preference_pair_rejections"]:
                self.assertIn("tool_schema", rejection["identity_mismatch"])
            # verifier certification output present
            self.assertTrue(
                (__import__("pathlib").Path(temporary) / "episode-certifications.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
