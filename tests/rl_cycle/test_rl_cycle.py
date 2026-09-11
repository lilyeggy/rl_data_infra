from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.collect_rl_rollouts import collect_group_rollouts
from scripts.train_grpo_lora import compute_group_advantages, run_grpo_preflight, mock_grpo_train
from scripts.orchestrate_rl_cycle import RLCycleOrchestrator


class TestRLCycle(unittest.TestCase):
    def test_group_rollout_collection_and_admission(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            summary = collect_group_rollouts(
                tasks=["task-alpha", "task-beta"],
                group_size=4,
                output_dir=out_dir,
                run_id="run-test-unit",
                policy_fingerprint="0" * 64,
                mock=True,
            )
            self.assertEqual(summary["tasks_count"], 2)
            self.assertEqual(summary["total_rollouts"], 8)
            self.assertEqual(summary["eligible_trajectories"], 8)
            self.assertEqual(summary["admitted_traces"], 8)

            admission_file = out_dir / "slime-admission.json"
            self.assertTrue(admission_file.exists())
            data = json.loads(admission_file.read_text())
            self.assertEqual(data["schema_version"], "slime-admission/v1")
            self.assertEqual(len(data["traces"]), 8)

    def test_grpo_advantage_computation(self):
        traces = [
            {"group_id": "g1", "trajectory_id": "t1", "reward": 1.0},
            {"group_id": "g1", "trajectory_id": "t2", "reward": 0.0},
            {"group_id": "g2", "trajectory_id": "t3", "reward": 1.0},
            {"group_id": "g2", "trajectory_id": "t4", "reward": 1.0},
        ]
        stats = compute_group_advantages(traces)
        # g1 has variance: mean 0.5, std 0.5
        mean_g1, std_g1, advs_g1 = stats["g1"]
        self.assertAlmostEqual(mean_g1, 0.5)
        self.assertGreater(advs_g1["t1"], 0.0)
        self.assertLess(advs_g1["t2"], 0.0)

        # g2 has no variance (both 1.0): advantage must be 0.0
        mean_g2, std_g2, advs_g2 = stats["g2"]
        self.assertAlmostEqual(mean_g2, 1.0)
        self.assertEqual(advs_g2["t3"], 0.0)
        self.assertEqual(advs_g2["t4"], 0.0)

    def test_full_orchestration_cycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            orchestrator = RLCycleOrchestrator(
                cycle_id="unit-cycle",
                tasks=["t1", "t2"],
                eval_tasks=["eval-1"],
                group_size=4,
                output_dir=out_dir,
                mock=True,
            )
            result = orchestrator.run_full_cycle()
            self.assertEqual(result["status"], "COMPLETED")
            self.assertEqual(result["gate"]["verdict"], "PROMOTED")

            # Check files created
            self.assertTrue((out_dir / "cycle-summary.json").exists())
            self.assertTrue((out_dir / "gate-report.json").exists())
            self.assertTrue((out_dir / "candidate_policy_adapter" / "adapter_config.json").exists())


if __name__ == "__main__":
    unittest.main()
