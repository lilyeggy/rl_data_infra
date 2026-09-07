from __future__ import annotations

import unittest

from src.orchestration import RolloutPoolMetrics, RolloutPoolStage


class RolloutPoolMetricsTest(unittest.TestCase):
    def test_stage_gauges_and_terminal_timings_match_pool_lifecycle(self) -> None:
        times = iter((0.0, 2.0, 5.0, 11.0, 14.0))
        metrics = RolloutPoolMetrics(clock=lambda: next(times))
        metrics.register("episode-1")
        self.assertEqual(metrics.snapshot().queued_depth, 1)
        metrics.transition("episode-1", RolloutPoolStage.INIT)
        self.assertEqual(metrics.snapshot().init_inflight, 1)
        metrics.transition("episode-1", RolloutPoolStage.RUN)
        self.assertEqual(metrics.snapshot().run_inflight, 1)
        metrics.transition("episode-1", RolloutPoolStage.POSTRUN)
        self.assertEqual(metrics.snapshot().postrun_inflight, 1)
        metrics.finish("episode-1", status="COMPLETED")

        payload = metrics.to_dict()
        self.assertEqual(payload["snapshot"]["completed_total"], 1)
        self.assertEqual(payload["snapshot"]["failed_total"], 0)
        episode = payload["episodes"][0]
        self.assertEqual(episode["queue_ms"], 2000.0)
        self.assertEqual(episode["init_ms"], 3000.0)
        self.assertEqual(episode["run_ms"], 6000.0)
        self.assertEqual(episode["postrun_ms"], 3000.0)

    def test_stage_regression_and_duplicate_registration_are_rejected(self) -> None:
        metrics = RolloutPoolMetrics(clock=lambda: 0.0)
        metrics.register("episode-1")
        with self.assertRaises(ValueError):
            metrics.register("episode-1")
        metrics.transition("episode-1", RolloutPoolStage.INIT)
        with self.assertRaises(ValueError):
            metrics.transition("episode-1", RolloutPoolStage.QUEUED)


if __name__ == "__main__":
    unittest.main()
