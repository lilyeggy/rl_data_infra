from __future__ import annotations

import unittest

from src.analysis.compare import AggregateComparison, ComparisonReport
from src.analysis.regression_gate import GateDecision, GateResult
from src.observatory.report import render_observatory_html


class ObservatoryTest(unittest.TestCase):
    def test_renders_read_only_evidence_shell(self) -> None:
        aggregate = AggregateComparison(
            control_success_rate=0.0,
            candidate_success_rate=1.0,
            success_rate_delta=1.0,
            control_infra_invalid_rate=0.0,
            candidate_infra_invalid_rate=0.0,
            infra_invalid_rate_delta=0.0,
            control_mean_tokens=10.0,
            candidate_mean_tokens=9.0,
            token_increase_ratio=-0.1,
            control_mean_duration_ms=10.0,
            candidate_mean_duration_ms=9.0,
            latency_increase_ratio=-0.1,
            control_target_slice_count=1,
            candidate_target_slice_count=0,
        )
        comparison = ComparisonReport(
            experiment_id="exp",
            experiment_checksum="0" * 64,
            pairs=(),
            unmatched_control_episode_ids=(),
            unmatched_candidate_episode_ids=(),
            compatibility_mismatches=(),
            paired_coverage=1.0,
            aggregate=aggregate,
            input_episode_checksums=(),
        )
        gate = GateResult(
            decision=GateDecision.ACCEPT,
            checks=(),
            comparison_checksum=comparison.checksum,
            config_checksum="1" * 64,
        )
        output = render_observatory_html(
            episodes=(), metrics=(), diagnoses=(), comparison=comparison, gate=gate
        )
        self.assertIn("Harness Observatory", output)
        self.assertIn("Episode Explorer", output)
        self.assertIn("Regression Gate", output)
        self.assertIn("ACCEPT", output)
        self.assertNotIn("fetch(", output)


if __name__ == "__main__":
    unittest.main()
