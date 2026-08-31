from __future__ import annotations

import unittest

from scripts.build_policy_holdout_report import holdout_decision


class PolicyHoldoutDecisionTest(unittest.TestCase):
    def test_accepts_only_newly_resolved_candidate(self) -> None:
        self.assertEqual(
            holdout_decision(baseline_resolved=False, candidate_resolved=True),
            ("ACCEPT", "IMPROVEMENT"),
        )

    def test_rejects_no_change_and_regression(self) -> None:
        self.assertEqual(
            holdout_decision(baseline_resolved=False, candidate_resolved=False),
            ("REJECT", "NO_IMPROVEMENT"),
        )
        self.assertEqual(
            holdout_decision(baseline_resolved=True, candidate_resolved=True),
            ("REJECT", "NO_IMPROVEMENT"),
        )
        self.assertEqual(
            holdout_decision(baseline_resolved=True, candidate_resolved=False),
            ("REJECT", "REGRESSION"),
        )


if __name__ == "__main__":
    unittest.main()
