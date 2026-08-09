from __future__ import annotations

import unittest

from scripts.select_swebench_candidates import rank_candidates, select_candidates


def row(instance_id: str, repo: str, fail: int, passed: int, chars: int = 100):
    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": "a" * 40,
        "problem_statement": "x" * chars,
        "FAIL_TO_PASS": [f"fail-{index}" for index in range(fail)],
        "PASS_TO_PASS": [f"pass-{index}" for index in range(passed)],
    }


class SelectSwebenchCandidatesTest(unittest.TestCase):
    def test_ranks_fewer_tests_then_shorter_prompt(self):
        rows = [
            row("owner__large-1", "owner/large", 2, 5, 20),
            row("owner__short-1", "owner/short", 1, 1, 200),
            row("owner__shorter-1", "owner/shorter", 1, 1, 100),
        ]

        ranked = rank_candidates(rows)

        self.assertEqual(
            [item["instance_id"] for item in ranked],
            ["owner__shorter-1", "owner__short-1", "owner__large-1"],
        )

    def test_skips_rows_without_reproducible_identity_or_tests(self):
        valid = row("owner__valid-1", "owner/repo", 1, 0)
        invalid_sha = {**valid, "instance_id": "bad-sha", "base_commit": "abc"}
        no_failing_test = {**valid, "instance_id": "no-fail", "FAIL_TO_PASS": []}

        ranked = rank_candidates([invalid_sha, no_failing_test, valid])

        self.assertEqual([item["instance_id"] for item in ranked], ["owner__valid-1"])

    def test_can_select_distinct_repositories(self):
        rows = [
            row("a__one-1", "a/repo", 1, 0, 10),
            row("a__two-1", "a/repo", 1, 0, 20),
            row("b__one-1", "b/repo", 1, 0, 30),
        ]

        selected = select_candidates(rows, 2, distinct_repos=True)

        self.assertEqual([item["instance_id"] for item in selected], ["a__one-1", "b__one-1"])

    def test_rejects_non_positive_limit(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            select_candidates([], 0)


if __name__ == "__main__":
    unittest.main()
