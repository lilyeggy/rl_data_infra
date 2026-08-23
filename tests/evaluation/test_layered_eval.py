from __future__ import annotations

import unittest

from src.contracts.canonical_action import (
    ActionType,
    ResultStatus,
    build_canonical_action,
)
from src.capture.pi_canonical_adapter import CanonicalEpisode
from src.evaluation import (
    evaluate_action_selection,
    evaluate_format,
    evaluate_trajectory_replay,
)

WORKSPACE = "/home/f630/homePLUS/agent-data-plane/swebench/sympy-23824"


def _episode() -> CanonicalEpisode:
    def action(kind, tool_src, args, status, ts) -> "object":
        return build_canonical_action(
            action_id=f"ca-{tool_src}-{ts}",
            action_type=ActionType(kind),
            canonical_tool_name=kind,
            source_tool_name=tool_src,
            source_harness="pi",
            arguments=args,
            workspace_root=WORKSPACE,
            observation="o",
            result_status=status,
            action_timestamp=f"2026-08-23T00:00:{ts:02d}Z",
            result_timestamp=f"2026-08-23T00:00:{ts + 1:02d}Z",
        )

    return CanonicalEpisode(
        episode_id="ep-1",
        task_id="sympy__sympy-23824",
        source_harness="pi",
        model_id="m",
        verifier_status="PASSED",
        actions=(
            action(
                "read_file", "read", {"path": f"{WORKSPACE}/sympy/core/a.py"},
                ResultStatus.SUCCEEDED, 1,
            ),
            action(
                "run_command", "bash", {"command": f"cd {WORKSPACE} && pytest"},
                ResultStatus.FAILED, 2,
            ),
            action(
                "search_code", "grep", {"pattern": "def bug", "path": WORKSPACE},
                ResultStatus.SUCCEEDED, 3,
            ),
        ),
    )


class FormatEvalTest(unittest.TestCase):
    def test_legal_submissions_pass(self) -> None:
        ep = _episode()
        result = evaluate_format(
            (
                {"name": "read_file", "arguments": {"path": "sympy/core/a.py"}},
                {"name": "run_command", "arguments": {"command": "pytest"}},
                {"type": "function", "function": {"name": "finish"}},
            ),
            ep,
        )
        self.assertEqual(result.submission_count, 3)
        self.assertEqual(result.legal_count, 3)
        self.assertEqual(result.schema_valid_count, 3)
        self.assertEqual(result.adapter_consumable_count, 3)
        self.assertEqual(result.failures, ())

    def test_format_failure_labeled(self) -> None:
        ep = _episode()
        result = evaluate_format(("not-json-at-all", {"name": "teleport", "arguments": {}}), ep)
        self.assertEqual(result.legal_count, 0)
        self.assertTrue(any(f.reason == "FORMAT_FAILURE" for f in result.failures))
        self.assertEqual(len(result.failures), 2)


class ActionSelectionTest(unittest.TestCase):
    def test_correct_tool_and_normalization(self) -> None:
        ep = _episode()
        submissions = (
            {"name": "read_file", "arguments": {"path": "sympy/core/a.py"}},
            {"name": "search_code", "arguments": {"pattern": "def bug"}},
            {"name": "edit_file", "arguments": {}},
        )
        result = evaluate_action_selection(
            episode=ep, history=(), submissions=tuple(submissions)
        )
        # first submission correct tool + exact match
        self.assertEqual(result.correct_tool, 1)
        self.assertEqual(result.exact_match, 1)

    def test_verifier_assisted_recovery(self) -> None:
        # after action[1] (run_command) FAILED, a good model avoids repeating it.
        ep = _episode()
        submissions = (
            # idx0 correct
            {"name": "read_file", "arguments": {"path": "sympy/core/a.py"}},
            # idx1 reference is run_command(FAILED); here the model chooses a
            # different valid tool -> verifier-assisted valid recovery
            {"name": "search_code", "arguments": {"pattern": "def bug"}},
            # idx2 reference search_code succeeded
            {"name": "search_code", "arguments": {"pattern": "x"}},
        )
        result = evaluate_action_selection(
            episode=ep, history=(), submissions=tuple(submissions)
        )


class ReplayTest(unittest.TestCase):
    def test_recovery_and_avoidance(self) -> None:
        ep = _episode()
        # 3 reference actions. Rewind after index 1 (run_command FAILED) and ask
        # the model to continue; a good continuation avoids repeating the failed
        # call and recovers with a different tool.
        result = evaluate_trajectory_replay(
            episode=ep,
            rewound_states=(
                (1, ({"name": "search_code", "arguments": {"pattern": "def bug"}},)),
            ),
        )
        self.assertEqual(result.failure_contexts, 1)
        self.assertEqual(result.recovered_after_failure, 1)
        self.assertEqual(result.avoided_repeat_invalid, 1)


if __name__ == "__main__":
    unittest.main()