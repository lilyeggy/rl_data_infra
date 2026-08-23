from __future__ import annotations

import unittest

from src.contracts._json import sha256_json
from src.contracts.canonical_action import (
    ActionType,
    CanonicalAction,
    ResultStatus,
    build_canonical_action,
)
from src.capture.pi_canonical_adapter import CanonicalEpisode
from src.exporters.canonical import (
    assert_training_view_is_leak_free,
    export_generic_agent_sft,
    export_harness_improvement,
    export_model_native,
)

WORKSPACE = "/home/f630/homePLUS/agent-data-plane/swebench/sympy-23824"


def _episode() -> CanonicalEpisode:
    def action(kind, tool, args, status, ts, obs=None) -> CanonicalAction:
        return build_canonical_action(
            action_id=f"ca-{tool}-{ts}",
            action_type=ActionType(kind),
            canonical_tool_name=kind,
            source_tool_name=({ "read_file": "read", "run_command": "bash", "search_code": "grep", "edit_file": "edit" }).get(kind),
            source_harness="pi",
            arguments=args,
            workspace_root=WORKSPACE,
            observation=obs,
            result_status=status,
            action_timestamp=f"2026-08-23T00:00:{ts:02d}Z",
            result_timestamp=f"2026-08-23T00:00:{ts + 1:02d}Z",
        )

    actions = [
        action(
            "read_file",
            "read_file",
            {"path": f"{WORKSPACE}/sympy/core/a.py", "limit": 40},
            ResultStatus.SUCCEEDED,
            1,
            obs="line content",
        ),
        action(
            "run_command",
            "run_command",
            {"command": f"cd {WORKSPACE} && python -m pytest tests/test_a.py"},
            ResultStatus.FAILED,
            2,
            obs="1 failed",
        ),
        action(
            "search_code",
            "search_code",
            {"pattern": "def bug", "path": WORKSPACE},
            ResultStatus.SUCCEEDED,
            3,
            obs="a.py:3",
        ),
        action(
            "edit_file",
            "edit_file",
            {"path": "sympy/core/a.py", "content": "def bug():\n    return 2\n"},
            ResultStatus.SUCCEEDED,
            4,
            obs="ok",
        ),
    ]
    return CanonicalEpisode(
        episode_id="episode-swe-1",
        task_id="sympy__sympy-23824",
        source_harness="pi",
        model_id="deepseek-v4-flash",
        verifier_status="PASSED",
        actions=tuple(actions),
    )


class GenericSFTExporterTest(unittest.TestCase):
    def test_exports_neutral_examples_with_types_and_no_leak(self) -> None:
        ep = _episode()
        examples, manifest = export_generic_agent_sft(
            ep, task="Fix bug in sympy/core/a.py"
        )
        # one example per action
        self.assertEqual(len(examples), len(ep.actions))
        # expected example types: first-action, failure-recovery, tool-selection, final-answer
        types = [ex.example_type for ex in examples]
        self.assertEqual(types[0], "first-action")
        self.assertEqual(types[2], "failure-recovery")  # search_code follows failed run_command
        self.assertEqual(types[3], "final-answer")
        self.assertEqual(types[1], "tool-selection")  # the failing action itself
        # normalized path :: no host leak
        for ex in examples:
            assert_training_view_is_leak_free(ex.to_dict(), label="generic-sft")
        # checksums present and deterministic
        self.assertTrue(manifest["checksum"])
        self.assertEqual(
            manifest["checksum"],
            sha256_json([ex.to_dict() for ex in examples]),
        )
        # lineage recorded
        self.assertEqual(manifest["episode_id"], ep.episode_id)


class ModelNativeExporterTest(unittest.TestCase):
    def test_three_renderers_produced_and_round_trip_tool_content(self) -> None:
        ep = _episode()
        examples, manifest = export_model_native(ep, task="Fix bug in file")
        self.assertEqual(len(examples), 3)
        renderers = sorted(ex.format for ex in examples)
        self.assertEqual(
            renderers,
            ["model-native-json", "model-native-openai", "model-native-qwen"],
        )
        # leak-free
        for example in examples:
            assert_training_view_is_leak_free(example.to_dict(), label="native")
        # qwen renderer has tool_calls on assistant messages
        qwen = next(ex for ex in examples if ex.renderer == "qwen")
        assistant = [m for m in qwen.messages if m["role"] == "assistant"][0]
        self.assertIn("tool_calls", assistant)
        self.assertEqual(assistant["tool_calls"][0]["name"], "read_file")
        # openai renderer
        openai = next(ex for ex in examples if ex.renderer == "openai")
        assistant_o = [m for m in openai.messages if m["role"] == "assistant"][0]
        self.assertEqual(assistant_o["tool_calls"][0]["type"], "function")
        # json renderer
        jsonview = next(ex for ex in examples if ex.renderer == "json")
        assistant_j = [m for m in jsonview.messages if m["role"] == "assistant"][0]
        self.assertEqual(assistant_j["tool_calls"][0]["action_type"], "read_file")
        # canonical episode never holds model-specific tokens
        for action in ep.actions:
            text = str(action.to_dict())
            self.assertNotIn("<tool_call>", text)

    def test_unknown_renderer_rejected(self) -> None:
        with self.assertRaises(ValueError):
            export_model_native(_episode(), task="x", renderers=("weird",))


class HarnessImprovementExporterTest(unittest.TestCase):
    def test_metrics_reasonably_computed(self) -> None:
        ep = _episode()
        metrics = export_harness_improvement(ep)
        self.assertEqual(metrics.episode_id, ep.episode_id)
        self.assertEqual(metrics.tool_call_count, 4)
        # one repeated search/read is not present; failures = 1
        self.assertEqual(metrics.repeated_failure_count, 1)
        self.assertIsNotNone(metrics.first_action_latency_ms)
        self.assertIsNotNone(metrics.time_to_patch_ms)
        self.assertTrue(metrics.checksum)


class LeakGuardTest(unittest.TestCase):
    def test_host_path_raises(self) -> None:
        with self.assertRaises(ValueError):
            assert_training_view_is_leak_free(
                {"path": "/home/f630/secret/x.py"}, label="t"
            )

    def test_workspace_placeholder_allowed(self) -> None:
        # $WORKSPACE token is neutralized and allowed
        assert_training_view_is_leak_free(
            {"command": "$WORKSPACE && git status"}, label="t"
        )


if __name__ == "__main__":
    unittest.main()