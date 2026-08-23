from __future__ import annotations

import json
import unittest

from src.assembly.episode_assembler import EpisodeAssembler
from src.capture.pi_adapter import PiJsonAdapter, PiOutcomeDeclaration, PiRunConfig
from src.contracts.agent_episode import (
    EpisodeVerifierStatus,
    ExecutionValidity,
    TaskStatus,
)
from src.learning import (
    ROLES,
    SFTDatasetReport,
    build_canonical_sft_dataset,
)
from tests.execution_fixtures import make_episode_context

WORKSPACE = "/home/f630/homePLUS/agent-data-plane/swebench/sympy-23824"


def _pi_stream() -> str:
    lines = [
        {"type": "session", "id": "s-1", "timestamp": 1789000000000, "cwd": WORKSPACE},
        {"type": "agent_start"},
        {"type": "turn_start"},
        {
            "type": "message_end",
            "message": {
                "role": "assistant", "timestamp": 1789000000100,
                "toolCallId": "call-1",
                "content": [
                    {"type": "toolCall", "id": "call-1", "name": "read",
                     "arguments": {"path": "sympy/core/a.py"}}
                ],
            },
        },
        {"type": "tool_execution_end", "toolCallId": "call-1", "toolName": "read",
         "result": {"text": "def f:\n"}},
        {
            "type": "message_end",
            "message": {
                "role": "assistant", "timestamp": 1789000000500,
                "toolCallId": "call-2",
                "content": [
                    {"type": "toolCall", "id": "call-2", "name": "bash",
                     "arguments": {"command": f"cd {WORKSPACE} && pytest"}}
                ],
            },
        },
        {"type": "tool_execution_end", "toolCallId": "call-2", "toolName": "bash",
         "isError": True, "result": {"isError": True, "text": "1 failed"}},
        {
            "type": "message_end",
            "message": {
                "role": "assistant", "timestamp": 1789000000900,
                "toolCallId": "call-3",
                "content": [
                    {"type": "toolCall", "id": "call-3", "name": "edit",
                     "arguments": {"path": "sympy/core/a.py", "content": "def f(): return 2"}}
                ],
            },
        },
        {"type": "tool_execution_end", "toolCallId": "call-3", "toolName": "edit",
         "result": {"stat": "ok"}},
        {
            "type": "message_end",
            "message": {
                "role": "assistant", "timestamp": 1789000001200,
                "stopReason": "stop",
                "content": [{"type": "text", "text": "Done\n"}],
            },
        },
        {"type": "agent_end", "willRetry": False},
    ]
    return "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n"


def _episode():
    records, issues = PiJsonAdapter  # placeholder replaced at runtime
    return None


def _build_episode() -> "object":
    from src.capture.pi_adapter import read_pi_ndjson

    records, issues = read_pi_ndjson(_pi_stream())
    config = PiRunConfig(model="deepseek-v4-flash")
    declaration = PiOutcomeDeclaration(
        task_status=TaskStatus.SUCCESS,
        execution_validity=ExecutionValidity.VALID,
        verifier_status=EpisodeVerifierStatus.PASSED,
        score=1.0,
    )
    result = PiJsonAdapter().convert(
        records,
        run_id="run-1", episode_id="episode-1", trace_id="trace-1",
        config=config, declared_outcome=declaration, source_issues=issues,
    )
    context = make_episode_context(capabilities=result.capabilities)
    assembly = EpisodeAssembler().assemble(result.events, contexts={"episode-1": context})
    return assembly.episodes[0]


class CanonicalSFTDatasetTest(unittest.TestCase):
    def test_builds_role_examples_with_stats_and_no_leak(self) -> None:
        episode = _build_episode()
        examples, report = build_canonical_sft_dataset(
            (episode,),
            task_ids={"episode-1": "sympy__sympy-23824"},
            workspace_roots={"episode-1": WORKSPACE},
        )
        self.assertIsInstance(report, SFTDatasetReport)
        self.assertEqual(report.episodes, 1)
        self.assertEqual(report.total_examples, 3)  # 3 tool calls
        self.assertEqual(report.leak_count, 0)
        roles = {ex.role for ex in examples}
        self.assertIn("first-action", roles)
        self.assertIn("tool-selection", roles)  # bash
        # bash (index1) failed; index2 is last -> final-answer. Recovery needs a
        # non-final action after a failure, covered in the 4-action test below.
        self.assertIn("final-answer", roles)
        self.assertTrue(all(ex.role in ROLES for ex in examples))
        self.assertTrue(all(ex.split == "TRAIN" for ex in examples))
        # no Pi-private marker
        for ex in examples:
            self.assertEqual(ex.system_prompt_marker, "<neutral-agent-instructions>")
        # canonical tool names only
        for ex in examples:
            self.assertIn(ex.action["tool"], {
                "read_file", "search_code", "list_directory", "run_command",
                "edit_file", "write_file", "finish", "tool_error",
                "environment_observation",
            })
        # checksum present
        self.assertTrue(report.checksum)

    def test_all_examples_serializable(self) -> None:
        episode = _build_episode()
        examples, report = build_canonical_sft_dataset(
            (episode,),
            task_ids={"episode-1": "t"},
            workspace_roots={"episode-1": WORKSPACE},
        )
        for ex in examples:
            self.assertTrue(ex.checksum)
            d = ex.to_dict()
            self.assertEqual(d["episode_id"], "episode-1")

    def test_failure_recovery_assigned_when_action_follows_failure(self) -> None:
        # Build a canonical episode directly: bash(fail) then read -> read is
        # failure-recovery (non-final), then edit(last) final-answer.
        from src.capture.pi_canonical_adapter import CanonicalEpisode
        from src.contracts.canonical_action import (
            ActionType,
            ResultStatus,
            build_canonical_action,
        )

        def act(kind, src, args, status, ts):
            return build_canonical_action(
                action_id=f"ca-{src}-{ts}",
                action_type=ActionType(kind), canonical_tool_name=kind,
                source_tool_name=src, source_harness="pi", arguments=args,
                workspace_root=WORKSPACE, observation="o", result_status=status,
                action_timestamp=f"2026-08-23T00:00:{ts:02d}Z",
            )

        canonical = CanonicalEpisode(
            episode_id="ep-r", task_id="sympy__sympy-23824", source_harness="pi",
            model_id="m", verifier_status="PASSED",
            actions=(
                act("read_file", "read", {"path": "a.py"}, ResultStatus.SUCCEEDED, 1),
                act("run_command", "bash", {"command": "pytest"}, ResultStatus.FAILED, 2),
                act("search_code", "grep", {"pattern": "bug"}, ResultStatus.SUCCEEDED, 3),
                act("edit_file", "edit", {"path": "a.py"}, ResultStatus.SUCCEEDED, 4),
            ),
        )
        # Wrap a lightweight module-level hook: build report from canonical via
        # the same role classifier function.
        from src.learning.canonical_sft_dataset import _role_of

        self.assertEqual(
            _role_of(canonical.actions[1], 1, 4, prev_failed=False), "tool-selection"
        )
        self.assertEqual(
            _role_of(canonical.actions[2], 2, 4, prev_failed=True), "failure-recovery"
        )
        self.assertEqual(
            _role_of(canonical.actions[3], 3, 4, prev_failed=False), "final-answer"
        )


if __name__ == "__main__":
    unittest.main()