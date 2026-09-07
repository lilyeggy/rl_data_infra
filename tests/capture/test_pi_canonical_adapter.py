from __future__ import annotations

import json
import unittest

from src.assembly.episode_assembler import EpisodeAssembler
from src.capture.pi_adapter import (
    PiJsonAdapter,
    PiOutcomeDeclaration,
    PiRunConfig,
    read_pi_ndjson,
)
from src.capture.pi_canonical_adapter import (
    CanonicalEpisode,
    convert_episode_to_canonical,
)
from src.contracts.agent_episode import (
    EpisodeVerifierStatus,
    ExecutionValidity,
    TaskStatus,
)
from tests.execution_fixtures import make_episode_context

WORKSPACE = "/home/f630/homePLUS/agent-data-plane/swebench/sympy-23824"


def _pi_stream() -> str:
    """A small realistic Pi NDJSON stream: read, grep, bash, edit, finish."""
    lines = [
        {"type": "session", "id": "s-1", "timestamp": 1789000000000, "cwd": WORKSPACE},
        {"type": "agent_start"},
        {"type": "turn_start"},
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "timestamp": 1789000000100,
                "toolCallId": "call-1",
                "tools": ["read", "grep", "bash", "edit", "write", "ls", "find"],
                "content": [
                    {"type": "toolCall", "id": "call-1", "name": "read",
                     "arguments": {"path": "sympy/core/a.py", "limit": 100}}
                ],
            },
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "call-1",
            "toolName": "read",
            "result": {"text": "def foo():\n    return 1\n"},
        },
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "timestamp": 1789000000500,
                "toolCallId": "call-2",
                "content": [
                    {"type": "toolCall", "id": "call-2", "name": "bash",
                     "arguments": {"command": f"cd {WORKSPACE} && python tests/test_a.py"}}
                ],
            },
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "call-2",
            "toolName": "bash",
            "isError": True,
            "result": {"isError": True, "text": f"cd: no such directory {WORKSPACE}/tests"},
        },
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "timestamp": 1789000000900,
                "toolCallId": "call-3",
                "content": [
                    {"type": "toolCall", "id": "call-3", "name": "grep",
                     "arguments": {"pattern": "def foo", "path": WORKSPACE}}
                ],
            },
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "call-3",
            "toolName": "grep",
            "result": {"text": "a.py:1:def foo():"},
        },
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "timestamp": 1789000001500,
                "toolCallId": "call-4",
                "content": [
                    {"type": "toolCall", "id": "call-4", "name": "edit",
                     "arguments": {"path": "sympy/core/a.py", "content": "def foo():\n    return 2\n"}}
                ],
            },
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "call-4",
            "toolName": "edit",
            "result": {"stat": "ok"},
        },
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "timestamp": 1789000002000,
                "stopReason": "stop",
                "content": [{"type": "text", "text": "Done.\n"}],
            },
        },
        {"type": "agent_end", "willRetry": False},
    ]
    return "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n"


class PiCanonicalAdapterTest(unittest.TestCase):
    def _episode(self) -> "object":
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
            run_id="run-swe",
            episode_id="episode-swe-1",
            trace_id="trace-swe-1",
            config=config,
            declared_outcome=declaration,
            source_issues=issues,
        )
        context = make_episode_context(capabilities=result.capabilities)
        assembly = EpisodeAssembler().assemble(
            result.events, contexts={"episode-swe-1": context}
        )
        return assembly.episodes[0]

    def test_convert_real_shaped_stream_to_canonical(self) -> None:
        episode = self._episode()
        canonical, issues = convert_episode_to_canonical(
            episode, workspace_root=WORKSPACE
        )
        self.assertIsInstance(canonical, CanonicalEpisode)
        # 4 tool calls
        self.assertEqual(len(canonical.actions), 4)
        types = [a.action_type.value for a in canonical.actions]
        self.assertEqual(
            types, ["read_file", "run_command", "search_code", "edit_file"]
        )
        # canonical names are harness-neutral
        self.assertEqual(
            [a.canonical_tool_name for a in canonical.actions],
            ["read_file", "run_command", "search_code", "edit_file"],
        )
        # path normalization
        self.assertEqual(canonical.actions[0].normalized_arguments["path"], "sympy/core/a.py")
        self.assertIn("$WORKSPACE", canonical.actions[1].normalized_arguments["command"])
        # grep with workspace root path
        self.assertEqual(canonical.actions[2].normalized_arguments["path"], ".")
        # bash error preserved, no fabrication
        self.assertEqual(canonical.actions[1].result_status.value, "ERROR")
        self.assertNotEqual(canonical.actions[1].observation, "ok")
        # edit succeeded
        self.assertEqual(canonical.actions[3].result_status.value, "SUCCEEDED")
        # strict pairing sanity: every evidence has source events
        for action in canonical.actions:
            self.assertTrue(action.evidence_event_ids)
            # pi tool call id present in raw attributes
            self.assertEqual(action.source_harness, "pi")
        self.assertEqual(canonical.verifier_status, "PASSED")
        self.assertTrue(canonical.checksum)
        # round trip
        restored = CanonicalEpisode.from_dict(canonical.to_dict())
        self.assertEqual(restored, canonical)
        self.assertEqual(restored.checksum, canonical.checksum)

    def test_first_action_not_dropped(self) -> None:
        episode = self._episode()
        canonical, _ = convert_episode_to_canonical(
            episode, workspace_root=WORKSPACE
        )
        self.assertEqual(canonical.actions[0].action_type.value, "read_file")
        self.assertEqual(canonical.actions[0].source_tool_name, "read")

    def test_no_workspace_infers_from_evidence(self) -> None:
        episode = self._episode()
        canonical, _ = convert_episode_to_canonical(episode, workspace_root=None)
        # Workspace root was inferred from the bash command evidence and paths
        # are normalized relative to it.
        self.assertFalse(
            any("/home/" in str(a.normalized_arguments) for a in canonical.actions)
        )
        self.assertTrue(
            canonical.actions[1].canonical_tool_name == "run_command"
        )


if __name__ == "__main__":
    unittest.main()