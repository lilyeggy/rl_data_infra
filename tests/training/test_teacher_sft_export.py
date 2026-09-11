from __future__ import annotations

import unittest

from scripts.export_teacher_sft import _assemble_agent_sft_messages


class TeacherSftExportTest(unittest.TestCase):
    def test_assembler_inserts_canonical_tool_observation_before_next_action(self) -> None:
        messages = [
            {"role": "user", "content": "fix it"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call-read", "type": "function",
                "function": {"name": "read", "arguments": {"path": "solution.py"}},
            }]},
            {"role": "assistant", "content": "done"},
        ]
        steps = [{
            "tool_call_id": "call-read", "tool_name": "read",
            "content": {"path": "solution.py"},
            "result": {"content": [{"type": "text", "text": "TODO"}]},
            "result_status": "SUCCEEDED",
        }]

        assembled = _assemble_agent_sft_messages(messages, steps)

        assert [item["role"] for item in assembled] == ["user", "assistant", "tool", "assistant"]
        assert assembled[2] == {
            "role": "tool", "tool_call_id": "call-read", "name": "read", "content": "TODO",
        }

    def test_assembler_fails_closed_for_unpaired_assistant_tool_call(self) -> None:
        with self.assertRaisesRegex(ValueError, "no canonical result"):
            _assemble_agent_sft_messages(
                [{"role": "assistant", "tool_calls": [{"id": "missing", "function": {"name": "read"}}]}],
                [],
            )


if __name__ == "__main__":
    unittest.main()
