from __future__ import annotations

import unittest

from src.capture.canonical_harness_adapter import (
    CANONICAL_TO_PI_TOOL,
    canonical_to_pi_call,
    parse_canonical_action,
    pi_result_to_observation,
)


class ParseCanonicalActionTest(unittest.TestCase):
    def test_bare_json(self) -> None:
        text = '{"action_type": "read_file", "arguments": {"path": "a.py", "limit": 40}}'
        ok, err, payload = parse_canonical_action(text)
        self.assertTrue(ok, err)
        self.assertEqual(payload["action_type"], "read_file")
        self.assertEqual(payload["arguments"]["path"], "a.py")

    def test_fenced_json(self) -> None:
        text = '```json\n{"action_type": "run_command", "arguments": {"command": "pytest"}}\n```'
        ok, err, payload = parse_canonical_action(text)
        self.assertTrue(ok, err)
        self.assertEqual(payload["action_type"], "run_command")

    def test_tool_call_wrapped(self) -> None:
        text = (
            "<tool_call>\n"
            '{"action_type": "edit_file", "arguments": {"path": "a.py"}}\n'
            "</tool_call>"
        )
        ok, err, _ = parse_canonical_action(text)
        self.assertTrue(ok, err)

    def test_fail_closed_unknown_type(self) -> None:
        ok, err, _ = parse_canonical_action('{"action_type": "teleport", "arguments": {}}')
        self.assertFalse(ok)
        self.assertIn("unknown", err)

    def test_fail_closed_not_json(self) -> None:
        ok, err, _ = parse_canonical_action("please fix the bug now")
        self.assertFalse(ok)

    def test_fail_closed_arguments_not_object(self) -> None:
        ok, err, _ = parse_canonical_action('{"action_type": "read_file", "arguments": "x"}')
        self.assertFalse(ok)


class CanonicalToPiCallTest(unittest.TestCase):
    def test_read_file_to_read(self) -> None:
        call = canonical_to_pi_call(
            {"action_type": "read_file", "arguments": {"path": "a.py", "limit": 10}}
        )
        self.assertEqual(call["name"], "read")
        self.assertEqual(call["arguments"], {"path": "a.py", "limit": 10})

    def test_search_code_to_grep(self) -> None:
        call = canonical_to_pi_call(
            {"action_type": "search_code", "arguments": {"pattern": "def bug", "path": "."}}
        )
        self.assertEqual(call["name"], "grep")
        self.assertEqual(call["arguments"], {"pattern": "def bug", "path": "."})

    def test_run_command_to_bash(self) -> None:
        call = canonical_to_pi_call(
            {"action_type": "run_command", "arguments": {"command": "python -m pytest"}}
        )
        self.assertEqual(call["name"], "bash")
        self.assertEqual(call["arguments"], {"command": "python -m pytest"})

    def test_edit_file_to_edit(self) -> None:
        call = canonical_to_pi_call(
            {"action_type": "edit_file", "arguments": {"path": "a.py", "content": "x"}}
        )
        self.assertEqual(call["name"], "edit")

    def test_fail_closed_unknown(self) -> None:
        with self.assertRaises(ValueError):
            canonical_to_pi_call({"action_type": "transmute", "arguments": {}})

    def test_workspace_placeholder_resolved(self) -> None:
        # ``cd $WORKSPACE && X`` collapses to X since Pi cwd == workspace.
        call = canonical_to_pi_call(
            {
                "action_type": "run_command",
                "arguments": {"command": "cd $WORKSPACE && git status"},
            },
            workspace_root=".",
        )
        self.assertEqual(call["name"], "bash")
        self.assertNotIn("$WORKSPACE", call["arguments"]["command"])
        self.assertEqual(call["arguments"]["command"], "git status")

    def test_workspace_placeholder_bare_token(self) -> None:
        # Model emits ``$WORKSPACE && X`` (no cd) -> should yield X.
        call = canonical_to_pi_call(
            {
                "action_type": "run_command",
                "arguments": {"command": "$WORKSPACE && python -m pytest"},
            },
            workspace_root=".",
        )
        self.assertEqual(call["arguments"]["command"], "python -m pytest")

    def test_workspace_placeholder_path_prefix(self) -> None:
        call = canonical_to_pi_call(
            {
                "action_type": "run_command",
                "arguments": {"command": "cat $WORKSPACE/sympy/x.py"},
            },
            workspace_root=".",
        )
        self.assertEqual(call["arguments"]["command"], "cat sympy/x.py")

    def test_prob_resolution(self) -> None:
        # The mapping is a subset: every Pi-tool-adjacent canonical maps to a Pi tool.
        import src.contracts.canonical_action as ca

        for action_type in ca.CANONICAL_ACTION_TYPES:
            self.assertIn(action_type, CANONICAL_TO_PI_TOOL)


class PiResultToObservationTest(unittest.TestCase):
    def test_wraps(self) -> None:
        obs = pi_result_to_observation("read", {"text": "content"})
        self.assertEqual(obs["tool"], "read")
        self.assertEqual(obs["observation"], {"text": "content"})


if __name__ == "__main__":
    unittest.main()