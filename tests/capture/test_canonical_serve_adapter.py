from __future__ import annotations

import unittest

from src.capture.canonical_serve_adapter import extract_tool_calls


class ExtractToolCallsTest(unittest.TestCase):
    def test_canonical_action_translated_to_pi(self) -> None:
        calls, diag = extract_tool_calls(
            '{"action_type": "read_file", "arguments": {"path": "a.py", "limit": 40}}'
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "read")
        self.assertEqual(calls[0]["arguments"], {"path": "a.py", "limit": 40})

    def test_pi_native_tool_call_passed_through(self) -> None:
        calls, diag = extract_tool_calls(
            '<tool_call>\n{"name": "grep", "arguments": {"pattern": "def"}}\n</tool_call>'
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "grep")

    def test_bookended_prose_around_canonical(self) -> None:
        calls, diag = extract_tool_calls(
            'Let me inspect.\n'
            '{"action_type": "search_code", "arguments": {"pattern": "bug", "path": "."}}\n'
        )
        self.assertEqual(calls[0]["name"], "grep")

    def test_no_tool_call_returns_empty_with_diagnostic(self) -> None:
        calls, diag = extract_tool_calls("I will now fix the bug directly.")
        self.assertEqual(calls, [])
        self.assertTrue(diag)

    def test_canonical_unknown_tool_returns_empty(self) -> None:
        calls, diag = extract_tool_calls('{"action_type": "teleport", "arguments": {}}')
        self.assertEqual(calls, [])
        self.assertTrue(diag)


if __name__ == "__main__":
    unittest.main()