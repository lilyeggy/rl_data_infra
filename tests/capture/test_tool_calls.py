"""Tests for tool-call recovery from a policy's generated text.

The trained policy emits bare tool-call JSON that stock vLLM parsers do not
match, so the data plane has to recover the calls the way the stage D/E rollout
servers did. These cases pin that behaviour: accept declared tools, refuse
anything fabricated.
"""

from __future__ import annotations

import json
import unittest

from src.capture.tool_calls import extract_tool_calls

TOOLS = [
    {"type": "function", "function": {"name": "read", "parameters": {}}},
    {"type": "function", "function": {"name": "bash", "parameters": {}}},
]


class ExtractToolCallsTest(unittest.TestCase):
    def test_recovers_the_bare_json_form_the_policy_emits(self) -> None:
        text = (
            "Let's read it.\n"
            '{"name": "read", "arguments": {"path": "solution.py"}}\n'
            "and then run it\n"
            '{"name": "bash", "arguments": {"command": "python solution.py"}}\n'
        )
        calls = extract_tool_calls(text, TOOLS)
        self.assertEqual([c["function"]["name"] for c in calls], ["read", "bash"])
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"path": "solution.py"}
        )

    def test_recovers_the_tagged_form_too(self) -> None:
        text = '<tool_call>{"name": "read", "arguments": {"path": "a.py"}}</tool_call>'
        calls = extract_tool_calls(text, TOOLS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["type"], "function")

    def test_nested_arguments_survive(self) -> None:
        # A regex cannot span the nested object; the brace scan must.
        text = '{"name": "read", "arguments": {"path": "a.py", "meta": {"x": {"y": 1}}}}'
        calls = extract_tool_calls(text, TOOLS)
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"])["meta"], {"x": {"y": 1}}
        )

    def test_undeclared_or_malformed_calls_are_refused(self) -> None:
        cases = (
            '{"name": "rm_rf", "arguments": {"path": "/"}}',  # undeclared tool
            '{"name": "read", "arguments": "solution.py"}',  # arguments not an object
            '{"name": "read", "arguments": {}}',  # empty arguments
            '{"name": "read", "arguments": {"path":',  # malformed json
            '{"arguments": {"path": "a.py"}}',  # no name
            '["read"]',  # not an object
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(extract_tool_calls(text, TOOLS), [])

    def test_no_tools_declared_means_no_calls(self) -> None:
        text = '{"name": "read", "arguments": {"path": "a.py"}}'
        self.assertEqual(extract_tool_calls(text, []), [])
        self.assertEqual(extract_tool_calls(text, None), [])
        self.assertEqual(extract_tool_calls("", TOOLS), [])

    def test_accepts_the_tuple_the_proxy_records(self) -> None:
        # The request evidence stores tools as a tuple, so a list-only check
        # silently disabled recovery on every framework-path rollout.
        text = '{"name": "read", "arguments": {"path": "a.py"}}'
        self.assertEqual(len(extract_tool_calls(text, tuple(TOOLS))), 1)

    def test_narration_between_calls_is_ignored(self) -> None:
        # The policy narrates a plan and then prints the calls; only the calls
        # are taken.
        text = (
            "Using the instructions given, I will use the read tool.\n"
            "Please find the tool calls below:\n\n"
            "1. Read solution.py:\n"
            '   {"name": "read", "arguments": {"path": "solution.py"}}\n\n'
            "2. Test it:\n"
            '   {"name": "bash", "arguments": {"command": "python solution.py"}}\n'
        )
        calls = extract_tool_calls(text, TOOLS)
        self.assertEqual([c["function"]["name"] for c in calls], ["read", "bash"])


if __name__ == "__main__":
    unittest.main()
