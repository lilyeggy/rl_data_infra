"""Contracts for the native token transport, driven by real run artifacts.

The fixtures under ``tests/fixtures/verl_pi/`` are captured verbatim from the
smoke16 GRPO run (``verlpi-smoke16-Mbpp-118-a0s0``):

* ``smoke16-a0s0-call0-generation.txt`` -- the policy's own first generation;
* ``smoke16-a0s0-pi-echo-assistant.json`` -- the assistant message Pi sent back
  on the following request.

The previous round of tests used invented input, so they passed while the real
harness could not have. These use the measured shapes.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path

from src.errors import ContractValidationError
from src.integrations.verl.native_transport import NativeTokenTransport, _tool_calls

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "verl_pi"
GENERATION_TEXT = (FIXTURES / "smoke16-a0s0-call0-generation.txt").read_text()
PI_ECHO = json.loads((FIXTURES / "smoke16-a0s0-pi-echo-assistant.json").read_text())

# The tool schema that episode actually ran with.
DECLARED_TOOLS = [
    {"type": "function", "function": {"name": name}}
    for name in ("bash", "edit", "ls", "read", "write")
]

END = 999
_BASE = 1000


class ReversibleTokenizer:
    """Character-level stand-in whose encode/decode round-trips exactly.

    The real tokenizer is unavailable on CPU, but the property under test --
    ``decode`` feeding the extractor while ``encode`` only ever adds *new* text --
    needs precisely that round-trip, so a lossy double would hide the bug class
    this module exists to catch.
    """

    eos_token = "<|im_end|>"
    eos_token_id = END

    def convert_tokens_to_ids(self, token: str) -> int:
        return END

    def decode(self, tokens) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return "".join("<|im_end|>" if t == END else chr(t - _BASE) for t in tokens)

    def encode(self, text: str, **_: object) -> list[int]:
        return [_BASE + ord(char) for char in text]

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, **kwargs
    ):
        parts = []
        for message in messages:
            rendered = "".join(
                f"<tool_call>{call['function']['name']}"
                f"{call['function']['arguments']}</tool_call>"
                for call in message.get("tool_calls") or []
            )
            content = message.get("content") or ""
            parts.append(
                f"<|im_start|>{message['role']}\n{content}{rendered}<|im_end|>\n"
            )
        text = "".join(parts)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return self.encode(text) if tokenize else text


class ParserDouble:
    """Stands in for verl's Hermes parser, calibrated to a real measurement.

    ``ToolParser.get_tool_parser("hermes", ...)`` was run over the ten real
    generations of the smoke16 run and returned zero calls from every one of
    them, because the policy emits bare JSON and the parser matches
    ``<tool_call>`` tags only. The double reproduces that measured behaviour.
    """

    def __init__(self, calls=(), content=""):
        self.calls = list(calls)
        self.content = content

    async def extract_tool_calls(self, token_ids):
        return self.content, list(self.calls)


class Call:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class TokenOutput:
    def __init__(self, token_ids, global_steps=0):
        self.token_ids = list(token_ids)
        self.log_probs = [-0.5] * len(self.token_ids)
        self.extra_fields = {"global_steps": global_steps}


class ServerManagerDouble:
    def __init__(self, token_ids):
        self.token_ids = list(token_ids)
        self.prompts: list[list[int]] = []

    async def generate(self, *, request_id, prompt_ids, sampling_params):
        self.prompts.append(list(prompt_ids))
        return TokenOutput(self.token_ids)


class ToolCallExtractionTest(unittest.TestCase):
    def test_real_generation_contains_no_tool_call_tags(self) -> None:
        """Documents the measured fact the extraction fallback exists for."""
        self.assertGreater(len(GENERATION_TEXT), 3000)
        self.assertEqual(GENERATION_TEXT.count("<tool_call>"), 0)
        self.assertEqual(GENERATION_TEXT.count("</tool_call>"), 0)

    def test_real_generation_yields_the_call_pi_actually_ran(self) -> None:
        # Pi executed exactly one read of solution.py on this episode; the second
        # (edit) call was cut off mid-JSON by the 1024 budget and must not be
        # invented.
        calls = _tool_calls((), GENERATION_TEXT, DECLARED_TOOLS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"path": "solution.py"}
        )
        self.assertEqual(calls[0]["type"], "function")

    def test_undeclared_tool_is_never_invented(self) -> None:
        text = '{"name": "rm", "arguments": {"path": "/"}}'
        self.assertEqual(_tool_calls((), text, DECLARED_TOOLS), [])

    def test_malformed_json_is_not_repaired(self) -> None:
        text = '{"name": "read", "arguments": {"path": "solution.py"'
        self.assertEqual(_tool_calls((), text, DECLARED_TOOLS), [])

    def test_tagged_generation_still_uses_the_upstream_parser(self) -> None:
        text = "<tool_call>{}</tool_call>"
        calls = _tool_calls(
            [Call("read", '{"path":"solution.py"}')], text, DECLARED_TOOLS
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read")

    def test_tagged_but_unparsed_is_rejected(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "malformed"):
            _tool_calls((), "<tool_call>{}</tool_call>", DECLARED_TOOLS)

    def test_partially_parsed_generation_is_rejected(self) -> None:
        text = "<tool_call>{}</tool_call><tool_call>{}</tool_call>"
        with self.assertRaisesRegex(ContractValidationError, "no repair"):
            _tool_calls([Call("read", "{}")], text, DECLARED_TOOLS)


class TransportAgainstRealArtifactsTest(unittest.TestCase):
    """Walks the transport with the real generation and Pi's real echo."""

    def setUp(self) -> None:
        self.tokenizer = ReversibleTokenizer()
        self.generation = self.tokenizer.encode(GENERATION_TEXT) + [END]
        self.server = ServerManagerDouble(self.generation)
        # The upstream parser sees the same untagged text it saw on the run.
        self.parser = ParserDouble(calls=(), content=GENERATION_TEXT)
        self.transport = NativeTokenTransport(
            loop=None,
            server_manager=self.server,
            tokenizer=self.tokenizer,
            parser=self.parser,
            episode_id="episode",
            policy_revision="f" * 64,
            sampling_params={},
            max_prompt=8000,
            max_response=40000,
            max_tokens=3584,
            max_requests=4,
            timeout=5.0,
            expected_engine_step=0,
        )
        self.first_request = {
            "model": "m",
            "messages": [
                {"role": "system", "content": "you are pi"},
                {"role": "user", "content": "implement Mbpp/118"},
            ],
            "tools": DECLARED_TOOLS,
        }

    def _follow_up(self, issued_id: str) -> dict:
        """Pi's next request: real echo shape, with the tool result appended."""
        echo = dict(
            PI_ECHO,
            tool_calls=[dict(PI_ECHO["tool_calls"][0], id=issued_id)],
        )
        return {
            "model": "m",
            "messages": [
                *self.first_request["messages"],
                echo,
                {
                    "role": "tool",
                    "tool_call_id": issued_id,
                    "content": '"""stub"""\n',
                },
            ],
            "tools": DECLARED_TOOLS,
        }

    def test_first_turn_reaches_pi_with_a_runnable_tool_call(self) -> None:
        response = asyncio.run(self.transport.generate(self.first_request))
        message = response["choices"][0]["message"]
        calls = message.get("tool_calls")
        self.assertTrue(calls, "the episode cannot advance without a tool call")
        self.assertEqual(calls[0]["function"]["name"], "read")
        self.assertEqual(response["choices"][0]["finish_reason"], "tool_calls")

    def test_native_generation_survives_verbatim_in_the_ledger(self) -> None:
        asyncio.run(self.transport.generate(self.first_request))
        context = self.transport.context
        ledger = list(context.tokens)
        start = context.initial_length
        self.assertEqual(
            ledger[start : start + len(self.generation)], self.generation
        )

    def test_real_pi_echo_is_accepted_and_becomes_the_second_turn(self) -> None:
        """The decisive multi-turn case: Pi echoes content: null and we accept."""
        first = asyncio.run(self.transport.generate(self.first_request))
        issued = first["choices"][0]["message"]["tool_calls"][0]["id"]
        asyncio.run(self.transport.generate(self._follow_up(issued)))

        context = self.transport.context
        ledger = list(context.tokens)
        # The first generation is still there, verbatim, followed by the new
        # observation -- prompt_{i+1} = prompt_i + generation_i + observation.
        start = context.initial_length
        self.assertEqual(
            ledger[start : start + len(self.generation)], self.generation
        )
        self.assertGreater(len(ledger), start + len(self.generation))
        # Turn two was prompted with exactly the ledger as it stood then, so the
        # strict contiguity the assembler requires holds by construction: the
        # ledger is prompt_1 + generation_1 + observation_1, plus whatever turn
        # two added on top.
        second_prompt = self.server.prompts[1]
        self.assertEqual(second_prompt, ledger[: len(second_prompt)])
        self.assertGreater(len(second_prompt), start + len(self.generation))

    def test_rewriting_a_prior_turn_is_rejected(self) -> None:
        first = asyncio.run(self.transport.generate(self.first_request))
        issued = first["choices"][0]["message"]["tool_calls"][0]["id"]
        request = self._follow_up(issued)
        request["messages"][1] = {"role": "user", "content": "compacted"}
        with self.assertRaisesRegex(ContractValidationError, "prior messages"):
            asyncio.run(self.transport.generate(request))

    def test_dropping_a_declared_tool_call_is_rejected(self) -> None:
        first = asyncio.run(self.transport.generate(self.first_request))
        issued = first["choices"][0]["message"]["tool_calls"][0]["id"]
        request = self._follow_up(issued)
        request["messages"][2] = dict(request["messages"][2], tool_calls=[])
        with self.assertRaisesRegex(ContractValidationError, "assistant response"):
            asyncio.run(self.transport.generate(request))

    def test_rewritten_non_empty_content_is_still_rejected(self) -> None:
        """Tolerating a *dropped* content must not tolerate an edited one."""
        first = asyncio.run(self.transport.generate(self.first_request))
        issued = first["choices"][0]["message"]["tool_calls"][0]["id"]
        request = self._follow_up(issued)
        request["messages"][2] = dict(request["messages"][2], content="I did nothing")
        with self.assertRaisesRegex(ContractValidationError, "assistant response"):
            asyncio.run(self.transport.generate(request))

    def test_truncated_generation_is_rejected(self) -> None:
        """No EOS means the turn was cut off, and that is never a valid turn."""
        self.server.token_ids = self.tokenizer.encode(GENERATION_TEXT)
        with self.assertRaisesRegex(ContractValidationError, "EOS"):
            asyncio.run(self.transport.generate(self.first_request))


if __name__ == "__main__":
    unittest.main()
