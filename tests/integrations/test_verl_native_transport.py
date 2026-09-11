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
import tempfile
import unittest
from pathlib import Path

import numpy

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
    def __init__(self, token_ids, global_steps=0, stop_reason="completed"):
        self.token_ids = list(token_ids)
        self.log_probs = [-0.5] * len(self.token_ids)
        self.stop_reason = stop_reason
        self.extra_fields = {"global_steps": global_steps}


class ServerManagerDouble:
    """Serves a generation the way the real engine does.

    Two behaviours are copied from the deployment rather than invented:
      * the step is `None` unless told otherwise, because
        `checkpoint_engine.backend=naive` never propagates `global_steps`;
      * a special stop token is stripped from the returned ids, which is vLLM's
        documented contract for `stop_token_ids` and the reason a cleanly
        finished turn arrives without `<|im_end|>`.
    """

    def __init__(self, token_ids, global_steps=0, strip_special_stop=True,
                 stop_reason="completed"):
        self.token_ids = list(token_ids)
        self.global_steps = global_steps
        self.strip_special_stop = strip_special_stop
        self.stop_reason = stop_reason
        self.prompts: list[list[int]] = []
        self.budgets: list[int] = []

    async def generate(self, *, request_id, prompt_ids, sampling_params):
        self.prompts.append(list(prompt_ids))
        self.budgets.append(int(sampling_params.get("max_tokens", 0)))
        ids = list(self.token_ids)
        if self.strip_special_stop and ids and ids[-1] == END:
            ids = ids[:-1]
        return TokenOutput(ids, global_steps=self.global_steps,
                           stop_reason=self.stop_reason)


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
        # What the engine actually returns: the special stop token is dropped.
        self.stripped = self.generation[:-1]
        self.server = ServerManagerDouble(self.generation)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.note_path = Path(self.tmp.name) / "engine-notes" / "episode.step.jsonl"
        self.turn_path = Path(self.tmp.name) / "engine-notes" / "episode.turns.jsonl"
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
            max_tokens=8192,
            max_requests=4,
            timeout=5.0,
            expected_engine_step=0,
            step_note_path=self.note_path,
            turn_note_path=self.turn_path,
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
        self.assertEqual(ledger[start : start + len(self.stripped)], self.stripped)

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
        self.assertEqual(ledger[start : start + len(self.stripped)], self.stripped)
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

    def test_generation_consuming_the_whole_budget_is_rejected(self) -> None:
        """Truncation is now detected by the budget, not by a missing token.

        The engine strips its special stop token, so absence of `<|im_end|>` is
        normal. What separates a finished turn from a cut-off one is whether the
        engine stopped early.
        """
        budget = self._first_budget()
        self.server.token_ids = [_BASE + ord("x")] * budget
        with self.assertRaisesRegex(ContractValidationError, "terminator"):
            asyncio.run(self.transport.generate(self.first_request))
        self.assertEqual(self.transport.truncated_turns, 1)
        record = json.loads(self.turn_path.read_text().splitlines()[0])
        self.assertEqual(record["tokens"], budget)
        self.assertEqual(record["budget"], budget)
        self.assertFalse(record["terminated"])

    def test_a_finished_turn_is_recorded_as_terminated(self) -> None:
        asyncio.run(self.transport.generate(self.first_request))
        record = json.loads(self.turn_path.read_text().splitlines()[0])
        self.assertEqual(record["tokens"], len(self.stripped))
        self.assertTrue(record["terminated"])
        self.assertEqual(self.transport.truncated_turns, 0)

    def test_stripped_terminator_is_restored_as_context_only(self) -> None:
        asyncio.run(self.transport.generate(self.first_request))
        context = self.transport.context
        self.assertTrue(context.terminator_stripped)
        # The generation arrives without its terminator, and it is not in the
        # loss: only the returned ids were recorded as generated tokens.
        ledger = list(context.tokens)
        stripped = self.generation[:-1]
        start = context.initial_length
        self.assertEqual(ledger[start:start + len(stripped)], stripped)

    def test_returned_terminator_needs_no_restoration(self) -> None:
        self.server.strip_special_stop = False
        asyncio.run(self.transport.generate(self.first_request))
        self.assertFalse(self.transport.context.terminator_stripped)

    def _first_budget(self) -> int:
        # Nothing has been generated yet, so the whole response budget is free.
        context = self.transport.context
        return min(self.transport.max_tokens, context.max_response)


class EngineStepEvidenceTest(unittest.TestCase):
    """The engine cannot always state its step; that must be recorded, not hidden.

    `checkpoint_engine.backend=naive` (the shipped default) routes the weight
    sync through `fsdp_workers.py:1735`, which discards `global_steps`, so the
    serving engine is never told which step it holds. The guard therefore cannot
    demand the engine's confirmation -- but it must not pretend to have it either.
    """

    def setUp(self) -> None:
        self.tokenizer = ReversibleTokenizer()
        generation = self.tokenizer.encode(GENERATION_TEXT) + [END]
        self.server = ServerManagerDouble(generation, global_steps=None)
        self.parser = ParserDouble(calls=(), content=GENERATION_TEXT)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.note_path = Path(self.tmp.name) / "engine-notes" / "episode.step.jsonl"
        self.turn_path = Path(self.tmp.name) / "engine-notes" / "episode.turns.jsonl"

    def _transport(self, expected_engine_step=0):
        return NativeTokenTransport(
            loop=None,
            server_manager=self.server,
            tokenizer=self.tokenizer,
            parser=self.parser,
            episode_id="episode",
            policy_revision="f" * 64,
            sampling_params={},
            max_prompt=8000,
            max_response=40000,
            max_tokens=8192,
            max_requests=4,
            timeout=5.0,
            expected_engine_step=expected_engine_step,
            step_note_path=self.note_path,
            turn_note_path=self.turn_path,
        )

    def _request(self):
        return {
            "model": "m",
            "messages": [
                {"role": "system", "content": "you are pi"},
                {"role": "user", "content": "implement Mbpp/118"},
            ],
            "tools": DECLARED_TOOLS,
        }

    def test_absent_engine_step_is_allowed_but_recorded(self) -> None:
        transport = self._transport()
        response = asyncio.run(transport.generate(self._request()))
        evidence = response["agent_data_plane_evidence"]
        self.assertIsNone(evidence["engine_global_steps"])
        self.assertFalse(evidence["engine_step_confirmed"])
        self.assertEqual(transport.unconfirmed_steps, 1)
        notes = self.note_path.read_text().splitlines()
        self.assertEqual(len(notes), 1)
        record = json.loads(notes[0])
        self.assertIsNone(record["engine_reported_step"])
        self.assertIn("naive", record["reason"])

    def test_numpy_step_stamp_is_recordable(self) -> None:
        """The manager stamps engine_step from a numpy array, so it is np.int64.

        json.dumps refuses numpy scalars; letting that TypeError escape turned
        the note into a 502 that killed every episode on its first call.
        """
        transport = self._transport(expected_engine_step=numpy.int64(0))
        asyncio.run(transport.generate(self._request()))
        record = json.loads(self.note_path.read_text().splitlines()[0])
        self.assertEqual(record["expected_engine_step"], 0)
        self.assertIsInstance(record["expected_engine_step"], int)

    def test_reported_step_is_confirmed_and_leaves_no_note(self) -> None:
        self.server.global_steps = 0
        transport = self._transport()
        response = asyncio.run(transport.generate(self._request()))
        evidence = response["agent_data_plane_evidence"]
        self.assertEqual(evidence["engine_global_steps"], 0)
        self.assertTrue(evidence["engine_step_confirmed"])
        self.assertEqual(transport.unconfirmed_steps, 0)
        self.assertFalse(self.note_path.exists())

    def test_wrong_reported_step_is_still_rejected(self) -> None:
        self.server.global_steps = 1
        with self.assertRaisesRegex(ContractValidationError, "has not synchronized"):
            asyncio.run(self._transport().generate(self._request()))

    def _follow_up(self, issued_id: str) -> dict:
        echo = dict(
            PI_ECHO,
            tool_calls=[dict(PI_ECHO["tool_calls"][0], id=issued_id)],
        )
        return {
            "model": "m",
            "messages": [
                *self._request()["messages"],
                echo,
                {"role": "tool", "tool_call_id": issued_id, "content": '"""stub"""\n'},
            ],
            "tools": DECLARED_TOOLS,
        }

    def test_step_changing_mid_episode_is_still_rejected(self) -> None:
        self.server.global_steps = 0
        transport = self._transport()
        first = asyncio.run(transport.generate(self._request()))
        issued = first["choices"][0]["message"]["tool_calls"][0]["id"]
        self.server.global_steps = 1
        with self.assertRaisesRegex(ContractValidationError, "changed within the episode"):
            asyncio.run(transport.generate(self._follow_up(issued)))


if __name__ == "__main__":
    unittest.main()
