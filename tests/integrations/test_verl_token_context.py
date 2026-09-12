"""CPU contracts for actual online token history (no fake GPU evidence)."""

from copy import deepcopy
import unittest

from src.errors import ContractValidationError
from src.integrations.verl.token_context import AppendOnlyTokenContext


class Tokenizer:
    eos_token = "<|im_end|>"
    eos_token_id = 999

    def convert_tokens_to_ids(self, token):
        return 999

    def decode(self, tokens):
        return "<|im_end|>" if tokens == [999] else ""

    def encode(self, text, **kwargs):
        return list(text.encode())

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        text = "".join(f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n"
                       for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return self.encode(text) if tokenize else text


class ContextTest(unittest.TestCase):
    def setUp(self):
        self.ctx = AppendOnlyTokenContext(Tokenizer(), max_prompt=500, max_response=1000)
        self.initial = [{"role": "user", "content": "read task"}]
        self.first = self.ctx.prepare(self.initial, [])
        self.assistant = {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "call1", "type": "function",
                            "function": {"name": "read", "arguments": '{"path":"task"}'}}],
        }
        # Deliberately impossible to get by re-encoding assistant.content.
        self.native = [123456, 789012, 999]
        self.ctx.accept(self.native, self.assistant)
        self.next = self.initial + [self.assistant, {
            "role": "tool", "tool_call_id": "call1", "content": "actual observation",
        }]

    def test_native_prefix_survives_structured_assistant(self):
        prompt = self.ctx.prepare(self.next, [])
        prefix = self.first + self.native
        self.assertEqual(prompt[:len(prefix)], prefix)
        self.assertIn(b"actual observation", bytes(prompt[len(prefix):]))

    def test_old_message_edit_rejected(self):
        messages = deepcopy(self.next)
        messages[0]["content"] = "compacted"
        with self.assertRaisesRegex(ContractValidationError, "rewrote"):
            self.ctx.prepare(messages, [])

    def test_missing_duplicate_wrong_result_rejected(self):
        variants = [self.next[:-1], self.next + [self.next[-1]], deepcopy(self.next)]
        variants[-1][-1]["tool_call_id"] = "foreign"
        for messages in variants:
            with self.subTest(messages=messages):
                with self.assertRaises(ContractValidationError):
                    self.ctx.prepare(messages, [])

    def test_schema_change_rejected(self):
        with self.assertRaisesRegex(ContractValidationError, "schema"):
            self.ctx.prepare(self.next, [{"function": "changed"}])

    def test_truncated_generation_is_accepted_as_a_turn(self):
        """A turn the budget cut off is a turn, as it is in verl's own loop.

        Measured on four smoke22 episodes: every turn spent its whole generation
        budget and never emitted <|im_end|>. Requiring the terminator yielded no
        trainable episode at all, while ``tool_agent_loop.py`` masks the ids of
        such a turn with 1 unconditionally (:246).
        """
        self.ctx.prepare(self.next, [])
        truncated = [11, 22, 33]
        self.ctx.accept(truncated, self.assistant)
        self.assertTrue(self.ctx.terminator_stripped)
        self.assertEqual(self.ctx.tokens[-len(truncated):], truncated)

    def test_missing_terminator_is_restored_as_context_only(self):
        self.ctx.prepare(self.next, [])
        cut_off = [11, 22, 33]
        self.ctx.accept(cut_off, self.assistant)
        ledger_after_cut_off = list(self.ctx.tokens)
        self.assertEqual(ledger_after_cut_off[-len(cut_off):], cut_off)
        third = self.next + [self.assistant, {
            "role": "tool", "tool_call_id": "call1", "content": "second observation",
        }]
        prompt = self.ctx.prepare(third, [])
        # Nothing was rewritten: the ledger is still the first prompt, its native
        # generation, its observation and then the cut-off generation, with only
        # the new suffix appended.
        self.assertEqual(prompt[:len(ledger_after_cut_off)], ledger_after_cut_off)
        # The terminator the cut-off turn never emitted is restored at the head of
        # that suffix, so the prompt stays on-template. No logprob was ever
        # returned for it, which is what makes it context and not an action.
        self.assertEqual(prompt[len(ledger_after_cut_off)], 999)
        self.assertGreater(len(prompt), len(ledger_after_cut_off) + 1)

    def test_no_continuation_after_terminal(self):
        self.ctx.prepare(self.next, [])
        self.ctx.accept([4, 999], {"role": "assistant", "content": "done"})
        with self.assertRaisesRegex(ContractValidationError, "terminal"):
            self.ctx.prepare(self.next, [])


if __name__ == "__main__":
    unittest.main()
