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

    def test_truncated_generation_rejected(self):
        self.ctx.prepare(self.next, [])
        with self.assertRaisesRegex(ContractValidationError, "EOS"):
            self.ctx.accept([1, 2, 3], {"role": "assistant", "content": "done"})

    def test_no_continuation_after_terminal(self):
        self.ctx.prepare(self.next, [])
        self.ctx.accept([4, 999], {"role": "assistant", "content": "done"})
        with self.assertRaisesRegex(ContractValidationError, "terminal"):
            self.ctx.prepare(self.next, [])


if __name__ == "__main__":
    unittest.main()
