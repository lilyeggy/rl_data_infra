"""Online, append-only context for the frozen Qwen/Hermes Pi adapter.

Chat messages are checked as protocol evidence, never used to reconstruct an
already generated span. Only the initial prompt and new observations are
encoded. This module has no framework imports so contract tests run on CPU.
"""

from copy import deepcopy
from collections.abc import Mapping
import json
from typing import Any

from src.errors import ContractValidationError


def _message(message: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(message)
    result["content"] = result.get("content") or ""
    if isinstance(result["content"], list):
        blocks = result["content"]
        if any(not isinstance(b, dict) or b.get("type") != "text" or
               not isinstance(b.get("text"), str) for b in blocks):
            raise ContractValidationError("multimodal content is unsupported")
        result["content"] = "".join(b["text"] for b in blocks)
    if not isinstance(result["content"], str):
        raise ContractValidationError("multimodal or structured content is unsupported")
    for call in result.get("tool_calls", []):
        try:
            call["function"]["arguments"] = json.loads(call["function"]["arguments"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractValidationError("invalid tool arguments; no JSON repair allowed") from exc
    return result


def _same_assistant_turn(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Compare only the parts of an assistant turn that are load-bearing.

    The model's own words are never taken from Pi: the native tokens live in the
    ledger and the prompt is built from the ledger, so an assistant ``content``
    field cannot influence what the policy sees. What does matter is which tools
    Pi decided to run, because their results are the observations appended next.

    Pi keeps the tool calls and drops the content -- measured on the smoke16 run,
    it echoed ``content: null`` while returning one of the declared calls -- so an
    empty content on either side is not a rewrite. A non-empty content must still
    match, which keeps the guard effective for any client that does echo it.
    """
    if actual.get("role") != expected.get("role"):
        return False
    if actual.get("tool_calls") != expected.get("tool_calls"):
        return False
    actual_content = actual.get("content") or ""
    expected_content = expected.get("content") or ""
    return not actual_content or actual_content == expected_content


class AppendOnlyTokenContext:
    def __init__(self, tokenizer: Any, *, max_prompt: int, max_response: int):
        self.tokenizer = tokenizer
        self.max_prompt = max_prompt
        self.max_response = max_response
        self.tokens: list[int] = []
        self.initial_length = 0
        self.messages: list[dict[str, Any]] = []
        self.tools: Any = None
        self.pending: dict[str, Any] | None = None
        self.turn_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if (not isinstance(self.turn_end_id, int) or
                tokenizer.decode([self.turn_end_id]) != "<|im_end|>"):
            raise ContractValidationError("adapter requires frozen Qwen im_end template")

    def prepare(self, messages: list[dict[str, Any]], tools: list[Any]) -> list[int]:
        current = [_message(m) for m in messages]
        if not self.tokens:
            if any(m.get("role") not in ("system", "user") for m in current):
                raise ContractValidationError("initial request contains replayed history")
            tokens = self.tokenizer.apply_chat_template(
                current, tools=tools, tokenize=True, add_generation_prompt=True
            )
            if isinstance(tokens, Mapping):
                tokens = tokens["input_ids"]
            if any(isinstance(t, bool) or not isinstance(t, int) for t in tokens):
                raise ContractValidationError("tokenizer returned non-integer prompt IDs")
            if not tokens or len(tokens) > self.max_prompt:
                raise ContractValidationError("initial prompt exceeds frozen budget")
            self.tokens = list(tokens)
            self.initial_length = len(tokens)
            self.tools = deepcopy(tools)
        else:
            if tools != self.tools:
                raise ContractValidationError("tool schema changed within episode")
            if self.pending is None or not self.pending.get("tool_calls"):
                raise ContractValidationError("unexpected request after terminal response")
            boundary = len(self.messages)
            expected = self.messages + [_message(self.pending)]
            # Prior turns are compared in full, including their content: Pi has
            # no business editing history it already sent.
            if current[:boundary] != self.messages:
                raise ContractValidationError("Pi rewrote prior messages")
            if not _same_assistant_turn(current[boundary], expected[boundary]):
                raise ContractValidationError("Pi rewrote the assistant response")
            observations = current[boundary + 1:]
            wanted = [c["id"] for c in self.pending["tool_calls"]]
            received = [m.get("tool_call_id") for m in observations]
            if (not observations or received != wanted or
                    any(m.get("role") != "tool" for m in observations)):
                raise ContractValidationError("missing, duplicate or mismatched tool results")
            # Render only to obtain the NEW template suffix. The prefix is not
            # re-tokenized and can never replace native model output.
            before = self.tokenizer.apply_chat_template(
                current[:boundary + 1], tools=tools, tokenize=False,
                add_generation_prompt=False,
            )
            after = self.tokenizer.apply_chat_template(
                current, tools=tools, tokenize=False, add_generation_prompt=True,
            )
            if not before.endswith("<|im_end|>\n") or not after.startswith(before):
                raise ContractValidationError("BLOCKED_TOKEN_CONTEXT: template boundary changed")
            suffix = "\n" + after[len(before):]
            self.tokens.extend(self.tokenizer.encode(suffix, add_special_tokens=False))
        self.messages = deepcopy(current)
        self.pending = None
        if self.remaining <= 0:
            raise ContractValidationError("episode context budget exhausted")
        return list(self.tokens)

    @property
    def remaining(self) -> int:
        return self.max_response - (len(self.tokens) - self.initial_length)

    def accept(self, token_ids: list[int], message: dict[str, Any]) -> None:
        if not token_ids or token_ids[-1] != self.turn_end_id:
            # v0.7.1 collapses length/stop to 'completed'. Require an actual
            # generated EOS; never synthesize one or accept a truncated turn.
            raise ContractValidationError("generation lacks native EOS; truncated or aborted")
        if len(token_ids) > self.remaining:
            raise ContractValidationError("generated response exceeds episode budget")
        self.tokens.extend(token_ids)
        self.pending = deepcopy(message)
