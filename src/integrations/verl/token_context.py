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
        self.terminator_stripped = False
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
            self.tokens.extend(self._next_suffix(current, tools))
        self.messages = deepcopy(current)
        self.pending = None
        if self.remaining <= 0:
            raise ContractValidationError("episode context budget exhausted")
        return list(self.tokens)

    def next_suffix(self, messages: list[dict[str, Any]], tools: list[Any]) -> list[int]:
        """The exact context tokens this request appends before generating.

        Read-only, so a caller can price the next turn before the engine is
        asked. `prepare` appends what this returns, which is what makes the
        episode's remaining budget knowable in advance.
        """
        if not self.tokens:
            # The initial prompt is not a suffix; `prepare` checks its own budget.
            return []
        return self._next_suffix([_message(m) for m in messages], tools)

    def _next_suffix(
        self, current: list[dict[str, Any]], tools: list[Any]
    ) -> list[int]:
        """`next_suffix` over already-normalized messages.

        Messages must not be normalized twice: `_message` parses tool-call
        arguments in place, so a second pass over its own output is an error.
        """
        if self.pending is None or not self.pending.get("tool_calls"):
            raise ContractValidationError("unexpected request after terminal response")
        boundary = len(self.messages)
        expected = self.messages + [_message(self.pending)]
        # Prior turns are compared in full, including their content: Pi has no
        # business editing history it already sent.
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
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        if self.terminator_stripped:
            # The ledger is missing the terminator the template puts at the end
            # of every assistant turn -- either the engine consumed one and
            # omits special stop tokens from its response, or the policy never
            # emitted one because the budget ran out. Restoring it keeps the
            # prompt on-template. It is context only: no logprob was returned
            # for it, so it must never enter the loss.
            suffix_ids = [self.turn_end_id, *suffix_ids]
        return suffix_ids

    def can_afford(self, context_tokens: int) -> bool:
        """True if spending `context_tokens` still leaves room to generate."""
        return self.remaining - context_tokens > 0

    @property
    def remaining(self) -> int:
        return self.max_response - (len(self.tokens) - self.initial_length)

    def accept(self, token_ids: list[int], message: dict[str, Any]) -> None:
        """Record one generation as the episode's next turn.

        A turn ends on ``<|im_end|>``, but that token is *special* to the
        tokenizer and vLLM does not return special stop tokens, so a turn the
        engine stopped on always arrives without it. Absence of the terminator
        therefore cannot distinguish a finished turn from a truncated one; the
        per-turn evidence does that (``terminated`` in the turn note).

        Both are recorded as a turn, because rejecting the truncated one is
        stricter than the framework this loop drives. verl's own
        ``tool_agent_loop.py`` appends ``response_mask += [1] *
        len(response_ids)`` unconditionally (:246) and terminates only on
        ``len(response_mask) >= self.response_length`` (:254); it never inspects
        ids for an EOS. Measured on four smoke22 episodes, this policy spent its
        entire generation budget on every turn without emitting ``<|im_end|>``,
        so requiring one yielded no trainable episode at all while the framework
        would have trained on every one of them.

        ``terminator_stripped`` marks that the ledger still owes the template a
        terminator before the next turn's observations. It is restored as
        *context only*: no logprob was ever returned for it, so the assembler
        masks it 0 and it can never reach the loss.
        """
        if not token_ids:
            raise ContractValidationError("generation returned no tokens")
        if len(token_ids) > self.remaining:
            raise ContractValidationError("generated response exceeds episode budget")
        self.terminator_stripped = token_ids[-1] != self.turn_end_id
        self.tokens.extend(token_ids)
        self.pending = deepcopy(message)
