"""Pi's synchronous protocol edge to verl's asynchronous token RPC."""

import asyncio
import json
import math
from pathlib import Path
from uuid import uuid4
from typing import Any

from src.errors import ContractValidationError
from src.capture.tool_calls import extract_tool_calls
from src.contracts._json import sha256_json
from src.integrations.verl.token_context import AppendOnlyTokenContext


def _tool_calls(upstream_calls: Any, text: str, tools: Any) -> list[dict[str, Any]]:
    """Turn one generation into the tool calls Pi will actually run.

    Stock vLLM tool parsers match the ``<tool_call>`` tagged form only, and this
    policy emits bare JSON: over the ten generations of the smoke16 run, every
    single one contained zero tags. Trusting the upstream parser alone therefore
    yields no tool call, Pi executes nothing, and the episode dies on its first
    turn -- which is exactly what the earlier HTTP path hit before it grew its
    own extraction. The same strict extractor is reused here (rather than
    reimplemented) so the repository holds one extraction policy, not two.

    This is extraction, not repair. An undeclared tool name, malformed JSON, a
    non-object or empty ``arguments`` object still yields nothing, and a tagged
    block the upstream parser refused is treated as malformed rather than being
    re-parsed: the tags are what the parser is responsible for, so if it declined
    them the generation is not trustworthy.
    """
    opened = text.count("<tool_call>")
    closed = text.count("</tool_call>")
    if upstream_calls:
        if opened != len(upstream_calls) or closed != len(upstream_calls):
            raise ContractValidationError(
                "generation mixes parsed and unparsed tool calls; no repair allowed"
            )
        return [
            {
                "id": f"call_{uuid4().hex}",
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in upstream_calls
        ]
    if opened or closed:
        raise ContractValidationError("malformed tool call block; no repair allowed")
    return extract_tool_calls(text, tools)


class NativeTokenTransport:
    def __init__(self, *, loop, server_manager, tokenizer, parser, episode_id,
                 policy_revision, sampling_params, max_prompt, max_response,
                 max_tokens, max_requests, timeout, expected_engine_step=None,
                 expected_tool_schema=None, step_note_path=None, turn_note_path=None):
        self.loop = loop
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.parser = parser
        self.episode_id = episode_id
        self.policy_revision = policy_revision
        self.sampling_params = dict(sampling_params)
        self.max_tokens = max_tokens
        self.max_requests = max_requests
        self.timeout = timeout
        self.calls = 0
        self.global_steps = None
        self.expected_engine_step = expected_engine_step
        self.expected_tool_schema = expected_tool_schema
        self.step_note_path = Path(step_note_path) if step_note_path else None
        self.turn_note_path = Path(turn_note_path) if turn_note_path else None
        self.unconfirmed_steps = 0
        self.truncated_turns = 0
        self.error = None
        self.context = AppendOnlyTokenContext(
            tokenizer, max_prompt=max_prompt, max_response=max_response
        )

    def _note_unconfirmed_step(self) -> None:
        """Record, once per episode, that the engine could not state its step.

        `checkpoint_engine.backend=naive` -- the shipped default -- routes the
        weight sync through ``fsdp_workers.py:1735``, whose ``update_weights``
        discards ``global_steps`` instead of reaching
        ``ServerAdapter.set_global_steps``. The engine's ``self.global_steps``
        therefore stays `None` and every ``TokenOutput`` reports it as such.

        That is a real gap in what we can show, so it is written to the episode
        rather than papered over: the policy identity we do bind is the
        trainer's own step plus the checkpoint digest, and this note marks where
        the engine's independent confirmation is missing.
        """
        self.unconfirmed_steps += 1
        if self.step_note_path is None:
            return
        # The manager stamps this from a numpy array, so it arrives as np.int64.
        # json.dumps refuses numpy scalars, and that TypeError once escaped as a
        # 502 that killed every episode on its first call.
        expected = self.expected_engine_step
        record = {
            "episode_id": str(self.episode_id),
            "call": int(self.calls),
            "expected_engine_step": None if expected is None else int(expected),
            "engine_reported_step": None,
            "reason": (
                "engine did not report a weight-sync step; "
                "checkpoint_engine.backend=naive does not propagate "
                "global_steps to the serving engine"
            ),
        }
        try:
            self.step_note_path.parent.mkdir(parents=True, exist_ok=True)
            with self.step_note_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except (OSError, TypeError, ValueError):
            # A missing note must never take the episode down; the flag on the
            # response extension still carries the fact for this call.
            pass

    def _record_turn(
        self, *, tokens: int, budget: int, stop_reason: Any, terminated: bool
    ) -> None:
        """Record every turn's length, so the policy's stopping behaviour is measured.

        Whether this policy terminates its turn at all is the question that
        decides how a budget-exhausted generation should be treated, and it can
        only be answered from real runs. One line per model call.
        """
        if not terminated:
            self.truncated_turns += 1
        if self.turn_note_path is None:
            return
        record = {
            "episode_id": str(self.episode_id),
            "call": int(self.calls),
            "tokens": int(tokens),
            "budget": int(budget),
            "stop_reason": None if stop_reason is None else str(stop_reason),
            "terminated": bool(terminated),
        }
        try:
            self.turn_note_path.parent.mkdir(parents=True, exist_ok=True)
            with self.turn_note_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except (OSError, TypeError, ValueError):
            pass

    def __call__(self, payload: dict[str, Any]):
        future = asyncio.run_coroutine_threadsafe(self.generate(payload), self.loop)
        try:
            return 200, future.result(timeout=self.timeout)
        except Exception as exc:
            future.cancel()
            self.error = f"{type(exc).__name__}: {exc}"
            return 502, {"error": {"type": "native_token_bridge_failed", "message": self.error}}

    async def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.error is not None or self.calls >= self.max_requests:
            raise ContractValidationError("episode failed or model request budget exhausted")
        self.calls += 1
        if self.expected_tool_schema is not None and sha256_json(payload.get("tools", [])) != self.expected_tool_schema:
            raise ContractValidationError("actual Pi tool schema differs from frozen policy identity")
        prompt = self.context.prepare(payload["messages"], payload.get("tools", []))
        budget = min(self.max_tokens, self.context.remaining)
        params = dict(self.sampling_params)
        params.update(max_tokens=budget, logprobs=True)
        # The engine must stop exactly at the turn terminator: this policy does
        # not always emit it, and without the stop the only limiter is the budget.
        # vLLM does not return special stop tokens, which `accept` accounts for.
        params["stop_token_ids"] = [self.context.turn_end_id]
        output = await self.server_manager.generate(
            request_id=self.episode_id, prompt_ids=prompt, sampling_params=params
        )
        ids = list(output.token_ids)
        probs = output.log_probs
        if probs is None or len(probs) != len(ids) or not all(math.isfinite(p) for p in probs):
            raise ContractValidationError("missing or invalid native logprobs")
        step = output.extra_fields.get("global_steps")
        if step is None:
            self._note_unconfirmed_step()
        else:
            if self.global_steps is not None and step != self.global_steps:
                raise ContractValidationError("engine weight-sync step changed within the episode")
            if self.expected_engine_step is not None and step != self.expected_engine_step:
                raise ContractValidationError(
                    "engine has not synchronized the expected checkpoint step"
                )
            self.global_steps = step
        content, upstream_calls = await self.parser.extract_tool_calls(ids)
        text = self.tokenizer.decode(ids)
        calls = _tool_calls(upstream_calls, text, payload.get("tools", []))
        # The template EOS is protocol framing; its native token remains in
        # the ledger and loss. Pi receives only assistant content.
        content = content.removesuffix("<|im_end|>")
        message = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = calls
        # A turn that ended on its own stops short of the budget. Consuming the
        # whole budget means the engine hit max_tokens, i.e. the turn was cut off
        # and the episode must not be treated as having completed it.
        stopped_early = len(ids) < budget
        aborted = getattr(output, "stop_reason", None) == "aborted"
        engine_finished = stopped_early and not aborted
        self._record_turn(
            tokens=len(ids),
            budget=budget,
            stop_reason=getattr(output, "stop_reason", None),
            terminated=engine_finished,
        )
        self.context.accept(
            ids, message, engine_finished=engine_finished, budget=budget
        )
        return {
            "id": f"chatcmpl-{uuid4().hex}", "object": "chat.completion",
            "model": payload["model"],
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(ids),
                      "total_tokens": len(prompt) + len(ids)},
            "agent_data_plane_evidence": {
                "prompt_token_ids": prompt, "response_token_ids": ids,
                "response_logprobs": list(probs),
                "backend_model_revision": self.policy_revision,
                "engine_global_steps": step,
                # False means the engine could not state which step it serves.
                # The policy identity still comes from the trainer's own
                # `global_steps` and, past step 0, from the on-disk checkpoint
                # digest the manager hashes -- but the engine's independent
                # confirmation is genuinely absent, and the episode records it.
                "engine_step_confirmed": step is not None,
                "effective_sampling_params": params,
            },
        }
