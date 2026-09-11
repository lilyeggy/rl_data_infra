"""Pi's synchronous protocol edge to verl's asynchronous token RPC."""

import asyncio
import math
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
                 max_tokens, max_requests, timeout, expected_engine_step=None, expected_tool_schema=None):
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
        self.error = None
        self.context = AppendOnlyTokenContext(
            tokenizer, max_prompt=max_prompt, max_response=max_response
        )

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
        params = dict(self.sampling_params)
        params.update(max_tokens=min(self.max_tokens, self.context.remaining), logprobs=True)
        params["stop_token_ids"] = [self.context.turn_end_id]
        output = await self.server_manager.generate(
            request_id=self.episode_id, prompt_ids=prompt, sampling_params=params
        )
        ids = list(output.token_ids)
        probs = output.log_probs
        if probs is None or len(probs) != len(ids) or not all(math.isfinite(p) for p in probs):
            raise ContractValidationError("missing or invalid native logprobs")
        step = output.extra_fields.get("global_steps")
        if step is None or (self.global_steps is not None and step != self.global_steps):
            raise ContractValidationError("missing or changed engine weight-sync step")
        if self.expected_engine_step is not None and step != self.expected_engine_step:
            raise ContractValidationError("engine has not synchronized the expected checkpoint step")
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
        self.context.accept(ids, message)
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
                "effective_sampling_params": params,
            },
        }
