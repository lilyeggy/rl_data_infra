"""vLLM-backed OpenAI inference server for E-stage Pi rollouts.

Same protocol and evidence contract as the D-stage transformers server, but the
generation backend is the repaired vLLM (see venv_sitecustomize.py). Two
reasons this matters for E:

1. Architecture fidelity (closeout §3.1): rollouts go through a vLLM engine,
   not a transformers fallback.
2. Token fidelity: the model is fed the *canonical* token stream
   (canonical_prompt + previous native response + observation), not a
   re-rendered approximation of it. Rollout logprobs are therefore computed on
   the same context the trainer will score, which is what the E logprob
   comparison (<=0.05 nat mean, <=0.5 nat P99) is meant to test.

Endpoints:
  POST /v1/chat/completions  (streaming + non-streaming; tools passthrough)
  GET  /healthz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VLLM_TREE = "/home/cxr/ds4-deploy/vllm"

# Same protocol helpers as the D-stage server; text is used only for Pi
# protocol conversion, never to rebuild training tokens.
from stage_d_infer_server import (  # noqa: E402
    _extract_tool_calls,
    _normalize_messages,
    _render_messages,
)


class State:
    llm = None
    tok = None
    lora_request = None
    model_name = "p0-qwen14b"
    calls = 0
    lock = threading.Lock()
    prev_prompt_text = None
    prev_native_response_ids = None
    canonical_ids = None


def _anchor_prompt(tok, prompt_text: str, actual_prompt_ids: list[int], messages):
    """Rebuild the canonical token stream; see the D-stage server for the contract."""
    with State.lock:
        prev_text = State.prev_prompt_text
        prev_native = State.prev_native_response_ids
        canonical = State.canonical_ids
    conversational = [m for m in messages if m.get("role") != "system"]
    if len(conversational) <= 1:
        return list(actual_prompt_ids), 0
    if prev_text is None:
        return list(actual_prompt_ids), 0
    if not prompt_text.startswith(prev_text):
        raise ValueError(
            "conversation-history-rewritten: Pi changed earlier history "
            "(compaction/truncation/edit); episode is not trainable"
        )
    delta_text = prompt_text[len(prev_text):]
    if prev_native and not delta_text:
        raise ValueError("conversation-delta-empty: Pi sent no new context")
    observation_ids = tok(delta_text, add_special_tokens=False)["input_ids"]
    built = list(canonical or actual_prompt_ids) + list(prev_native or []) + list(observation_ids)
    return built, len(observation_ids)


def generate_with_logprobs(messages, tools, sampling, max_tokens: int):
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    tok = State.tok
    prompt_text = _render_messages(tok, messages, tools)
    rendered_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    canonical_ids, observation_len = _anchor_prompt(tok, prompt_text, rendered_ids, messages)

    temperature = float(sampling.get("temperature", 1.0))
    do_sample = temperature > 0
    params = SamplingParams(
        temperature=temperature if do_sample else 0.0,
        top_p=float(sampling.get("top_p", 1.0)) if do_sample else 1.0,
        max_tokens=max_tokens,
        logprobs=1,
    )
    outputs = State.llm.generate(
        [{"prompt_token_ids": list(canonical_ids)}],
        sampling_params=params,
        lora_request=LoRARequest("p0", 1, State.lora_request),
    )
    completion = outputs[0].outputs[0]
    gen_ids = [int(t) for t in completion.token_ids]
    logprobs: list[float] = []
    for step, token_id in enumerate(gen_ids):
        step_logprobs = completion.logprobs[step] if completion.logprobs else None
        if not step_logprobs or token_id not in step_logprobs:
            logprobs.append(float("nan"))
        else:
            logprobs.append(float(step_logprobs[token_id].logprob))
    text = tok.decode(gen_ids, skip_special_tokens=False)
    tool_calls = _extract_tool_calls(text, tools)
    with State.lock:
        State.prev_prompt_text = prompt_text
        State.prev_native_response_ids = list(gen_ids)
        State.canonical_ids = canonical_ids
    return canonical_ids, gen_ids, logprobs, text, tool_calls, observation_len, rendered_ids


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._send(200, {"status": "ok", "calls": State.calls, "backend": "vllm"})
        else:
            self._send(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": "not_found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid_json"})
            return
        messages = payload.get("messages", [])
        tools = payload.get("tools", [])
        sampling = {k: v for k, v in payload.items() if k not in ("model", "messages", "tools", "stream")}
        max_tokens = min(int(payload.get("max_tokens", 1024)), 1024)
        try:
            prompt_ids, gen_ids, logprobs, text, tool_calls, observation_len, rendered_ids = (
                generate_with_logprobs(messages, tools, sampling, max_tokens)
            )
        except Exception as exc:  # noqa: BLE001 - surface as 500, never fake
            import traceback

            traceback.print_exc()
            self._send(500, {"error": "inference_failed", "message": str(exc)[:500]})
            return
        with State.lock:
            State.calls += 1
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model_name = payload.get("model", State.model_name)
        if payload.get("stream") is True:
            if tool_calls:
                frames = [
                    f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n",
                ]
                for call in tool_calls:
                    declaration = {
                        "index": len(frames) - 1,
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["function"]["name"]},
                    }
                    frames.append(f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'tool_calls': [declaration]}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n")
                    argument_delta = {
                        "index": len(frames) - 2,
                        "function": {"arguments": call["function"]["arguments"]},
                    }
                    frames.append(f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'tool_calls': [argument_delta]}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n")
                frames.append(f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]}, ensure_ascii=False)}\n\n")
                frames.append("data: [DONE]\n\n")
            else:
                frames = [
                    f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': text}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n",
                    f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}, ensure_ascii=False)}\n\n",
                    "data: [DONE]\n\n",
                ]
            body = "".join(frames).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._send(200, {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "finish_reason": "tool_calls" if tool_calls else "stop",
                "message": {"role": "assistant", "content": text,
                            "tool_calls": tool_calls or None},
            }],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(gen_ids),
                "total_tokens": len(prompt_ids) + len(gen_ids),
            },
            "agent_data_plane_evidence": {
                "prompt_token_ids": prompt_ids,
                "response_token_ids": gen_ids,
                "response_logprobs": logprobs,
                "backend_model_revision": State.model_name,
                "backend": "vllm",
                "canonical_prompt_token_ids": prompt_ids,
                "observation_token_count": observation_len,
                "rendered_prompt_token_ids": rendered_ids,
            },
        })

    def log_message(self, *args: object) -> None:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--port", type=int, default=8931)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--revision", default="adapter-rev-p0")
    args = parser.parse_args()

    # vLLM's git-based version metadata is not present in the process env; the
    # tree is importable via sitecustomize.py in this venv.
    from transformers import AutoTokenizer
    from vllm import LLM

    State.tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    State.model_name = args.revision

    llm_kwargs = dict(
        model=args.base_model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_lora_rank=8,
        enable_prefix_caching=False,
        enforce_eager=True,
    )
    if args.attention_backend:
        llm_kwargs["attention_backend"] = args.attention_backend
    t0 = time.time()
    try:
        State.llm = LLM(**llm_kwargs)
    except TypeError:
        llm_kwargs.pop("attention_backend", None)
        State.llm = LLM(**llm_kwargs)
    State.lora_request = args.adapter
    print(f"[{time.time()-t0:.1f}s] vllm infer-server on 127.0.0.1:{args.port} adapter={args.adapter}", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
