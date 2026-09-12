"""Minimal OpenAI-compatible inference server for D-stage Pi rollouts.

Deviation context: vLLM is not importable on this host (precompiled install
is broken), so verl-managed vLLM rollout is unavailable. This server runs the
frozen 14B Base + P0 adapter via transformers on GPU1 and returns NATIVE token
ids + logprobs per call. It never fabricates tokens: response_token_ids and
response_logprobs come from the actual generate() output scores.

Endpoints:
  POST /v1/chat/completions  (non-streaming; tools passthrough for Pi)
  GET  /healthz
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class State:
    model = None
    tok = None
    model_name = "p0-qwen14b"
    calls = 0
    lock = threading.Lock()


def _normalize_messages(messages):
    """Flatten OpenAI content blocks to plain text for the P0 chat template.

    Input normalization only; native output tokens are never altered.
    Non-text blocks (e.g. images) are refused instead of silently dropped.
    """
    normalized = []
    for message in messages:
        message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            texts = []
            for block in content:
                if not isinstance(block, dict):
                    raise ValueError("message content block must be an object")
                block_type = block.get("type", "text")
                if block_type == "text":
                    texts.append(str(block.get("text", "")))
                else:
                    raise ValueError(f"unsupported content block: {block_type}")
            message["content"] = "\n".join(texts)
        normalized.append(message)
    return normalized


def generate_with_logprobs(messages, tools, sampling, max_tokens: int):
    tok = State.tok
    prompt_text = tok.apply_chat_template(
        _normalize_messages(messages), tools=tools or None, tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tok(prompt_text, return_tensors="pt").to("cuda")
    prompt_ids = inputs["input_ids"][0].tolist()
    do_sample = float(sampling.get("temperature", 1.0)) > 0
    with torch.inference_mode():
        out = State.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=float(sampling.get("temperature", 1.0)) if do_sample else None,
            top_p=float(sampling.get("top_p", 1.0)) if do_sample else None,
            return_dict_in_generate=True,
            output_scores=True,
        )
    gen_ids = out.sequences[0][len(prompt_ids):].tolist()
    logprobs: list[float] = []
    for step, scores in enumerate(out.scores):
        log_prob = torch.log_softmax(scores[0].float(), dim=-1)[gen_ids[step]].item()
        logprobs.append(log_prob)
    text = tok.decode(gen_ids, skip_special_tokens=False)
    tool_calls = _extract_tool_calls(text, tools)
    return prompt_ids, gen_ids, logprobs, text, tool_calls


def _extract_tool_calls(text: str, tools) -> list:
    """Parse model-native tool-call JSON into OpenAI tool_calls.

    Reuses the upstream Qwen/Hermes convention (verl's HermesToolParser logic):
    `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` blocks plus the
    bare-JSON variant this SFT policy emits. Text decoding is only used for
    protocol conversion; native training tokens are never rewritten.
    Only tool names declared in `tools` are accepted; anything else (or
    malformed JSON) yields no tool calls instead of a repaired fabrication.
    """
    import re

    if not tools:
        return []
    allowed = set()
    for tool in tools:
        if isinstance(tool, dict):
            fn = tool.get("function", {})
            if isinstance(fn, dict) and fn.get("name"):
                allowed.add(fn["name"])
    candidates: list[str] = []
    candidates.extend(re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL))
    # P0 emits [{...,"name":"<tool>","type":"toolCall"}] with nested arguments
    # objects; regex cannot span nested braces, so scan balanced {...} blocks.
    for start in range(len(text)):
        if text[start] != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            char = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:end + 1])
                    break
    calls = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        name = parsed.get("name")
        arguments = parsed.get("arguments", {})
        if name not in allowed:
            continue
        if not isinstance(arguments, dict):
            continue
        # Drop empty-argument calls: Pi executes them, fails, and retries in
        # a loop. Returning no tool calls lets Pi re-ask instead of spinning.
        if not arguments:
            continue
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
            },
        })
    return calls


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
            self._send(200, {"status": "ok", "calls": State.calls})
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
            prompt_ids, gen_ids, logprobs, text, tool_calls = generate_with_logprobs(
                messages, tools, sampling, max_tokens
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
        # Pi tool protocol: parsed tool_calls + text; Pi executes tools.
        # Native token evidence travels alongside for the proxy to capture.
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
            "native_evidence": {
                "prompt_token_ids": prompt_ids,
                "response_token_ids": gen_ids,
                "response_logprobs": logprobs,
            },
        })

    def log_message(self, *args: object) -> None:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--port", type=int, default=8931)
    args = parser.parse_args()
    State.tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda")
    State.model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    State.model.eval()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"infer-server on 127.0.0.1:{args.port} model={args.adapter}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
