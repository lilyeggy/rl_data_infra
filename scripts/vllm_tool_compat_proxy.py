#!/usr/bin/env python3
"""OpenAI-compatible vLLM edge adapter for legacy/Qwen tool-call text.

The backend remains vLLM. This adapter only normalizes Qwen's occasional
``<tools>`` or fenced JSON output into standard OpenAI ``tool_calls`` and
replays buffered responses as SSE for clients such as Pi.
"""
from __future__ import annotations

import json
import re
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

BACKEND = sys.argv[1]
HOST = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8001


def _tool_from_text(text: str) -> list[tuple[str, dict]]:
    found: list[tuple[str, dict]] = []
    # Qwen sometimes emits several adjacent JSON objects inside <tools> or a
    # fenced block. raw_decode lets us extract each nested object safely.
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("name"), str) and isinstance(value.get("arguments"), dict):
            item = (value["name"], value["arguments"])
            if item not in found:
                found.append(item)
    # Also accept Qwen's function-tag variant with JSON in an attribute.
    function_matches = re.findall(r"<function\s+name=\"([^\"]+)\"\s+arguments='(.*?)'", text, re.S)
    function_matches += re.findall(r'<function\s+name="([^"]+)"\s+arguments="(.*?)"', text, re.S)
    for name, raw in function_matches:
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(arguments, dict) and (name, arguments) not in found:
            found.append((name, arguments))
    return found


def _normalize(payload: dict) -> dict:
    for choice in payload.get("choices", []):
        message = choice.get("message", {})
        # vLLM 0.6.1 with --return-tokens-as-token-ids encodes sampled IDs in
        # the OpenAI logprob token string (``token_id:<int>``), rather than in
        # the newer choice.token_ids field requested by Polar.  Materialize
        # that stable canonical field at the compatibility boundary so Polar
        # and the Data Plane consume the same response-token evidence without
        # modifying either Harness or Polar's upstream code.
        if not isinstance(choice.get("token_ids"), list):
            content = choice.get("logprobs", {}).get("content", [])
            token_ids: list[int] = []
            for entry in content if isinstance(content, list) else []:
                token = entry.get("token") if isinstance(entry, dict) else None
                if not isinstance(token, str) or not token.startswith("token_id:"):
                    token_ids = []
                    break
                try:
                    token_ids.append(int(token.removeprefix("token_id:")))
                except ValueError:
                    token_ids = []
                    break
            if token_ids:
                choice["token_ids"] = token_ids
        if message.get("tool_calls") or not isinstance(message.get("content"), str):
            continue
        parsed = _tool_from_text(message["content"])
        if not parsed:
            continue
        tool_calls = []
        for name, arguments in parsed:
            tool_calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                               "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}})
        message["content"] = ""
        message["tool_calls"] = tool_calls
        choice["finish_reason"] = "tool_calls"
    return payload


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(length))
        streaming = bool(request.get("stream"))
        request["stream"] = False
        request.pop("stream_options", None)
        # Polar's newer vLLM engine asks recent servers for these fields per
        # request. The A6000 vLLM 0.6.1 deployment exposes response token ids
        # through the server-level --return-tokens-as-token-ids switch instead
        # and rejects the newer request flags with HTTP 400.
        request.pop("return_token_ids", None)
        request.pop("return_prompt_token_ids", None)
        body = json.dumps(request).encode()
        upstream = Request(BACKEND, data=body, method="POST", headers={"Content-Type": "application/json"})
        try:
            with urlopen(upstream, timeout=900) as response:
                payload = _normalize(json.load(response))
        except Exception as exc:
            self.send_response(502); self.end_headers(); self.wfile.write(str(exc).encode()); return
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        if not streaming:
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(encoded))); self.end_headers(); self.wfile.write(encoded); return
        choice = payload.get("choices", [{}])[0]
        message = choice.get("message", {})
        delta = {"role": "assistant"}
        if message.get("content") is not None: delta["content"] = message.get("content")
        if message.get("tool_calls"): delta["tool_calls"] = message["tool_calls"]
        chunk = {"id": payload.get("id", "chatcmpl-compat"), "object": "chat.completion.chunk", "created": payload.get("created", 0), "model": payload.get("model"), "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        finish = {"id": chunk["id"], "object": "chat.completion.chunk", "created": chunk["created"], "model": chunk["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason", "stop")}]}
        usage = payload.get("usage")
        if usage: finish["usage"] = usage
        out = ("data: " + json.dumps(chunk) + "\n\n" + "data: " + json.dumps(finish) + "\n\n" + "data: [DONE]\n\n").encode()
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

    def log_message(self, *_args: object) -> None:
        return


ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
