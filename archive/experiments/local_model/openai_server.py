"""OpenAI-compatible inference server for the local student model.

Serves Qwen2.5-7B-Instruct (+ optional LoRA adapter) behind a minimal
/v1/chat/completions endpoint so the REAL Pi harness can drive the local model
via a custom provider (no micro-harness, no train/eval skew).

Design notes
------------
- transformers (not vLLM) to avoid the torch2.13/cu130 wheel minefield; our
  scale (small task suite, small GRPO groups) does not need vLLM throughput.
- Tool calling: Qwen2.5-Instruct emits Hermes-style `<tool_call>{...}</tool_call>`
  blocks. We parse them into OpenAI `tool_calls` so Pi executes its real tools
  (read/grep/find/ls) and feeds results back through the same loop.
- /v1/reload_adapter hot-swaps the LoRA adapter so the GRPO trainer can push
  updated weights into the inference server (decoupled inference/training).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("LOCAL_MODEL", "/root/rivermind-data/models/qwen2.5-7b-instruct")
ADAPTER = os.environ.get("LOCAL_ADAPTER") or None  # e.g. .../qwen-sft-adapter-7b-v3
PORT = int(os.environ.get("PORT", "8000"))
_DEFAULT_TEMP_ENV = float(os.environ.get("QWEN_SERV_TEMPERATURE", "0.0"))
# Mutable default sampling temperature, toggled at runtime via /v1/set_sampling
_state = {"temperature": _DEFAULT_TEMP_ENV}

app = FastAPI()
print(f"[server] loading base={MODEL_PATH} adapter={ADAPTER}", flush=True)
_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if _tokenizer.pad_token is None:
    _tokenizer.pad_token = _tokenizer.eos_token
_base = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
)
_model = _base
_adapter_name: str | None = None
if ADAPTER:
    from peft import PeftModel

    _model = PeftModel.from_pretrained(_base, ADAPTER)
    _adapter_name = ADAPTER
_model = _model.to("cuda")
_model.eval()
print(f"[server] ready adapter={_adapter_name}", flush=True)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
# Some models (e.g. Coder-14B) emit tool calls as a bare or code-fenced
# {"name":..., "arguments":...} object instead of the <tool_call> wrapper.
# Accept all three so the harness is robust across models' tool-call formats.
_CODE_FENCE_TOOL_RE = re.compile(r"```(?:json)?\s*(\{\s*\"name\".*?\})\s*```", re.S)
_BARE_TOOL_RE = re.compile(r"(\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"arguments\"\s*:\s*\{.*?\}\s*\})", re.S)


def _iter_tool_payloads(text: str):
    """Yield {"name","arguments"} payloads from any supported tool-call format.
    Try the native <tool_call> wrapper first, then code fences, then bare JSON.
    The name+arguments shape check prevents misfiring on answer JSON."""
    for rx in (_TOOL_CALL_RE, _CODE_FENCE_TOOL_RE, _BARE_TOOL_RE):
        payloads = []
        for m in rx.finditer(text):
            p = _loads(m.group(1))
            if isinstance(p, dict) and p.get("name") and "arguments" in p:
                payloads.append(p)
        if payloads:
            return payloads
    return []


def _flatten_content(content) -> str:
    """Pi sends content as a list of parts ([{type:text,text:...}]); the Qwen
    template wants a plain string. Flatten."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif "text" in p:
                    parts.append(str(p.get("text")))
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(content)


# Long multi-turn agent contexts (big file dumps) OOM the 24GB card on a single
# forward pass. Bound every message's size, keeping head+tail (the code around
# an edit lives at both ends of a file dump). This keeps total prompt tokens in
# a range the 7B server can actually forward without OOM-500.
_MSG_CHAR_CAP = 2000


def _cap(text: str) -> str:
    if len(text) <= _MSG_CHAR_CAP:
        return text
    half = _MSG_CHAR_CAP // 2
    return text[:half] + f"\n...[+{len(text) - _MSG_CHAR_CAP} chars truncated]...\n" + text[-half:]


def _normalize_messages(raw: list[dict]) -> list[dict]:
    """Coerce OpenAI messages into what the Qwen chat template expects."""
    out: list[dict] = []
    for m in raw:
        role = m.get("role")
        if role == "developer":  # safety; we ask Pi to send system instead
            role = "system"
        content = _flatten_content(m.get("content"))
        if role == "assistant" and m.get("tool_calls"):
            blocks = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                blocks.append(
                    "<tool_call>\n"
                    + json.dumps({"name": fn.get("name"), "arguments": _loads(fn.get("arguments"))}, ensure_ascii=False)
                    + "\n</tool_call>"
            )
            content = (content + "\n" + "\n".join(blocks)).strip()
        # Pi's native tool-result messages are not guaranteed to contain our
        # historical <tool_response> marker. Cap every non-system turn before
        # chat templating so a long sequence of file reads cannot create a
        # >32k-token prompt (and a later CUDA indexing failure).
        if role != "system":
            content = _cap(content)
        out.append({"role": role, "content": content})
    return out


def _loads(s):
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s) if isinstance(s, str) and s.strip() else {}
    except Exception:
        return {}


ROLLOUT_LOG = os.environ.get("ROLLOUT_LOG", "/root/rivermind-data/qwen-rollout-log.jsonl")
_EVIDENCE_EXTENSION = "agent_data_plane_evidence"
# Set from a manifest of the exact base shards and active adapter.  A hot
# adapter reload without an explicit replacement revision deliberately clears
# this value, so its responses cannot be mistaken for RL-ready evidence.
_model_revision = os.environ.get("LOCAL_MODEL_REVISION") or None
# Transformers generation on one model instance is not thread-safe. FastAPI
# may dispatch sync handlers to multiple worker threads, so serialize the
# CUDA-critical section and let concurrent requests queue at the API boundary.
_GENERATION_LOCK = threading.Lock()


def _generate(messages, tools, max_tokens, temperature):
    # Clamp generation length: tool calls and the final JSON answer are short
    # (<60 tokens). Uncapped sampling at temp>0 rambles to the token budget,
    # inflating multi-turn context and OOMing the 24GB card.
    max_tokens = int(min(max_tokens or 200, 200))
    text = _tokenizer.apply_chat_template(
        messages,
        tools=tools if tools else None,
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = _tokenizer(text, return_tensors="pt")
    prompt_ids = ids.input_ids[0]
    # Hard cap total prompt length so a long multi-turn agent context can never
    # OOM the 24GB card, regardless of how many tool results accumulated.
    # Keep the head (system + problem) and the tail (recent context) — the
    # middle (old file dumps) is the most dispensable.
    MAX_PROMPT = 8000
    if int(prompt_ids.shape[0]) > MAX_PROMPT:
        half = MAX_PROMPT // 2
        prompt_ids = torch.cat([prompt_ids[:half], prompt_ids[-half:]])
    # A token ID outside the loaded embedding table triggers a CUDA
    # device-side assert. That assertion poisons the whole CUDA context, so
    # validate on CPU before the first device operation and return a normal
    # request error instead of turning every later rollout into a 500.
    embedding_vocab = int(_model.get_input_embeddings().num_embeddings)
    smallest = int(prompt_ids.min().item())
    largest = int(prompt_ids.max().item())
    if smallest < 0 or largest >= embedding_vocab:
        raise ValueError(
            f"prompt token IDs [{smallest}, {largest}] exceed model embedding vocab "
            f"[0, {embedding_vocab - 1}]"
        )
    ids = {"input_ids": prompt_ids.unsqueeze(0).to("cuda"),
           "attention_mask": torch.ones_like(prompt_ids).unsqueeze(0).to("cuda")}
    n_in = int(prompt_ids.shape[0])
    do_sample = temperature and temperature > 0
    try:
        with torch.no_grad():
            out = _model.generate(
                **ids,
                max_new_tokens=max_tokens,
                do_sample=bool(do_sample),
                temperature=(temperature if do_sample else None),
                top_p=(0.9 if do_sample else None),
                pad_token_id=_tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise
    finally:
        torch.cuda.empty_cache()
    gen_ids = out.sequences[0][n_in:]
    gen_text = _tokenizer.decode(gen_ids, skip_special_tokens=True)
    if len(out.scores) != int(gen_ids.shape[0]):
        raise RuntimeError("generation scores do not align with completion token ids")
    behavior_logprobs = [
        float(torch.log_softmax(score.float(), dim=-1)[0, token_id].item())
        for score, token_id in zip(out.scores, gen_ids.tolist(), strict=True)
    ]
    return {
        "text": gen_text,
        "prompt_ids": prompt_ids.tolist(),
        "completion_ids": gen_ids.tolist(),
        "completion_logprobs": behavior_logprobs,
        "n_in": n_in,
        "n_out": int(gen_ids.shape[0]),
    }


class ChatReq(BaseModel):
    messages: list[dict]
    model: str | None = None
    tools: list[dict] | None = None
    tool_choice: object | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool | None = None


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": _adapter_name or MODEL_PATH, "object": "model"}]}


@app.get("/health")
def health():
    return {"ok": True, "adapter": _adapter_name, "temperature": _state["temperature"]}


@app.post("/v1/set_sampling")
def set_sampling(payload: dict):
    """Toggle default sampling temperature at runtime (GRPO needs diverse
    rollouts within a group; eval needs greedy). Avoids server restarts."""
    _state["temperature"] = float(payload.get("temperature", 0.0))
    return {"ok": True, "temperature": _state["temperature"]}


@app.post("/v1/reload_adapter")
def reload_adapter(payload: dict):
    """Hot-swap the LoRA adapter (GRPO weight sync). Empty path -> base."""
    global _model, _adapter_name, _model_revision
    from peft import PeftModel

    path = payload.get("path") or None
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa"
    )
    if path:
        m = PeftModel.from_pretrained(m, path)
    _model = m.to("cuda")
    _model.eval()
    _adapter_name = path
    revision = payload.get("model_revision")
    _model_revision = revision if isinstance(revision, str) and revision else None
    torch.cuda.empty_cache()
    return {"ok": True, "adapter": _adapter_name, "model_revision": _model_revision}


@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    messages = _normalize_messages(req.messages)
    tools = req.tools or None
    temperature = req.temperature if req.temperature is not None else _state["temperature"]
    max_tokens = req.max_tokens or 1024
    with _GENERATION_LOCK:
        gen = _generate(messages, tools, max_tokens, temperature)
    gen_text, n_in, n_out = gen["text"], gen["n_in"], gen["n_out"]

    # Record the rollout for the GRPO trainer (token ids + behavior logprobs).
    try:
        with open(ROLLOUT_LOG, "a") as f:
            f.write(json.dumps({
                "ts": time.time(),
                "temperature": temperature,
                "prompt_ids": gen["prompt_ids"],
                "completion_ids": gen["completion_ids"],
                "completion_logprobs": gen["completion_logprobs"],
                "text": gen_text,
            }) + "\n")
    except Exception:  # noqa: BLE001 - logging must never break inference
        pass

    tool_calls = []
    for payload in _iter_tool_payloads(gen_text):
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": payload.get("name"),
                    "arguments": json.dumps(payload.get("arguments", {}), ensure_ascii=False),
                },
            }
        )

    if tool_calls:
        message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        finish = "tool_calls"
    else:
        # strip any tool_call markup leakage from plain content
        content = _TOOL_CALL_RE.sub("", gen_text).strip()
        message = {"role": "assistant", "content": content}
        finish = "stop"

    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model_name = req.model or (_adapter_name or MODEL_PATH)

    if req.stream:
        # Pi's openai-completions streams via SSE. We generate greedily/in one
        # shot then replay as a single-delta chunk stream ending in [DONE].
        def sse():
            delta: dict = {"role": "assistant"}
            if tool_calls:
                delta["tool_calls"] = [
                    {"index": i, "id": tc["id"], "type": "function", "function": tc["function"]}
                    for i, tc in enumerate(tool_calls)
                ]
            else:
                delta["content"] = message["content"]
            chunk1 = {
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk1, ensure_ascii=False)}\n\n"
            chunk2 = {
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                "usage": {"prompt_tokens": int(n_in), "completion_tokens": int(n_out), "total_tokens": int(n_in + n_out)},
            }
            yield f"data: {json.dumps(chunk2, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": int(n_in),
            "completion_tokens": int(n_out),
            "total_tokens": int(n_in + n_out),
        },
        _EVIDENCE_EXTENSION: {
            "backend_model_revision": _model_revision,
            "prompt_token_ids": gen["prompt_ids"],
            "response_token_ids": gen["completion_ids"],
            "response_logprobs": gen["completion_logprobs"],
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
