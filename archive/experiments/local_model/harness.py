"""ARCHIVED: shared micro-harness for retired local-model experiments.

The harness is the "controllable gray box" around a local model: it defines
the action protocol, executes tool actions safely, and feeds results back.
"""

from __future__ import annotations

import json
import os
import re
import fnmatch
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.task_suite import (
    ALL_TASKS,
    DEV_TASKS,
    EVAL_TASKS,
    TRAIN_TASKS,
    expected_answer,
    extra_files,
    task_file_payload,
)

SYSTEM = (
    "You are an agent with tools in a filesystem. While you need information, "
    "respond with a tool action as JSON: {\"action\":\"read\",\"path\":\"...\"} "
    "or {\"action\":\"find\",\"pattern\":\"...\"}. You will receive the tool "
    "result next. Only when you have read the needed file, respond with the "
    "final answer as plain JSON {\"task_id\":\"...\",\"status\":\"...\","
    "\"failure_count\":N}. Harness policy: if a requested file is missing, "
    "search with find pattern task-*.json, then read the unique file found. "
    "Never retry the missing path."
)

# Historical totals are preserved by task_suite for tasks 1-5; EXPECTED is the
# backward-compatible failure-count map used by older callers.
EXPECTED = {t: expected_answer(t)["failure_count"] for t in ALL_TASKS}
MAX_TURNS = 6
MAX_NEW = 96
MAX_LEN = 768


def device_and_dtype():
    if os.environ.get("LOCAL_MODEL_DEVICE") == "cpu":
        torch.set_num_threads(32)
        return "cpu", torch.float32
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    torch.set_num_threads(32)
    return "cpu", torch.float32


def load_model(model_path: str, adapter_path: str | None = None, device: str = "cpu", dtype=torch.float32):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if os.environ.get("LOCAL_MODEL_FP32") == "1":
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()
    return model, tokenizer


def parse_output(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None, None
    if data.get("action"):
        return "tool", data
    if "task_id" in data and "failure_count" in data:
        return "answer", data
    return None, None


def execute(ws: Path, action: dict):
    kind = action.get("action")
    if kind == "read":
        target = ws / action.get("path", "")
        existed = target.exists()
        return (target.read_text().strip() if existed else "ERROR: file not found"), existed
    if kind == "find":
        pattern = action.get("pattern", "*")
        names = sorted(
            p.name for p in ws.iterdir() if p.is_file() and fnmatch.fnmatch(p.name, pattern)
        )
        return (", ".join(names) if names else "ERROR: no match"), bool(names)
    return "ERROR: unknown action", False


def setup_task(ws_root: Path, task_index: int) -> Path:
    ws = ws_root / f"task-{task_index}"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / f"task-{task_index}.json").write_text(
        json.dumps(task_file_payload(task_index))
    )
    for name, body in extra_files(task_index).items():
        (ws / name).write_text(body)
    return ws


def user_prompt(task_index: int) -> str:
    return (
        f"Start by reading missing-{task_index}.json. Then answer with JSON fields "
        f"task_id, status, failure_count. status must be exactly SUCCESS once you "
        f"have read the task file; failure_count is the number of records with "
        f"status exactly FAILURE inside that file. Ignore all other statuses."
    )


def verify(data: dict, task_index: int) -> bool:
    return data == expected_answer(task_index)


def rollout(model, tokenizer, ws_root: Path, task_index: int, *, sample: bool, device: str = "cpu", temperature: float = 0.4, top_p: float = 0.9):
    ws = setup_task(ws_root, task_index)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_prompt(task_index)},
    ]
    recovered = False
    for _ in range(MAX_TURNS):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        kwargs = dict(max_new_tokens=MAX_NEW, pad_token_id=tokenizer.eos_token_id)
        if sample:
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            kwargs.update(do_sample=False)
        with torch.no_grad():
            out = model.generate(**inputs, **kwargs)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        kind, data = parse_output(text)
        if kind is None:
            return messages, 0.0, "PARSE_FAIL"
        if kind == "answer":
            ok = verify(data, task_index)
            messages.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
            reward = 1.0 if ok else (0.5 if recovered else 0.0)
            return messages, reward, "ANSWER"
        result, existed = execute(ws, data)
        if data.get("action") == "read" and existed:
            recovered = True
        messages.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
        messages.append({"role": "user", "content": f"Tool result: {result}"})
    return messages, 0.0, "NO_TERMINAL"


def render_and_mask(tokenizer, messages, max_len: int = MAX_LEN):
    """Tokenize each message piece separately and concatenate.

    Concatenation guarantees the trainable mask aligns exactly with assistant
    content, avoiding BPE merge drift across piece boundaries.
    """

    input_ids: list[int] = []
    trainable: list[bool] = []
    for message in messages:
        piece = (
            "<|im_start|>" + message["role"] + "\n" + message["content"] + "<|im_end|>\n"
        )
        toks = tokenizer(
            piece, add_special_tokens=False, truncation=True, max_length=max_len
        )["input_ids"]
        input_ids.extend(toks)
        trainable.extend([message["role"] == "assistant"] * len(toks))
        if len(input_ids) >= max_len:
            break
    input_ids = input_ids[:max_len]
    trainable = trainable[:max_len]
    positions = [t for t in range(len(input_ids) - 1) if trainable[t + 1]]
    return input_ids, positions, trainable
