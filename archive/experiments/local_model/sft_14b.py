"""ARCHIVED: LoRA SFT for the 14B student on trajectories collected through the REAL Pi
harness (datagen_14b.py's sft_tokens.jsonl: server-logged prompt_ids/completion_ids).

Training on the exact token sequences the served model produced guarantees the
train/inference format matches (same chat template + tools). Prompt tokens are
masked (-100); loss is computed only on completion positions, selecting those
positions' logits before the float32 softmax to avoid OOM on long multi-turn ctx.

Run AFTER stopping the inference server (14B train alone fits in 40G).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("LOCAL_MODEL", "/root/rivermind-data/models/qwen2.5-coder-14b-instruct")
MAX_LEN = 4096  # cap total sequence length; datagen trajectories are short (3-4 turns)


def load_examples(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        p, c = d["prompt_ids"], d["completion_ids"]
        if not p or not c:
            continue
        ids = p + c
        if len(ids) > MAX_LEN:
            # keep the tail (recent context + completion); drop the far head
            overflow = len(ids) - MAX_LEN
            p = p[overflow:] if overflow < len(p) else []
            ids = p + c
            if not p:
                continue
        labels = [-100] * len(p) + list(c)
        rows.append((ids, labels))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="sft_tokens.jsonl from datagen_14b")
    ap.add_argument("--output", required=True, help="output LoRA adapter dir")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    args = ap.parse_args()

    device = "cuda"
    rows = load_examples(Path(args.data))
    print(f"[sft] loaded {len(rows)} training turns from {args.data}", flush=True)
    if not rows:
        raise SystemExit("no training rows")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager"
    )
    model = get_peft_model(
        base,
        LoraConfig(
            r=args.rank, lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        ),
    )
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    # LoRA + grad-checkpointing: input embeddings must require grad, otherwise
    # the checkpointed forward produces "None of the inputs have
    # requires_grad=True" -> logits detach -> loss.backward() fails.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    # .to(device) AFTER the require-grads hook: earlier this was called before
    # the hook and the embedding's requires_grad got reset, detaching logits.
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    print(f"[sft] trainable params: {sum(p.numel() for p in params)/1e6:.1f}M", flush=True)

    step = 0
    for epoch in range(args.epochs):
        tot = 0.0
        for ids, labels in rows:
            input_ids = torch.tensor([ids], device=device)
            labels_t = torch.tensor([labels], device=device)  # -100 masks prompt
            out = model(input_ids=input_ids, labels=labels_t)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            tot += loss.item()
            step += 1
            if step % 20 == 0:
                print(f"[sft] epoch{epoch} step{step} loss={tot / (step or 1):.4f}", flush=True)
            del out
            torch.cuda.empty_cache()
        print(f"[sft] === epoch {epoch} mean loss {tot/len(rows):.4f} ===", flush=True)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tok.save_pretrained(args.output)
    print(f"[sft] saved adapter -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
