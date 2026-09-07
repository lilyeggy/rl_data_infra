"""ARCHIVED: LoRA SFT for the 14B student, retained only for experiment history."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("LOCAL_MODEL", "/root/rivermind-data/models/qwen2.5-14b-instruct")


def load_rows(data: Path):
    rows = []
    for line in data.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        p, c = d["prompt_ids"], d["completion_ids"]
        if not p or not c:
            continue
        rows.append((p + c, [-100] * len(p) + list(c)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    rows = load_rows(Path(args.data))
    print("samples:", len(rows), flush=True)
    if not rows:
        raise SystemExit("no rows")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager"
    )
    model = get_peft_model(
        base,
        LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        ),
    )
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = model.to("cuda")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    print("trainable:", sum(p.numel() for p in params) / 1e6, "M", flush=True)

    step = 0
    for epoch in range(args.epochs):
        tot = 0.0
        for i, (ids, labels) in enumerate(rows):
            it = torch.tensor([ids], device="cuda")
            lt = torch.tensor([labels], device="cuda")
            out = model(input_ids=it, labels=lt)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            tot += loss.item()
            step += 1
            if step % 20 == 0:
                print("step", step, "loss", round(loss.item(), 4), flush=True)
            del out
            torch.cuda.empty_cache()
        print("=== epoch", epoch, "mean loss", round(tot / len(rows), 4), "===", flush=True)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tok.save_pretrained(args.output)
    print("SFT_SAVED ->", args.output, flush=True)


if __name__ == "__main__":
    main()
