#!/usr/bin/env python3
"""LoRA SFT for Qwen2.5-Coder-14B on an APPS token-level package.

Consumes apps-token-sft/v1 packages (sft_tokens.jsonl with prompt_ids /
completion_ids produced by build_apps_sft_package.py).  Trains only on
completion positions; prompt tokens are masked with -100.

Verbose logging: package verification, per-step loss, running mean, ETA,
GPU memory, per-epoch summaries, periodic checkpoint saves.

Run AFTER stopping the inference server: 14B LoRA training alone fits in 40G.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def _verify_package(package: Path) -> tuple[list[dict], dict]:
    """Reload the package and re-verify its content checksums (fail-closed)."""
    manifest = json.loads((package / "package-manifest.json").read_text())
    data_path = package / "sft_tokens.jsonl"
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    expected = manifest["files"]["sft_tokens.jsonl"]
    if digest != expected:
        raise SystemExit(
            f"package checksum mismatch: {data_path.name} {digest} != manifest {expected}"
        )
    print(
        f"[verify] sft_tokens.jsonl checksum OK ({digest[:16]}...), "
        f"manifest examples={manifest.get('examples')} tasks={manifest.get('tasks')} "
        f"rejected_eligibility={manifest.get('rejected_eligibility')}",
        flush=True,
    )
    rows = []
    for line in data_path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if len(rows) != manifest.get("examples"):
        raise SystemExit(f"row count {len(rows)} != manifest examples {manifest.get('examples')}")
    print(f"[verify] loaded {len(rows)} rows, manifest count matches", flush=True)
    return rows, manifest


def _gpu_mem() -> str:
    try:
        used = torch.cuda.memory_allocated() / 2**30
        reserved = torch.cuda.memory_reserved() / 2**30
        return f"gpu alloc={used:.1f}G reserved={reserved:.1f}G"
    except Exception:
        return "gpu n/a"


def load_examples(rows: list[dict], max_len: int) -> list[tuple[list[int], list[int]]]:
    out = []
    dropped = 0
    for d in rows:
        p, c = list(d["prompt_ids"]), list(d["completion_ids"])
        if not p or not c:
            dropped += 1
            continue
        if len(p) + len(c) > max_len:
            overflow = len(p) + len(c) - max_len
            p = p[overflow:] if overflow < len(p) else []
            if not p:
                dropped += 1
                continue
        out.append((p + c, [-100] * len(p) + list(c)))
    print(
        f"[data] trainable rows: {len(out)} (dropped {dropped} empty/overlong), "
        f"mean length {sum(len(x[0]) for x in out) / max(1, len(out)):.0f} tokens",
        flush=True,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", required=True, help="apps-token-sft/v1 package dir")
    ap.add_argument("--output", required=True, help="output LoRA adapter dir")
    ap.add_argument("--model", required=True, help="base model path")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--log-every", type=int, default=1, help="print every N steps")
    ap.add_argument("--save-every-epoch", action="store_true", default=True)
    args = ap.parse_args()

    package = Path(args.package)
    print(f"[start] package={package} output={args.output} model={args.model}", flush=True)
    rows, _manifest = _verify_package(package)
    examples = load_examples(rows, args.max_len)
    if not examples:
        raise SystemExit("no trainable rows")
    total_steps = args.epochs * len(examples)
    print(f"[plan] epochs={args.epochs} steps/epoch={len(examples)} total_steps={total_steps}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print("[model] loading base model (bf16)...", flush=True)
    t0 = time.monotonic()
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager"
    )
    print(f"[model] loaded in {time.monotonic() - t0:.1f}s", flush=True)
    model = get_peft_model(
        base,
        LoraConfig(
            r=args.rank,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = model.to("cuda")
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params) / 1e6
    print(f"[model] trainable params: {trainable:.1f}M  {_gpu_mem()}", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr)
    step = 0
    run_start = time.monotonic()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_start = time.monotonic()
        for idx, (ids, labels) in enumerate(examples):
            input_ids = torch.tensor([ids], device="cuda")
            labels_t = torch.tensor([labels], device="cuda")
            out = model(input_ids=input_ids, labels=labels_t)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            value = loss.item()
            epoch_loss += value
            step += 1
            if step % args.log_every == 0 or idx == 0:
                elapsed = time.monotonic() - run_start
                eta = elapsed / step * (total_steps - step)
                print(
                    f"[train] epoch{epoch} step {idx + 1}/{len(examples)} "
                    f"(global {step}/{total_steps}) loss={value:.4f} "
                    f"epoch_mean={epoch_loss / (idx + 1):.4f} "
                    f"eta={eta / 60:.0f}min {_gpu_mem()}",
                    flush=True,
                )
            del out
        mean = epoch_loss / len(examples)
        took = time.monotonic() - epoch_start
        print(
            f"[train] === epoch {epoch} done: mean loss {mean:.4f}, "
            f"took {took / 60:.1f}min ===",
            flush=True,
        )
        if args.save_every_epoch:
            ckpt = Path(args.output) / f"epoch{epoch}"
            print(f"[save] checkpoint -> {ckpt}", flush=True)
            model.save_pretrained(ckpt)
            tok.save_pretrained(ckpt)

    final = Path(args.output)
    print(f"[save] final adapter -> {final}", flush=True)
    model.save_pretrained(final)
    tok.save_pretrained(final)
    total = time.monotonic() - run_start
    record = {
        "package": str(package),
        "examples": len(examples),
        "epochs": args.epochs,
        "lr": args.lr,
        "rank": args.rank,
        "final_epoch_mean_losses": "see log",
        "trainable_params_million": trainable,
        "total_time_minutes": total / 60,
        "adapter_output": str(final),
    }
    (final / "train-record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"[done] total {total / 60:.1f}min; record -> {final / 'train-record.json'}", flush=True)


if __name__ == "__main__":
    main()
