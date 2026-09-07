#!/usr/bin/env python3
"""Fast LoRA SFT for Qwen2.5-Coder-14B on an APPS token-level package.

Speed improvements over the naive per-example loop:
- SDPA attention (A6000/Ampere) instead of eager;
- length-bucketed dynamic batching: examples are sorted by length and grouped
  so each batch stays under a token budget, then padded (attention_mask);
- batch order reshuffled per epoch with a fixed seed.

Consumes apps-token-sft/v1 packages (sft_tokens.jsonl).  Prompt tokens are
masked with -100; loss only on completion positions.

Run AFTER stopping the inference server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def _verify_package(package: Path) -> list[dict]:
    manifest = json.loads((package / "package-manifest.json").read_text())
    data_path = package / "sft_tokens.jsonl"
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    expected = manifest["files"]["sft_tokens.jsonl"]
    if digest != expected:
        raise SystemExit(f"package checksum mismatch: {digest} != manifest {expected}")
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    if len(rows) != manifest.get("examples"):
        raise SystemExit(f"row count {len(rows)} != manifest {manifest.get('examples')}")
    print(
        f"[verify] package OK: {len(rows)} rows, tasks={manifest.get('tasks')}, "
        f"checksum {digest[:16]}...",
        flush=True,
    )
    return rows


def load_examples(rows: list[dict], max_len: int) -> list[tuple[list[int], list[int]]]:
    out, dropped = [], 0
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
        f"[data] trainable rows: {len(out)} (dropped {dropped}), "
        f"mean len {sum(len(x[0]) for x in out) / max(1, len(out)):.0f} tok",
        flush=True,
    )
    return out


def make_batches(
    examples: list[tuple[list[int], list[int]]],
    token_budget: int,
) -> list[list[int]]:
    """Sort by length, greedily pack into batches under the padded-token budget."""
    order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]))
    batches: list[list[int]] = []
    current: list[int] = []
    for i in order:
        n = len(examples[i][0])
        width = max(n, *(len(examples[j][0]) for j in current)) if current else n
        if current and width * (len(current) + 1) > token_budget:
            batches.append(current)
            current = [i]
        else:
            current.append(i)
    if current:
        batches.append(current)
    return batches


def collate(
    examples: list[tuple[list[int], list[int]]],
    indices: list[int],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seqs = [examples[i][0] for i in indices]
    labels = [examples[i][1] for i in indices]
    max_len = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    label_t = torch.full((len(seqs), max_len), -100, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for r, (s, l) in enumerate(zip(seqs, labels)):
        input_ids[r, : len(s)] = torch.tensor(s)
        label_t[r, : len(l)] = torch.tensor(l)
        mask[r, : len(s)] = 1
    return input_ids, label_t, mask


def chunked_completion_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    label_t: torch.Tensor,
    mask: torch.Tensor,
    chunk: int = 2048,
) -> torch.Tensor:
    """CE only on completion positions, computed in fp32 chunks to avoid
    materializing a [B*T, V] fp32 softmax (which OOMs a 48G card at 14B)."""
    out = model(input_ids=input_ids, attention_mask=mask)
    logits = out.logits  # [B, T, V] bf16
    shift_logits = logits[:, :-1, :]
    shift_labels = label_t[:, 1:]
    sel = shift_labels != -100
    picked = shift_logits[sel]      # [N, V] bf16
    targets = shift_labels[sel]     # [N]
    total = picked.shape[0]
    if total == 0:
        raise RuntimeError("batch has no completion tokens")
    loss_sum = picked.new_zeros((), dtype=torch.float32)
    for start in range(0, total, chunk):
        piece = picked[start : start + chunk].float()
        tgt = targets[start : start + chunk]
        loss_sum = loss_sum + torch.nn.functional.cross_entropy(
            piece, tgt, reduction="sum"
        )
    return loss_sum / total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--token-budget", type=int, default=12288,
                    help="max padded tokens per batch")
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--resume-adapter", default=None,
                    help="LoRA checkpoint to continue from (weights only; optimizer resets)")
    ap.add_argument("--start-epoch", type=int, default=0,
                   help="epoch index to start from when resuming")
    args = ap.parse_args()

    print(f"[start] package={args.package} output={args.output} model={args.model}", flush=True)
    rows = _verify_package(Path(args.package))
    examples = load_examples(rows, args.max_len)
    if not examples:
        raise SystemExit("no trainable rows")

    batches = make_batches(examples, args.token_budget)
    batch_padded = [max(len(examples[i][0]) for i in b) * len(b) for b in batches]
    print(
        f"[batches] {len(batches)} batches (token budget {args.token_budget}), "
        f"mean padded/batch {sum(batch_padded) / len(batches):.0f} tok, "
        f"max {max(batch_padded)} tok; total_steps={len(batches) * args.epochs}",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    attn_impl = os.environ.get("ATTN_IMPL", "sdpa")
    print("[model] loading base model (bf16, sdpa)...", flush=True)
    t0 = time.monotonic()
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )
    print(f"[model] loaded in {time.monotonic() - t0:.1f}s", flush=True)
    if args.resume_adapter:
        import peft
        print(f"[model] resuming LoRA weights from {args.resume_adapter}", flush=True)
        model = peft.PeftModel.from_pretrained(base, args.resume_adapter, is_trainable=True)
    else:
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
    print(f"[model] trainable params: {trainable:.1f}M", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr)
    total_steps = len(batches) * args.epochs
    step = 0
    run_start = time.monotonic()
    tokens_seen = 0
    for epoch in range(args.start_epoch, args.epochs):
        rng = random.Random(args.seed + epoch)
        order = list(range(len(batches)))
        rng.shuffle(order)
        epoch_loss = 0.0
        epoch_weighted = 0.0
        epoch_examples = 0
        epoch_tokens = 0
        epoch_weighted = 0.0
        epoch_start = time.monotonic()
        for bi, b_idx in enumerate(order):
            indices = batches[b_idx]
            input_ids, label_t, mask = collate(examples, indices, tok.pad_token_id)
            input_ids, label_t, mask = (
                input_ids.cuda(), label_t.cuda(), mask.cuda(),
            )
            loss = chunked_completion_loss(model, input_ids, label_t, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            real_tokens = int(mask.sum())
            value = loss.item()
            epoch_loss += value
            epoch_weighted += value * len(indices)
            epoch_examples = epoch_examples + len(indices)  # type: ignore[name-defined]
            epoch_tokens += real_tokens
            tokens_seen += real_tokens
            step += 1
            elapsed = time.monotonic() - run_start
            if step % args.log_every == 0 or bi == 0:
                eta = elapsed / step * (total_steps - step)
                tps = tokens_seen / elapsed
                print(
                    f"[train] epoch{epoch} batch {bi + 1}/{len(batches)} "
                    f"(global {step}/{total_steps}, bs={len(indices)}, "
                    f"padded={input_ids.numel()}) loss={value:.4f} "
                    f"epoch_mean={epoch_weighted / max(1, epoch_examples):.4f} "
                    f"{tps:.0f} tok/s eta={eta / 60:.0f}min",
                    flush=True,
                )
            del input_ids, label_t, mask
            torch.cuda.empty_cache()
        took = time.monotonic() - epoch_start
        print(
            f"[train] === epoch {epoch} done: mean loss {epoch_weighted / max(1, epoch_examples):.4f}, "
            f"{epoch_tokens} tokens, took {took / 60:.1f}min ===",
            flush=True,
        )
        ckpt = Path(args.output) / f"epoch{epoch}"
        print(f"[save] checkpoint -> {ckpt}", flush=True)
        model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)

    final = Path(args.output)
    model.save_pretrained(final)
    tok.save_pretrained(final)
    total = time.monotonic() - run_start
    record = {
        "package": args.package,
        "examples": len(examples),
        "batches_per_epoch": len(batches),
        "token_budget": args.token_budget,
        "epochs": args.epochs,
        "lr": args.lr,
        "rank": args.rank,
        "attn": attn_impl,
        "trainable_params_million": trainable,
        "total_time_minutes": total / 60,
        "adapter_output": str(final),
    }
    (final / "train-record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"[done] total {total / 60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
