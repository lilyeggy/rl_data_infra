#!/usr/bin/env python3
"""Lightweight, self-contained GRPO LoRA trainer for Agentic RL.

Consumes slime-admission/v1 batches (produced by collect_rl_rollouts.py or
admit-slime CLI). Computes group-relative advantages per task group, calculates
PPO-clip + KL loss strictly over action_mask tokens, and saves updated LoRA
adapters (Policy-vNext).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.contracts._json import canonical_json_bytes, sha256_json


def _verify_admission(admission_path: Path) -> dict[str, Any]:
    """Load and verify slime admission JSON payload."""
    if not admission_path.exists():
        raise SystemExit(f"admission file not found: {admission_path}")
    data = json.loads(admission_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "slime-admission/v1":
        raise SystemExit(f"unexpected schema_version: {data.get('schema_version')}")
    traces = data.get("traces", [])
    if not traces:
        raise SystemExit("no traces found in admission batch")
    return data


def compute_group_advantages(traces: list[dict[str, Any]]) -> dict[str, tuple[float, float, dict[str, float]]]:
    """Compute per-group mean, std, and per-trace advantage.

    Returns:
        dict mapping group_id -> (mean, std, {trajectory_id: advantage})
    """
    by_group = defaultdict(list)
    for t in traces:
        gid = t["group_id"]
        tid = t["trajectory_id"]
        rew = float(t.get("reward", 0.0))
        by_group[gid].append((tid, rew))

    group_stats: dict[str, tuple[float, float, dict[str, float]]] = {}
    for gid, items in by_group.items():
        rewards = [r for _, r in items]
        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        std = math.sqrt(variance)

        adv_map: dict[str, float] = {}
        for tid, r in items:
            if std > 1e-6:
                adv_map[tid] = (r - mean) / (std + 1e-6)
            else:
                adv_map[tid] = 0.0
        group_stats[gid] = (mean, std, adv_map)

    return group_stats


def run_grpo_preflight(admission_data: dict[str, Any]) -> dict[str, Any]:
    """Inspect admission batch, token statistics and advantage spread."""
    traces = admission_data["traces"]
    group_stats = compute_group_advantages(traces)

    total_action_tokens = sum(sum(t.get("loss_mask", [])) for t in traces)
    groups_with_variance = sum(1 for _, std, _ in group_stats.values() if std > 1e-6)

    report = {
        "schema_version": admission_data.get("schema_version"),
        "total_traces": len(traces),
        "total_groups": len(group_stats),
        "groups_with_variance": groups_with_variance,
        "total_action_tokens": total_action_tokens,
        "mean_tokens_per_trace": total_action_tokens / max(1, len(traces)),
        "policy_fingerprint": admission_data.get("policy_fingerprint"),
        "preflight_status": "PASSED" if traces else "FAILED",
    }
    return report


def mock_grpo_train(
    admission_data: dict[str, Any],
    output_dir: Path,
    epochs: int,
    lr: float,
) -> dict[str, Any]:
    """Simulate training step for offline testing and continuous integration."""
    traces = admission_data["traces"]
    group_stats = compute_group_advantages(traces)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute simulated losses
    steps = 0
    total_loss = 0.0
    for t in traces:
        gid = t["group_id"]
        tid = t["trajectory_id"]
        adv = group_stats[gid][2].get(tid, 0.0)
        # Mock loss: smaller loss when advantage aligns
        mock_loss = max(0.01, 0.5 - 0.1 * adv)
        total_loss += mock_loss
        steps += 1

    mean_loss = total_loss / max(1, steps)

    new_policy_fingerprint = hashlib.sha256(
        f"{admission_data.get('policy_fingerprint')}-updated-{time.time()}".encode("utf-8")
    ).hexdigest()

    adapter_config = {
        "base_model": "qwen2.5-coder-14b-instruct",
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "policy_fingerprint": new_policy_fingerprint,
    }
    (output_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2) + "\n")
    (output_dir / "adapter_model.bin").write_bytes(b"MOCK_LORA_WEIGHTS_RL_POLICY_V1")

    train_record = {
        "trainer": "grpo-lora-trainer/v1",
        "total_traces": len(traces),
        "epochs": epochs,
        "learning_rate": lr,
        "final_mean_loss": round(mean_loss, 4),
        "policy_fingerprint_in": admission_data.get("policy_fingerprint"),
        "policy_fingerprint_out": new_policy_fingerprint,
        "adapter_output": str(output_dir),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "train-record.json").write_text(json.dumps(train_record, indent=2) + "\n")
    return train_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-batch", required=True, help="Path to slime-admission.json")
    parser.add_argument("--output", required=True, help="Output directory for updated LoRA adapter")
    parser.add_argument("--model", default="qwen2.5-coder-14b-instruct", help="Base model path")
    parser.add_argument("--base-adapter", default=None, help="Base adapter to initialize policy from")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for policy LoRA")
    parser.add_argument("--clip", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--beta", type=float, default=0.01, help="KL penalty coefficient beta")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--preflight-only", action="store_true", help="Perform preflight verification only")
    parser.add_argument("--mock-training", action="store_true", help="Run simulated training without CUDA")
    args = parser.parse_args()

    admission_data = _verify_admission(Path(args.admission_batch))
    preflight = run_grpo_preflight(admission_data)

    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return

    output_dir = Path(args.output)
    if args.mock_training:
        record = mock_grpo_train(admission_data, output_dir, args.epochs, args.lr)
        print(json.dumps(record, indent=2))
        return

    # Real GPU training branch
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            f"PyTorch / transformers / peft not available: {exc}. Use --mock-training on CPU."
        )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for real model training. Use --mock-training for testing.")

    device = "cuda"
    traces = admission_data["traces"]
    group_stats = compute_group_advantages(traces)

    print(f"[model] loading base model: {args.model}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    if args.base_adapter and Path(args.base_adapter).exists():
        print(f"[model] attaching base adapter: {args.base_adapter}...", flush=True)
        policy = PeftModel.from_pretrained(base, args.base_adapter, is_trainable=True).to(device)
    else:
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy = get_peft_model(base, lora_cfg).to(device)

    policy.train()
    if hasattr(policy, "gradient_checkpointing_enable"):
        policy.gradient_checkpointing_enable()

    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        valid_steps = 0
        for t in traces:
            gid = t["group_id"]
            tid = t["trajectory_id"]
            adv = group_stats[gid][2].get(tid, 0.0)
            if abs(adv) < 1e-6:
                # Zero advantage gives zero gradient; skip forward pass
                continue

            prompt_ids = list(t["prompt_ids"])
            response_ids = list(t["response_ids"])
            loss_mask = list(t["loss_mask"])
            old_logprobs = list(t["response_logprobs"])

            full_ids = prompt_ids + response_ids
            input_tensor = torch.tensor([full_ids], device=device)

            logits = policy(input_ids=input_tensor).logits[0]  # [seq_len, vocab]
            # Slicing response positions
            n_in = len(prompt_ids)
            n_out = len(response_ids)
            resp_logits = logits[n_in - 1 : n_in + n_out - 1]  # [n_out, vocab]
            log_probs = torch.log_softmax(resp_logits.float(), dim=-1)

            target_ids = torch.tensor(response_ids, device=device)
            new_logprobs = log_probs[torch.arange(n_out, device=device), target_ids]

            mask_tensor = torch.tensor(loss_mask, device=device, dtype=torch.float32)
            old_tensor = torch.tensor(old_logprobs, device=device, dtype=torch.float32)

            log_ratio = new_logprobs - old_tensor
            ratio = torch.exp(log_ratio)

            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv
            pg_loss = -torch.min(s1, s2)

            # KL penalty
            kl = torch.exp(-log_ratio) + log_ratio - 1.0
            total_token_loss = pg_loss + args.beta * kl

            masked_loss = (total_token_loss * mask_tensor).sum() / (mask_tensor.sum() + 1e-6)
            masked_loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            epoch_loss += masked_loss.item()
            valid_steps += 1
            print(
                f"  [step {valid_steps}] {tid} (adv={adv:+.4f}): "
                f"mean_ratio={ratio.mean().item():.4f}, "
                f"pg_loss={pg_loss.mean().item():.4f}, "
                f"kl={kl.mean().item():.4f}, "
                f"total_loss={masked_loss.item():.4f}",
                flush=True,
            )

        print(
            f"[train] epoch {epoch + 1}/{args.epochs} completed, "
            f"active steps={valid_steps}, mean loss={epoch_loss / max(1, valid_steps):.4f}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)

    record = {
        "trainer": "grpo-lora-trainer/v1",
        "total_traces": len(traces),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "adapter_output": str(output_dir),
    }
    (output_dir / "train-record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"[done] saved updated policy adapter to {output_dir}")


if __name__ == "__main__":
    main()
