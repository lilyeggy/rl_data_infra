#!/usr/bin/env python3
"""Stage E: one native GRPO update on a certified round batch (dual GPU, FSDP2).

Uses verl's own algorithm code -- no reimplementation of the objective:
  * advantages : verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage
  * loss       : verl.trainer.ppo.core_algos.compute_policy_loss (PPO clip)

Flow per round:
  1. Load base + previous adapter; FSDP2-shard over both GPUs (LoRA trainable).
  2. Pre-update forward over the assembled sequences -> training-side logprobs.
     Compare them with the rollout logprobs recorded by vLLM on the SAME
     canonical token stream (closeout E: <=0.05 nat mean, <=0.5 nat P99).
  3. GRPO advantages -> clipped policy loss -> one AdamW step at lr=1e-6.
  4. Synchronous LoRA-only checkpoint (DTensor gather, no async saver).

The batch is tiny (a few thousand response tokens), so every rank evaluates the
whole batch: with FSDP2 the forward is replicated anyway, the loss is therefore
identical on all ranks, and gradient averaging across ranks becomes a no-op.
That keeps the objective exactly "one token-mean GRPO objective over the batch"
instead of an approximation of it.

The consumed batch is never replayed, and no half episode is ever resumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True, help="previous-round adapter")
    parser.add_argument("--round-json", required=True, help="stage_e_assemble output")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--next-generation", required=True, help="e.g. P1")
    parser.add_argument("--group-id", default="mbpp118-group")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--max-seq-len", type=int, default=6144)
    args = parser.parse_args()

    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    from verl.trainer.ppo.core_algos import (
        compute_grpo_outcome_advantage,
        compute_policy_loss,
    )

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    torch.manual_seed(0)

    out_dir = Path(args.output_dir)
    t0 = time.time()
    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[rank{rank} +{time.time() - t0:7.1f}s] {msg}"
        if rank == 0:
            print(line, flush=True)
        log_lines.append(line)

    round_data = json.loads(Path(args.round_json).read_text())
    episodes = [e for e in round_data["episodes"] if "error" not in e]
    if len(episodes) < 2:
        log(f"INSUFFICIENT-EPISODES assembled={len(episodes)}")
        return 10
    rewards = {e["episode_id"]: float(e["reward"]) for e in episodes}
    if len(set(rewards.values())) < 2:
        log(f"NO-REWARD-VARIANCE rewards={sorted(set(rewards.values()))}")
        return 11
    log(f"batch: {len(episodes)} episodes rewards={sorted(rewards.values())}")

    from peft import PeftModel
    from torch.distributed.fsdp import fully_shard
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    assert hasattr(base, "model") and hasattr(base.model, "layers"), "unexpected model shape"
    for layer in base.model.layers:
        fully_shard(layer)
    fully_shard(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log(f"fsdp2 world={world} trainable={trainable/1e6:.2f}M frozen={frozen/1e9:.2f}B")

    def response_logprobs(episode) -> torch.Tensor:
        """Training-side logprob of each response token, aligned to response_ids."""
        prompt_ids = list(episode["prompt_ids"])
        response_ids = list(episode["response_ids"])
        total = len(prompt_ids) + len(response_ids)
        if total > args.max_seq_len:
            raise ValueError(f"sequence {total} exceeds --max-seq-len {args.max_seq_len}")
        input_ids = torch.tensor([prompt_ids + response_ids], device="cuda", dtype=torch.long)
        outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
        start = len(prompt_ids) - 1
        end = start + len(response_ids)
        resp_logits = outputs.logits[0, start:end, :].float()
        targets = torch.tensor(response_ids, device="cuda", dtype=torch.long)
        return -F.cross_entropy(resp_logits, targets, reduction="none")

    # ---- phase 1: pre-update (old) training-side logprobs -------------------
    model.eval()
    old_logprobs: list[torch.Tensor] = []
    with torch.no_grad():
        for episode in episodes:
            old_logprobs.append(response_logprobs(episode).float().cpu())
    log("pre-update training-side logprobs computed for the whole batch")

    # ---- logprob comparison: rollout (vLLM) vs training side ----------------
    diffs = []
    per_episode = []
    for old_t, episode in zip(old_logprobs, episodes):
        mask = torch.tensor(episode["response_mask"], dtype=torch.float32)
        rollout = torch.tensor(episode["rollout_logprobs"], dtype=torch.float32)
        d = (old_t - rollout).abs()[mask > 0]
        diffs.append(d)
        per_episode.append({
            "episode_id": episode["episode_id"],
            "masked_tokens": int(mask.sum()),
            "mean_abs_nat": float(d.mean()),
            "max_abs_nat": float(d.max()),
        })
    all_d = torch.cat(diffs)
    mean_abs = float(all_d.mean())
    p99 = float(torch.quantile(all_d, 0.99))
    comparison = {
        "masked_tokens": int(all_d.numel()),
        "mean_abs_nat": mean_abs,
        "p99_abs_nat": p99,
        "quantiles_abs_nat": {str(p): float(torch.quantile(all_d, p)) for p in (0.5, 0.9, 0.95, 0.99)},
        "threshold_mean_nat": 0.05,
        "threshold_p99_nat": 0.5,
        "within_threshold": bool(mean_abs <= 0.05 and p99 <= 0.5),
        "per_episode": per_episode,
    }
    log(f"logprob-comparison mean_abs={mean_abs:.5f} p99={p99:.5f} "
        f"within_threshold={comparison['within_threshold']}")

    # ---- GRPO advantages via verl's own implementation ----------------------
    max_resp = max(len(e["response_ids"]) for e in episodes)
    bs = len(episodes)
    token_level_rewards = torch.zeros(bs, max_resp)
    response_mask = torch.zeros(bs, max_resp)
    for i, episode in enumerate(episodes):
        r = len(episode["response_ids"])
        token_level_rewards[i, r - 1] = float(episode["reward"])
        response_mask[i, :r] = torch.tensor(episode["response_mask"], dtype=torch.float32)
    advantages, _returns = compute_grpo_outcome_advantage(
        token_level_rewards, response_mask, np.array([args.group_id] * bs)
    )
    masked_adv = advantages[response_mask > 0]
    log(f"grpo advantages ok masked_mean={masked_adv.mean():.4f} masked_std={masked_adv.std():.4f}")

    # ---- phase 2: one clipped-policy update round ---------------------------
    total_masked = float(response_mask.sum().item())
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )
    optimizer.zero_grad()
    model.train()
    loss_sum = 0.0
    clipfrac_acc: list[float] = []
    ppo_kl_acc: list[float] = []
    for i, episode in enumerate(episodes):
        logprob = response_logprobs(episode)
        r = len(episode["response_ids"])
        mask = response_mask[i, :r].to("cuda")
        adv = advantages[i, :r].to("cuda")
        old_local = old_logprobs[i].to("cuda")
        pg_loss, clipfrac, ppo_kl, _lower = compute_policy_loss(
            old_local, logprob, adv, mask,
            cliprange=args.clip_ratio, loss_agg_mode="token-mean",
        )
        weight = float(mask.sum().item()) / total_masked
        (pg_loss * weight).backward()
        loss_sum += float(pg_loss.item()) * weight
        clipfrac_acc.append(float(clipfrac))
        ppo_kl_acc.append(float(ppo_kl))

    grad_norm = torch.sqrt(sum(
        (p.grad.detach().float() ** 2).sum()
        for p in model.parameters() if p.grad is not None
    ))
    log(f"pre-step loss={loss_sum:.6f} grad_norm={float(grad_norm):.6f}")

    before = torch.tensor(
        [p.detach().float().sum().item() for _, p in model.named_parameters() if p.requires_grad],
        device="cuda",
    )
    optimizer.step()
    torch.cuda.synchronize()
    optimizer.zero_grad()
    after = torch.tensor(
        [p.detach().float().sum().item() for _, p in model.named_parameters() if p.requires_grad],
        device="cuda",
    )
    changed_local = int(((after - before).abs() > 0).sum().item())
    drift_local = float((after - before).abs().sum().item())
    changed = torch.tensor([changed_local], device="cuda")
    drift = torch.tensor([drift_local], device="cuda")
    dist.all_reduce(changed, op=dist.ReduceOp.SUM)
    dist.all_reduce(drift, op=dist.ReduceOp.SUM)
    peak = torch.cuda.max_memory_allocated() / 1e9
    log(f"native-update ok changed={int(changed.item())} drift={float(drift.item()):.6f} peak_GB={peak:.1f}")
    if changed_local == 0:
        log("FATAL: no parameter changed after the update")
        return 12

    # ---- synchronous checkpoint (LoRA-only, DTensor gathered) ---------------
    def _full_cpu(t: torch.Tensor) -> torch.Tensor:
        full = t.full_tensor() if hasattr(t, "full_tensor") else t
        return full.detach().to("cpu")

    lora_state = {n: _full_cpu(p) for n, p in model.named_parameters() if p.requires_grad}
    result: dict = {
        "tag": "stage-e-grpo-update",
        "round_index": round_data["round_index"],
        "policy_generation_from": round_data["policy"]["generation"],
        "policy_generation_to": args.next_generation,
        "policy_checksum_from": round_data["policy"]["checksum"],
        "learning_rate": args.learning_rate,
        "clip_ratio": args.clip_ratio,
        "loss_agg_mode": "token-mean",
        "episodes": [e["episode_id"] for e in episodes],
        "rewards": rewards,
        "advantage_stats": {
            "masked_mean": float(masked_adv.mean()),
            "masked_std": float(masked_adv.std()),
            "shape": list(advantages.shape),
        },
        "logprob_comparison": comparison,
        "world_size": world,
        "device": torch.cuda.get_device_name(),
        "trainable_params": trainable,
        "frozen_params": frozen,
        "loss": loss_sum,
        "grad_norm_local": float(grad_norm),
        "changed_tensors_all_ranks": int(changed.item()),
        "drift_sum_all_ranks": float(drift.item()),
        "peak_memory_GB_rank0": peak,
        "clipfrac_mean": sum(clipfrac_acc) / len(clipfrac_acc),
        "ppo_kl_mean": sum(ppo_kl_acc) / len(ppo_kl_acc),
        "previous_adapter_sha256": sha256_file(Path(args.adapter) / "adapter_model.safetensors"),
    }
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        next_dir = out_dir / f"adapter-{args.next_generation.lower()}"
        model.save_pretrained(next_dir, state_dict=lora_state)
        tok.save_pretrained(next_dir)
        sha = sha256_file(next_dir / "adapter_model.safetensors")
        result["next_adapter_sha256"] = sha
        result["next_adapter_dir"] = str(next_dir)
        # Recovery check: reload the saved adapter from disk and re-infer.
        fresh = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to("cuda:0")
        reloaded = PeftModel.from_pretrained(fresh, str(next_dir))
        reloaded.eval()
        with torch.inference_mode():
            gen = reloaded.generate(
                **tok("def add(a, b):", return_tensors="pt").to("cuda:0"),
                max_new_tokens=16, do_sample=False,
            )
        result["reload_reinfer_head"] = tok.decode(gen[0], skip_special_tokens=True)[:160]
        result["elapsed_s"] = time.time() - t0
        (out_dir / "update-result.json").write_text(json.dumps(result, indent=2) + "\n")
        log(f"saved {args.next_generation} sha={sha[:16]} reload-reinfer ok")
    dist.barrier()
    if rank == 0:
        (out_dir / "update.log").write_text("\n".join(log_lines) + "\n")
        print("STAGE-E-UPDATE-PASS", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
