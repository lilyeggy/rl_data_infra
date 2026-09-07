#!/usr/bin/env python
"""ARCHIVED: Day-01 two-Blackwell Megatron backward smoke test.

真实流程：加载已转换的 torch_dist 权重 -> BF16 forward -> backward ->
optimizer step -> 记录 initial loss / grad norm / lr / peak mem / 参数 checksum
(before/after) / step duration -> save checkpoint -> mutate -> reload -> 校验 checksum。

用法（bash 里 source 模型脚本后追加 megatron 参数）：
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=.../Megatron-LM .venv/bin/python scripts/smoke_megatron_step.py \
      ${MODEL_ARGS[@]} --seq-length 512 --bf16 ... --save X --load X
"""
import hashlib
import json
import os
import time

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "12355")
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
os.environ.setdefault("WORLD_SIZE", str(WORLD_SIZE))
os.environ.setdefault("RANK", str(RANK))
os.environ.setdefault("LOCAL_RANK", str(LOCAL_RANK))

import torch
import torch.distributed as dist

from megatron.core.enums import ModelType
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.training import get_model, get_megatron_optimizer_config
from megatron.core.optimizer import get_megatron_optimizer

import slime_plugins.mbridge  # noqa: F401  (AutoBridge)
from slime.backends.megatron_utils.arguments import set_default_megatron_args
from slime.backends.megatron_utils.initialize import init
from slime.backends.megatron_utils.model_provider import get_model_provider_func


def rank0_print(*args, **kwargs):
    if RANK == 0:
        print(*args, **kwargs)


def param_checksum(model, tag):
    """Streaming sha256 over all local-shard params' fp32 cpu bytes (rank0 only)."""
    if RANK != 0:
        return None
    h = hashlib.sha256()
    n = 0
    for name, p in model.named_parameters():
        data = p.detach().float().cpu().numpy()
        h.update(data.tobytes())
        n += 1
    digest = h.hexdigest()
    rank0_print(f"[smoke] checksum({tag}): {digest} over {n} params (rank0 shard)", flush=True)
    return digest


def master_grad_norm(model, optimizer):
    """梯度范数：优先取模型参数上的 grad（bf16 路径），回退 fp32 master grads。"""
    total_sq = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.detach().float().pow(2).sum().item()
            count += 1
    if count == 0:
        master = getattr(optimizer, "fp32_from_fp16_params", None) or getattr(
            optimizer, "fp32_from_fp32_params", None
        )
        if master is not None:
            for group in master:
                for p in group:
                    if p.grad is not None:
                        total_sq += p.grad.detach().float().pow(2).sum().item()
                        count += 1
    norm = total_sq ** 0.5
    print(f"[smoke] grad norm={norm:.6f} over {count} tensors", flush=True)
    return norm


def main():
    args = parse_args()
    args = set_default_megatron_args(args)
    args.megatron_to_hf_mode = "raw"  # 训练/加载路径无需 HF bridge 转换模式
    validate_args(args)

    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        device_id=torch.device(f"cuda:{LOCAL_RANK}"),
    )
    init(args)

    seq_len = args.seq_length
    micro_batch = args.micro_batch_size

    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=True)
    rank0_print(
        f"[smoke] model built, params={sum(p.numel() for p in model[0].parameters())/1e9:.3f}B (rank0 shard)",
        flush=True,
    )

    # 载入转换后的权重
    load_checkpoint(model, None, None)
    checksum_a = param_checksum(model[0], "after_load(A)")

    optimizer_config, config_overrides = get_megatron_optimizer_config(args)
    optimizer = get_megatron_optimizer(
        optimizer_config, model, config_overrides=config_overrides
    )

    torch.manual_seed(1234)
    device = torch.cuda.current_device()
    input_ids = torch.randint(0, args.vocab_size, (micro_batch, seq_len), dtype=torch.long, device=device)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(micro_batch, -1)
    labels = input_ids.clone()
    loss_mask = torch.ones((micro_batch, seq_len), dtype=torch.float32, device=device)

    # one real step
    t0 = time.time()
    optimizer.zero_grad()
    loss = model[0](input_ids=input_ids, position_ids=position_ids, attention_mask=None, labels=labels, loss_mask=loss_mask)
    loss_reduced = loss.float().mean()
    initial_loss = float(loss_reduced.detach())
    rank0_print(f"[smoke] initial loss={initial_loss:.6f}", flush=True)
    loss_reduced.backward()
    torch.cuda.synchronize()
    # megatron-core 0.16: step() 返回 (update_successful, grad_norm, num_zeros_in_grad)
    step_result = optimizer.step()
    if isinstance(step_result, tuple) and len(step_result) >= 2:
        update_successful, grad_norm = step_result[0], float(step_result[1])
        rank0_print(f"[smoke] grad norm={grad_norm:.6f} (from optimizer.step)", flush=True)
    else:
        grad_norm = float(optimizer.get_grad_norm())
        rank0_print(f"[smoke] grad norm={grad_norm:.6f} (get_grad_norm)", flush=True)
    torch.cuda.synchronize()
    step_duration = time.time() - t0

    lr = optimizer.param_groups[0]["lr"] if hasattr(optimizer, "param_groups") else args.lr
    checksum_b = param_checksum(model[0], "after_step(B)")
    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
    peak_reserved = torch.cuda.max_memory_reserved() / 1e9

    # checkpoint save -> mutate -> reload -> verify
    save_checkpoint(1, model, optimizer, None, 0)
    dist.barrier()
    if RANK == 0:
        with torch.no_grad():
            for p in model[0].parameters():
                p.add_(torch.randn_like(p, dtype=p.dtype) * 0.01)
                break
    checksum_mutated = param_checksum(model[0], "mutated(M)")
    load_checkpoint(model, None, None, load_arg="save")  # 从 save 目录重载，验证保存一致性
    checksum_c = param_checksum(model[0], "after_reload(C)")

    if RANK == 0:
        result = {
            "seq_length": seq_len,
            "micro_batch_size": micro_batch,
            "dtype": "bf16" if args.bf16 else "fp16" if args.fp16 else "fp32",
            "initial_loss": initial_loss,
            "grad_norm": grad_norm,
            "learning_rate": lr,
            "step_duration_s": round(step_duration, 3),
            "peak_allocated_gb": round(peak_alloc, 2),
            "peak_reserved_gb": round(peak_reserved, 2),
            "checksum_after_load_A": checksum_a,
            "checksum_after_step_B": checksum_b,
            "checksum_mutated_M": checksum_mutated,
            "checksum_after_reload_C": checksum_c,
            "checksum_changed_after_step": checksum_a != checksum_b,
            "checksum_restored_after_reload": checksum_b == checksum_c,
            "recompute": f"{args.recompute_granularity}/{args.recompute_method}",
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "tensor_model_parallel_size": args.tensor_model_parallel_size,
            "save_dir": args.save,
        }
        print("\n=== SMOKE_RESULT ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
