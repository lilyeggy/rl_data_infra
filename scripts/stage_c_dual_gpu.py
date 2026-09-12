#!/usr/bin/env python3
"""Stage C dual-GPU verification (torchrun, 2 procs): verl-stack FSDP LoRA smoke.

Covers closeout Stage C on both RTX PRO 6000:
  1. 14B Base + frozen P0 loads (via verl 0.7.1 HFModelConfig + PEFT).
  2. FSDP2 shards base over 2 GPUs; only LoRA params trainable.
  3. One native optimizer update; adapter weights change numerically.
  4. Checkpoint saves on rank 0; P1 reloads and re-infers.
  5. vLLM LoRA availability recorded honestly (import probe, no fake).

Launch: torchrun --nproc_per_node=2 stage_c_dual_gpu.py --base-model ... \
  --adapter ... --output-dir ... (no CUDA_VISIBLE_DEVICES mask).
Products tagged framework-smoke.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default="def fibonacci(n):")
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == 2, f"dual-GPU run required, got world={world}"
    torch.cuda.set_device(rank)

    out = Path(args.output_dir)
    t0 = time.time()
    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[rank{rank} +{time.time() - t0:7.1f}s] {msg}"
        if rank == 0:
            print(line, flush=True)
        log_lines.append(line)

    log(f"device={torch.cuda.get_device_name()} world={world}")
    import verl

    assert verl.__version__ == "0.7.1", verl.__version__
    from verl.workers.config import HFModelConfig

    model_cfg = HFModelConfig(
        path=args.base_model, lora_rank=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_adapter_path=args.adapter,
    )
    log(f"verl HFModelConfig ok rank={model_cfg.lora_rank} alpha={model_cfg.lora_alpha}")

    # vLLM LoRA availability probe (honest record, never faked).
    vllm_info: dict = {}
    try:
        import vllm  # noqa: F401
        from vllm.lora.request import LoRARequest

        vllm_info = {"importable": True,
                     "version": getattr(vllm, "__version__", "unknown"),
                     "LoRARequest": repr(LoRARequest)[:80]}
    except Exception as exc:  # noqa: BLE001
        vllm_info = {"importable": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    log(f"vllm probe: {vllm_info}")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.distributed.fsdp import fully_shard

    tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    # FSDP2: shard each transformer layer over 2 GPUs.
    assert hasattr(base, "model") and hasattr(base.model, "layers"), "unexpected model shape"
    for layer in base.model.layers:
        fully_shard(layer)
    fully_shard(model)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log(f"fsdp2-sharded params: trainable={trainable} ({trainable/1e6:.2f}M) frozen={frozen}")
    assert trainable > 0 and frozen > trainable

    before_local = torch.tensor(
        [p.detach().float().sum().item() for _, p in model.named_parameters() if p.requires_grad],
        device="cuda",
    )
    inputs = tok(args.prompt, return_tensors="pt").to("cuda")
    labels = inputs["input_ids"].clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-6)
    optimizer.zero_grad()
    loss = model(input_ids=inputs["input_ids"], labels=labels).loss
    loss.backward()
    grad_norm = torch.norm(torch.stack(
        [p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None]
    )).item()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    log(f"native-update ok: loss={loss.item():.4f} grad_norm={grad_norm:.4f} peak_GB={peak/1e9:.1f}")

    after_local = torch.tensor(
        [p.detach().float().sum().item() for _, p in model.named_parameters() if p.requires_grad],
        device="cuda",
    )
    changed_local = int(((after_local - before_local).abs() > 0).sum().item())
    drift_local = float((after_local - before_local).abs().sum().item())
    changed_t = torch.tensor([changed_local], device="cuda")
    drift_t = torch.tensor([drift_local], device="cuda")
    dist.all_reduce(changed_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(drift_t, op=dist.ReduceOp.SUM)
    log(f"numeric-change(all-ranks): changed_sum={int(changed_t.item())} drift_sum={float(drift_t.item()):.6f}")
    assert int(changed_t.item()) > 0
    dist.barrier()

    evidence: dict = {}
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    # FSDP2 shards parameters as DTensors; safetensors cannot read sharded
    # storage pointers. Gather the small LoRA-only full tensors to rank 0 via
    # DTensor.full_tensor(), then save with PEFT on rank 0.
    def _full_cpu(t: torch.Tensor) -> torch.Tensor:
        full = t.full_tensor() if hasattr(t, "full_tensor") else t
        return full.detach().to("cpu")

    lora_state = {n: _full_cpu(p) for n, p in model.named_parameters() if p.requires_grad}
    if rank == 0:
        model.save_pretrained(out / "adapter-p1", state_dict=lora_state)
        tok.save_pretrained(out / "adapter-p1")
        p1_sha = sha256_file(out / "adapter-p1" / "adapter_model.safetensors")
        # Reload P1 (fresh base on cuda:0) + re-infer.
        fresh = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to("cuda:0")
        reloaded = PeftModel.from_pretrained(fresh, str(out / "adapter-p1"))
        reloaded.eval()
        with torch.inference_mode():
            gen = reloaded.generate(
                **tok(args.prompt, return_tensors="pt").to("cuda:0"),
                max_new_tokens=32, do_sample=False)
        decoded = tok.decode(gen[0], skip_special_tokens=True)
        log(f"reload-reinfer ok: {decoded[:100]!r}")
        evidence = {
            "tag": "framework-smoke",
            "placement": "dual-GPU FSDP2 sharded, torchrun world=2",
            "verl_version": verl.__version__,
            "vllm_probe": vllm_info,
            "trainable_params": trainable,
            "frozen_params": frozen,
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "peak_memory_GB_rank0": peak / 1e9,
            "changed_tensors_all_ranks": int(changed_t.item()),
            "drift_sum_all_ranks": float(drift_t.item()),
            "p0_adapter_sha256": sha256_file(Path(args.adapter) / "adapter_model.safetensors"),
            "p1_adapter_sha256": p1_sha,
            "reinfer_head": decoded[:200],
            "elapsed_s": time.time() - t0,
        }
        (out / "smoke-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    dist.barrier()
    if rank == 0:
        (out / "smoke.log").write_text("\n".join(log_lines) + "\n")
        log("SMOKE-PASS-DUAL")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
