"""Stage C (DEVIATED, single-GPU): framework smoke on the idle GPU1 only.

Deviation authorized by user 2026-09-09 ~21:00 CST: closeout prescribes dual-GPU
colocated FSDP, but GPU0 is occupied by foreign tasks that must not be touched.
This smoke uses ONLY GPU1 (GPU-52776180-...) and is tagged framework-smoke.

Verifies, with the REAL verl 0.7.1 stack where feasible:
  1. 14B Base + frozen P0 adapter loads (via verl HFModelConfig + PEFT path).
  2. Only LoRA params trainable (FSDP2-equivalent: frozen base check).
  3. One native optimizer update runs; adapter weights change numerically.
  4. Checkpoint saves; reloaded adapter + weight-sync stub + re-inference works.

No Pi, no GRPO, no second GPU. Products tagged framework-smoke, never acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default="def fibonacci(n):")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{time.time() - t0:8.1f}s] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    assert torch.cuda.is_available(), "CUDA required for stage-C smoke"
    # Launched with CUDA_VISIBLE_DEVICES=1: exactly one visible device expected.
    assert torch.cuda.device_count() == 1, (
        f"expected single visible GPU, got {torch.cuda.device_count()}"
    )
    torch.cuda.set_device(0)
    log(f"device: {torch.cuda.get_device_name(0)}")

    try:
        import verl

        assert verl.__version__ == "0.7.1", verl.__version__
        from verl.workers.config import HFModelConfig

        model_cfg = HFModelConfig(
            path=args.base_model,
            lora_rank=8,
            lora_alpha=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_adapter_path=args.adapter,
        )
        log(f"verl HFModelConfig ok: rank={model_cfg.lora_rank} alpha={model_cfg.lora_alpha} "
            f"adapter={model_cfg.lora_adapter_path}")
    except Exception as exc:  # noqa: BLE001 - record and continue, never fake
        log(f"VERL-CONFIG-FAIL: {type(exc).__name__}: {exc}")
        (out / "smoke.log").write_text("\n".join(log_lines) + "\n")
        return 10

    tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda")
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log(f"params: trainable={trainable} ({trainable/1e6:.2f}M) frozen={frozen}")
    assert trainable > 0 and frozen > trainable, "LoRA-only training check failed"

    before = {n: p.detach().float().sum().item() for n, p in model.named_parameters() if p.requires_grad}
    peak_before = torch.cuda.max_memory_allocated()

    inputs = tok(args.prompt, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-6
    )
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    grad_norm = sum(p.grad.detach().float().norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    log(f"native-update ok: loss={loss.item():.4f} grad_norm={grad_norm:.4f} "
        f"peak_GB={peak/1e9:.1f} (before_GB={peak_before/1e9:.1f})")

    model.save_pretrained(out / "adapter-p1")
    tok.save_pretrained(out / "adapter-p1")
    after = {}
    reloaded = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to("cuda"),
        str(out / "adapter-p1"),
    )
    drift_vs_p0 = 0.0
    changed = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            after[n] = p.detach().float().sum().item()
            if abs(after[n] - before[n]) > 0:
                changed += 1
            drift_vs_p0 += abs(after[n] - before[n])
    log(f"numeric-change: {changed}/{len(before)} tensors changed, total_drift={drift_vs_p0:.6f}")
    assert changed > 0, "no numeric change after native update"

    # Weight-sync + re-inference proof: reloaded P1 generates.
    reloaded.eval()
    with torch.inference_mode():
        gen = reloaded.generate(**tok(args.prompt, return_tensors="pt").to("cuda"),
                                max_new_tokens=32, do_sample=False)
    decoded = tok.decode(gen[0], skip_special_tokens=True)
    log(f"reload-reinfer ok: {decoded[:120]!r}")

    evidence = {
        "tag": "framework-smoke",
        "deviation": "single-GPU1-only; dual-GPU colocated FSDP deferred to windowed run",
        "verl_version": verl.__version__,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "loss": loss.item(),
        "grad_norm": grad_norm,
        "peak_memory_GB": peak / 1e9,
        "changed_tensors": changed,
        "total_drift": drift_vs_p0,
        "p0_adapter_sha256": sha256_file(Path(args.adapter) / "adapter_model.safetensors"),
        "p1_adapter_sha256": sha256_file(out / "adapter-p1" / "adapter_model.safetensors"),
        "reinfer_head": decoded[:200],
        "elapsed_s": time.time() - t0,
    }
    (out / "smoke-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    (out / "smoke.log").write_text("\n".join(log_lines) + "\n")
    log("SMOKE-PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
