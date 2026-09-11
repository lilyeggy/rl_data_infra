#!/usr/bin/env python3
"""Stage E recovery verification at the P1 checkpoint boundary (closeout E).

After the first update, prove the checkpoint boundary is recoverable in a fresh
process -- without replaying the consumed batch and without resuming a half
episode:

  1. Load base + P1 from disk in a new process.
  2. Compare the in-memory LoRA tensors bitwise against the saved safetensors
     (the checkpoint must encode exactly the post-update policy).
  3. Score the round-1 sequences once (forward only) as a consistency probe.
  4. Run one forward/backward to prove training can continue from this boundary
     (gradients finite and non-zero) -- no optimizer step is taken, so the
     consumed batch is not re-consumed.

Single GPU; read-only with respect to the round-1 rollout evidence.
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
    parser.add_argument("--adapter", required=True, help="P1 checkpoint dir")
    parser.add_argument("--round-json", required=True, help="the consumed round batch")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    adapter = Path(args.adapter)
    result: dict = {
        "tag": "stage-e-recovery-check",
        "adapter_dir": str(adapter),
        "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
    }

    tok = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda")
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
    model.eval()
    result["loaded_ok"] = True

    # (2) bitwise identity between in-memory LoRA tensors and the saved file.
    # PEFT names adapters `<...>.lora_A.default.weight`; safetensors drops the
    # `.default` segment only.
    saved = load_file(str(adapter / "adapter_model.safetensors"))
    mismatches = []
    checked = 0
    for name, param in model.named_parameters():
        key = name.replace(".default", "")
        if key in saved:
            checked += 1
            if not torch.equal(param.detach().float().cpu(), saved[key].float()):
                mismatches.append(name)
    result["tensors_compared"] = checked
    result["bitwise_mismatches"] = mismatches
    result["bitwise_identical"] = checked > 0 and not mismatches

    # (3) consistency probe: score the consumed round-1 batch once, forward only.
    round_data = json.loads(Path(args.round_json).read_text())
    episodes = [e for e in round_data["episodes"] if "error" not in e]
    scored = []
    for episode in episodes:
        prompt_ids = list(episode["prompt_ids"])
        response_ids = list(episode["response_ids"])
        input_ids = torch.tensor([prompt_ids + response_ids], device="cuda", dtype=torch.long)
        with torch.no_grad():
            logits = model(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False
            ).logits
        start = len(prompt_ids) - 1
        nll = F.cross_entropy(
            logits[0, start:start + len(response_ids), :].float(),
            torch.tensor(response_ids, device="cuda", dtype=torch.long),
            reduction="none",
        )
        mask = torch.tensor(episode["response_mask"], dtype=torch.float32)
        p1_lp = (-nll).float().cpu()
        rollout = torch.tensor(episode["rollout_logprobs"], dtype=torch.float32)
        shift = (p1_lp - rollout)[mask > 0]
        scored.append({
            "episode_id": episode["episode_id"],
            "masked_tokens": int(mask.sum()),
            "finite_logprobs": bool(torch.isfinite(p1_lp[mask > 0]).all()),
            "mean_p1_minus_rollout_nat": float(shift.mean()),
            "mean_abs_p1_minus_rollout_nat": float(shift.abs().mean()),
        })
    result["rescored_episodes"] = scored
    result["all_logprobs_finite"] = all(s["finite_logprobs"] for s in scored)

    # (4) continuation proof: one backward from the boundary, no optimizer step.
    model.train()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    episode = episodes[0]
    prompt_ids = list(episode["prompt_ids"])
    response_ids = list(episode["response_ids"])
    input_ids = torch.tensor([prompt_ids + response_ids], device="cuda", dtype=torch.long)
    logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False).logits
    start = len(prompt_ids) - 1
    loss = F.cross_entropy(
        logits[0, start:start + len(response_ids), :].float(),
        torch.tensor(response_ids, device="cuda", dtype=torch.long),
    )
    loss.backward()
    grads = [p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None]
    grad_norm = float(torch.sqrt(sum(g ** 2 for g in grads))) if grads else 0.0
    params_after = {p: p.detach().clone() for p in model.parameters() if p.requires_grad}
    result["continuation"] = {
        "loss": float(loss.item()),
        "grad_norm": grad_norm,
        "grad_tensors": len(grads),
        "grad_finite": bool(all(torch.isfinite(g).all() for g in grads)),
        "step_taken": False,
        "note": "no optimizer step: the consumed batch is not re-consumed",
    }
    del params_after
    result["elapsed_s"] = time.time() - t0
    result["pass"] = bool(
        result["bitwise_identical"] and result["all_logprobs_finite"] and grad_norm > 0
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("STAGE-E-RECOVERY-" + ("PASS" if result["pass"] else "FAIL"), flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
