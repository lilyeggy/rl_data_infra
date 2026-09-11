#!/usr/bin/env python3
"""
Merge a trained LoRA adapter (SFT or RL) into the base model weights.

Produces a standalone, fully merged model directory compatible with
HuggingFace Transformers, vLLM, SGLang, and Ollama without requiring PEFT.

Mathematical equivalent:
    W_merged = W_base + (alpha / r) * (B @ A)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora(
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
    device: str = "cpu",
    torch_dtype: str = "bfloat16",
) -> None:
    t0 = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dtype = getattr(torch, torch_dtype)
    print(f"[1/4] Loading base model from {base_model_path} ({torch_dtype})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"[2/4] Attaching LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print(f"[3/4] Fusing LoRA weights into base weights via merge_and_unload()...")
    merged_model = model.merge_and_unload()

    print(f"[4/4] Saving merged model and tokenizer to {out_path}...")
    merged_model.save_pretrained(out_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(out_path)

    elapsed = time.time() - t0
    print(f"Successfully exported merged model to: {out_path} (took {elapsed:.1f}s)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", required=True, help="Base model directory")
    parser.add_argument("--adapter", required=True, help="LoRA adapter directory")
    parser.add_argument("--output", required=True, help="Merged model output directory")
    parser.add_argument("--device", default="cpu", help="Device for merging (cpu or cuda)")
    parser.add_argument("--dtype", default="bfloat16", help="Torch dtype (bfloat16 or float16)")
    args = parser.parse_args()

    merge_lora(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        output_dir=args.output,
        device=args.device,
        torch_dtype=args.dtype,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
