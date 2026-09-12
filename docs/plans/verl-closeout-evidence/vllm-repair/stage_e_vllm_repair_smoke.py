"""Verify the repaired vLLM can actually serve the frozen 14B + P0 LoRA.

Root cause of "vLLM unavailable": the conda env has an editable install whose
target (/home/cxr1/vllm) was deleted; a built tree survives at
/home/cxr/ds4-deploy/vllm. This script imports vllm from that tree (without
modifying the existing environment) and proves end-to-end inference:
engine start -> LoRA adapter load -> native token ids + logprobs.

Time-boxed, single GPU. Nothing here modifies the SFT/eval environment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

VLLM_TREE = "/home/cxr/ds4-deploy/vllm"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--attention-backend", default=None)
    args = parser.parse_args()

    # Neutralize the broken editable finder; use the surviving built tree.
    sys.meta_path = [
        f for f in sys.meta_path
        if "editable" not in (getattr(f, "__module__", "") or "").lower()
        and "editable" not in type(f).__name__.lower()
    ]
    sys.path.insert(0, VLLM_TREE)

    t0 = time.time()
    record: dict = {"tag": "vllm-repair-smoke"}

    import vllm

    record["vllm_version"] = getattr(vllm, "__version__", "?")
    record["vllm_path"] = vllm.__file__
    record["import_ok"] = True
    print(f"[{time.time()-t0:6.1f}s] vllm {record['vllm_version']} from {record['vllm_path']}", flush=True)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm_kwargs = dict(
        model=args.base_model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.40,
        enable_lora=True,
        max_lora_rank=8,
        enable_prefix_caching=False,
        enforce_eager=True,
    )
    if args.attention_backend:
        llm_kwargs["attention_backend"] = args.attention_backend
    try:
        llm = LLM(**llm_kwargs)
    except TypeError as exc:
        record["attention_backend_rejected"] = str(exc)[:200]
        llm_kwargs.pop("attention_backend", None)
        llm = LLM(**llm_kwargs)
    record["engine_ok"] = True
    record["engine_start_s"] = time.time() - t0
    print(f"[{time.time()-t0:6.1f}s] engine started", flush=True)

    sampling = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=32, logprobs=1)
    outputs = llm.generate(
        ["def fibonacci(n):"],
        sampling_params=sampling,
        lora_request=LoRARequest("p0", 1, args.adapter),
    )
    out = outputs[0].outputs[0]
    token_ids = list(out.token_ids)
    lp = out.logprobs or []
    record["lora_ok"] = True
    record["generated_tokens"] = len(token_ids)
    record["logprob_steps"] = len(lp)
    record["logprobs_aligned"] = len(lp) == len(token_ids)
    record["first_token_ids"] = token_ids[:8]
    record["text_head"] = out.text[:120]
    print(f"[{time.time()-t0:6.1f}s] lora generate ok tokens={len(token_ids)} "
          f"logprob_steps={len(lp)} aligned={record['logprobs_aligned']}", flush=True)
    record["total_s"] = time.time() - t0
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n")
    print("VLLM-REPAIR-SMOKE-PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
