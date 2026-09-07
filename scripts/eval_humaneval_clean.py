#!/usr/bin/env python3
"""Clean HumanEval benchmark evaluation runner.

Evaluates Base Model, SFT (Policy-v0), or RL (Policy-vNext) on the standard
HumanEval benchmark (164 tasks) using sandboxed subprocess execution.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_solution_code(raw_text: str, entry_point: str) -> str:
    """Extract python function implementation from model completion."""
    pattern = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
    matches = pattern.findall(raw_text)
    if matches:
        code = matches[-1]
    else:
        code = raw_text

    return code


def run_code_sandbox(code: str, test: str, entry_point: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Execute code + test in isolated python process."""
    full_script = (
        "import sys\n"
        "import math\n"
        "from typing import *\n\n"
        f"{code}\n\n"
        f"{test}\n\n"
        f"check({entry_point})\n"
        "sys.exit(0)\n"
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full_script)
        tmp_path = f.name

    try:
        res = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        passed = res.returncode == 0
        err = res.stderr.strip()
    except subprocess.TimeoutExpired:
        passed = False
        err = "TimeoutExpired"
    except Exception as exc:
        passed = False
        err = str(exc)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return passed, err


def evaluate_humaneval(
    *,
    dataset_path: Path,
    model_path: str,
    adapter_path: str | None,
    output_path: Path,
    device: str = "cuda:0",
    limit: int | None = None,
) -> dict[str, Any]:
    print(f"[eval] Loading dataset from {dataset_path}...", flush=True)
    with open(dataset_path, "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    if limit is not None:
        tasks = tasks[:limit]

    print(f"[eval] Total tasks to evaluate: {len(tasks)}", flush=True)

    print(f"[eval] Loading tokenizer: {model_path}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[eval] Loading base model on {device}...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)

    if adapter_path and Path(adapter_path).exists():
        print(f"[eval] Attaching adapter: {adapter_path}...", flush=True)
        model = PeftModel.from_pretrained(base, adapter_path).to(device)
    else:
        print("[eval] Evaluating pure base model (no adapter)...", flush=True)
        model = base

    model.eval()

    results = []
    passed_count = 0

    for idx, sample in enumerate(tasks):
        task_id = sample["task_id"]
        entry_point = sample["entry_point"]
        prompt_code = sample["prompt"]
        test = sample["test"]

        chat_prompt = (
            "<|im_start|>system\nYou are an expert python programmer. Complete the following function. "
            "Respond ONLY with the complete python implementation in a ```python ``` codeblock.<|im_end|>\n"
            f"<|im_start|>user\n{prompt_code}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        inputs = tok(chat_prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        elapsed = time.time() - t0

        resp_text = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
        extracted = extract_solution_code(resp_text, entry_point)

        if f"def {entry_point}" not in extracted:
            candidate_code = prompt_code + "\n" + extracted
        else:
            candidate_code = extracted

        passed, err = run_code_sandbox(candidate_code, test, entry_point)
        if passed:
            passed_count += 1

        print(
            f"[{idx+1}/{len(tasks)}] {task_id} ({entry_point}): "
            f"{'PASSED' if passed else 'FAILED'} ({elapsed:.2f}s) | "
            f"Pass@1 so far: {passed_count}/{idx+1} ({passed_count/(idx+1)*100:.1f}%)",
            flush=True,
        )

        results.append({
            "task_id": task_id,
            "entry_point": entry_point,
            "passed": passed,
            "error": err if not passed else "",
            "time_seconds": round(elapsed, 2),
            "generated_tokens": int(out.shape[1] - prompt_len),
        })

    pass_rate = round(passed_count / max(1, len(tasks)), 4)
    summary = {
        "model": model_path,
        "adapter": adapter_path,
        "total_tasks": len(tasks),
        "passed_tasks": passed_count,
        "pass_at_1": pass_rate,
        "pass_percentage": f"{pass_rate * 100:.2f}%",
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[eval complete] Pass@1: {passed_count}/{len(tasks)} ({summary['pass_percentage']})")
    print(f"Results written to: {output_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    evaluate_humaneval(
        dataset_path=args.dataset,
        model_path=args.model,
        adapter_path=args.adapter,
        output_path=args.output,
        device=args.device,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
