#!/usr/bin/env python3
"""Evaluate Model / Policy on APPS Holdout Tasks with Greedy Decoding."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_python_code(text: str) -> str:
    pattern = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip() + "\n"
    return text.strip() + "\n"


def evaluate_apps(
    manifest_path: Path,
    tasks_file: Path,
    model_path: str,
    adapter_path: str | None,
    output_dir: Path,
    tag: str,
    device: str = "cuda:0",
    max_new_tokens: int = 640,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    workspaces_dir = output_dir / f"workspaces_{tag}"
    workspaces_dir.mkdir(parents=True, exist_ok=True)

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks_all = manifest_data["tasks"]
    task_ids = json.loads(tasks_file.read_text(encoding="utf-8"))

    print(f"[{tag}] Loading tokenizer and base model: {model_path}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)

    if adapter_path and Path(adapter_path).exists():
        print(f"[{tag}] Attaching adapter: {adapter_path}...", flush=True)
        policy = PeftModel.from_pretrained(base, adapter_path).to(device)
    else:
        print(f"[{tag}] Using base model...", flush=True)
        policy = base

    policy.eval()

    stop_token_ids = [tok.eos_token_id, 151643, 151645]
    stop_token_ids = list(set(tid for tid in stop_token_ids if tid is not None))

    results = []
    passed_count = 0
    python_bin = sys.executable
    verify_script = Path(__file__).resolve().parent / "verify_apps.py"

    t0_all = time.time()
    for idx, task_id in enumerate(task_ids):
        if task_id not in tasks_all:
            continue
        task_data = tasks_all[task_id]
        question = task_data["question"]
        ws = workspaces_dir / task_id
        ws.mkdir(parents=True, exist_ok=True)

        prompt = (
            "<|im_start|>system\nYou are an expert competitive programmer. Implement the requested solution in python3. "
            "The program must read input from standard input (sys.stdin) and write the answer to standard output (sys.stdout). "
            "Output only the executable Python code inside a ```python ``` codeblock without unnecessary chatter.<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        inputs = tok(prompt, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]

        t0 = time.time()
        with torch.no_grad():
            out = policy.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
                eos_token_id=stop_token_ids,
            )
        gen_time = time.time() - t0

        resp_ids = out[0][prompt_len:].cpu().numpy().tolist()
        resp_text = tok.decode(resp_ids, skip_special_tokens=True)
        code = extract_python_code(resp_text)
        (ws / "solution.py").write_text(code, encoding="utf-8")

        verifier_out = ws / "verifier-output.json"
        cmd = [
            python_bin,
            str(verify_script),
            "--manifest",
            str(manifest_path),
            "--task-id",
            task_id,
            "--source-worktree",
            str(ws),
            "--python",
            python_bin,
            "--output",
            str(verifier_out),
            "--timeout",
            "10",
        ]
        subprocess.run(cmd, capture_output=True, text=True)

        is_passed = False
        cases_passed = 0
        total_cases = 0
        if verifier_out.exists():
            try:
                rep = json.loads(verifier_out.read_text(encoding="utf-8"))
                is_passed = bool(rep.get("resolved", False))
                cases_passed = rep.get("passed_cases", 0)
                total_cases = rep.get("case_count", 0)
            except Exception:
                pass

        if is_passed:
            passed_count += 1

        results.append({
            "task_id": task_id,
            "difficulty": task_data.get("difficulty", "unknown"),
            "passed": is_passed,
            "cases_passed": cases_passed,
            "total_cases": total_cases,
            "gen_time": round(gen_time, 2),
            "tokens": len(resp_ids),
        })

        if (idx + 1) % 10 == 0 or idx == len(task_ids) - 1:
            print(
                f"[{tag}] Evaluated {idx+1}/{len(task_ids)} tasks: "
                f"passed={passed_count}/{idx+1} ({passed_count/(idx+1)*100:.1f}%), "
                f"elapsed={time.time()-t0_all:.1f}s",
                flush=True,
            )

    summary = {
        "tag": tag,
        "model": model_path,
        "adapter": adapter_path,
        "total_tasks": len(results),
        "passed_tasks": passed_count,
        "pass_rate": round(passed_count / max(1, len(results)), 4),
        "pass_rate_percent": round(passed_count / max(1, len(results)) * 100, 2),
        "tasks": results,
    }

    summary_file = output_dir / f"{tag}_apps_holdout_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(
        f"[{tag}] Finished: {passed_count}/{len(results)} ({summary['pass_rate_percent']}%) -> {summary_file}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--tasks-file", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=640)
    args = parser.parse_args()

    evaluate_apps(
        manifest_path=args.manifest,
        tasks_file=args.tasks_file,
        model_path=args.model,
        adapter_path=args.adapter,
        output_dir=args.output_dir,
        tag=args.tag,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
