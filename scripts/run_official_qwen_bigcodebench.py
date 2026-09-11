#!/usr/bin/env python3
"""
Run official BigCodeBench evaluation for Qwen2.5-Coder models matching
the official Qwen2.5-Coder technical report protocol.

Supports:
  - BigCodeBench-Hard (148 tasks) and BigCodeBench-Full (1140 tasks)
  - Splits: complete (default) and instruct
  - Models: Base 14B, SFT Policy-v0, RL Policy-v2
  - Fast batched greedy generation on CUDA
  - Local multi-process test evaluation via untrusted_check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from bigcodebench.eval import untrusted_check
from bigcodebench.provider.utility import make_raw_chat_prompt
from bigcodebench.sanitize import sanitize


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load BigCodeBench jsonl dataset."""
    tasks = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def generate_samples(
    tasks: List[Dict[str, Any]],
    model_path: str,
    adapter_path: Optional[str],
    output_jsonl: Path,
    split: str = "complete",
    subset: str = "hard",
    device: str = "cuda:0",
    batch_size: int = 4,
    max_new_tokens: int = 1280,
    resume: bool = True,
    direct_completion: bool = False,
) -> List[Dict[str, Any]]:
    """Generate completions using Qwen instruct chat template and greedy decoding."""
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    completed_samples: Dict[str, Dict[str, Any]] = {}
    if resume and output_jsonl.exists():
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    s = json.loads(line)
                    completed_samples[s["task_id"]] = s
        print(f"Resuming: found {len(completed_samples)} existing samples in {output_jsonl}")

    remaining_tasks = [t for t in tasks if t["task_id"] not in completed_samples]
    if not remaining_tasks:
        print(f"All {len(tasks)} tasks already generated.")
        return list(completed_samples.values())

    tok_path = adapter_path if (adapter_path and (Path(adapter_path) / "tokenizer_config.json").exists()) else model_path
    print(f"Loading tokenizer from {tok_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    print(f"Loading model from {model_path} onto {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    if adapter_path:
        print(f"Attaching LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    instruction_prefix = (
        "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
    )
    response_prefix = (
        "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"
    )

    print(f"Generating {len(remaining_tasks)} tasks (batch_size={batch_size}, max_new_tokens={max_new_tokens})...")
    t0 = time.time()

    samples_file = open(output_jsonl, "a" if resume else "w", encoding="utf-8")

    for i in range(0, len(remaining_tasks), batch_size):
        chunk = remaining_tasks[i : i + batch_size]
        prompts = []
        metas = []

        for task in chunk:
            tid = task["task_id"]
            prompt_text = task["complete_prompt"] if split == "complete" else task["instruct_prompt"]
            if direct_completion:
                chat_p = prompt_text
            else:
                chat_p = make_raw_chat_prompt(
                    task_prompt=prompt_text,
                    subset=subset,
                    split=split,
                    instruction_prefix=instruction_prefix,
                    response_prefix=response_prefix,
                    tokenizer=tokenizer,
                    prefill=True,
                )
            prompts.append(chat_p)
            metas.append((tid, task.get("entry_point"), task, prompt_text))

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, (tid, entry_point, task, prompt_text) in enumerate(metas):
            gen_tokens = outputs[j][input_len:]
            raw_code = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            if direct_completion:
                cleaned = sanitize(prompt_text + raw_code, entry_point)
            else:
                cleaned = sanitize(raw_code, entry_point)
            sample_entry = {
                "task_id": tid,
                "solution": cleaned,
                "raw_completion": raw_code,
            }
            completed_samples[tid] = sample_entry
            samples_file.write(json.dumps(sample_entry, ensure_ascii=False) + "\n")
            samples_file.flush()

        done_count = len(completed_samples)
        elapsed = time.time() - t0
        print(
            f"  Generated {done_count}/{len(tasks)} tasks "
            f"({elapsed:.1f}s, {done_count / max(1, elapsed):.2f} tasks/s)...",
            flush=True,
        )

    samples_file.close()

    # Free model memory from GPU
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return [completed_samples[t["task_id"]] for t in tasks if t["task_id"] in completed_samples]


def eval_task_worker(args: tuple) -> Dict[str, Any]:
    """Worker function for running untrusted_check in parallel."""
    tid, task, solution, calibrated = args
    entry_point = task["entry_point"]
    test_code = task["test"]

    # In calibrated mode, prepend code_prompt + pass to ensure function skeleton
    if calibrated:
        full_code = task["code_prompt"] + "\n    pass\n" + solution
    else:
        full_code = solution

    try:
        stat, details = untrusted_check(
            code=full_code,
            test_code=test_code,
            entry_point=entry_point,
            max_as_limit=None,
            max_data_limit=None,
            max_stack_limit=None,
            min_time_limit=1.0,
            gt_time_limit=20.0,
        )
    except Exception as exc:
        stat = "fail"
        details = {"error": str(exc)}

    return {
        "task_id": tid,
        "status": stat,
        "passed": (stat == "pass"),
        "details": details,
    }


def evaluate_samples(
    tasks: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    output_eval_json: Path,
    workers: int = 8,
    calibrated: bool = True,
) -> Dict[str, Any]:
    """Evaluate samples using parallel untrusted_check execution."""
    task_map = {t["task_id"]: t for t in tasks}
    sample_map = {s["task_id"]: s for s in samples}

    worker_args = []
    for tid, task in task_map.items():
        if tid in sample_map:
            worker_args.append((tid, task, sample_map[tid]["solution"], calibrated))

    print(f"Evaluating {len(worker_args)} tasks with {workers} workers (calibrated={calibrated})...")
    t0 = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(eval_task_worker, arg): arg[0] for arg in worker_args}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            done = len(results)
            if done % 20 == 0 or done == len(worker_args):
                print(f"  Evaluated {done}/{len(worker_args)} tasks ({(time.time()-t0):.1f}s)...", flush=True)

    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    pass_rate = passed_count / max(1, total_count)

    summary = {
        "total": total_count,
        "passed": passed_count,
        "pass@1": round(pass_rate, 4),
        "pass@1_percent": round(pass_rate * 100, 2),
        "calibrated": calibrated,
        "evaluation_time_seconds": round(time.time() - t0, 2),
    }

    report = {
        "summary": summary,
        "results": sorted(results, key=lambda x: x["task_id"]),
    }

    output_eval_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_eval_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n=======================================================")
    print(f"BigCodeBench Evaluation Results:")
    print(f"  Total:  {total_count}")
    print(f"  Passed: {passed_count}")
    print(f"  Pass@1: {pass_rate*100:.2f}% ({passed_count}/{total_count})")
    print(f"  Output: {output_eval_json}")
    print(f"=======================================================\n")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Official BigCodeBench evaluation for Qwen2.5-Coder models")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to BigCodeBench jsonl file")
    parser.add_argument("--split", choices=["complete", "instruct"], default="complete", help="Prompt split")
    parser.add_argument("--subset", choices=["hard", "full"], default="hard", help="Task subset")
    parser.add_argument("--base-model", type=str, default="/home/cxr/agentic/models/qwen2.5-coder-14b-instruct")
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter path (optional)")
    parser.add_argument("--model-tag", type=str, required=True, help="Model identifier (base, sft-v0, rl-v2)")
    parser.add_argument("--output-dir", type=Path, default=Path("/home/cxr/agentic/rl-runs/official-qwen-bigcodebench"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1280)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-only", action="store_true", help="Skip generation and evaluate existing samples")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing samples")
    parser.add_argument(
        "--direct-completion",
        action="store_true",
        help="Direct code completion mode (without chat template) for base models",
    )
    args = parser.parse_args()

    tasks = load_dataset(args.dataset)
    print(f"Loaded {len(tasks)} tasks from {args.dataset} (split={args.split}, subset={args.subset})")

    samples_jsonl = args.output_dir / f"{args.subset}_{args.split}" / f"{args.model_tag}_samples.jsonl"
    eval_json = args.output_dir / f"{args.subset}_{args.split}" / f"{args.model_tag}_eval_results.json"

    if not args.eval_only:
        samples = generate_samples(
            tasks=tasks,
            model_path=args.base_model,
            adapter_path=args.adapter,
            output_jsonl=samples_jsonl,
            split=args.split,
            subset=args.subset,
            device=args.device,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            resume=not args.no_resume,
            direct_completion=args.direct_completion,
        )
    else:
        samples = []
        with open(samples_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

    report = evaluate_samples(
        tasks=tasks,
        samples=samples,
        output_eval_json=eval_json,
        workers=args.workers,
        calibrated=True,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
