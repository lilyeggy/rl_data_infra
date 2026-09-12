#!/usr/bin/env python3
"""Run official Qwen2.5-Coder report evaluation using EvalPlus (HumanEval & MBPP).

Strictly follows Qwen2.5-Coder technical report evaluation protocol:
- Datasets: HumanEval (164 tasks), HumanEval+ (164 tasks), MBPP (378 tasks), MBPP+ (378 tasks)
- Prompt format: Official EvalPlus Chat Template for Qwen Instruct
- Decoding: Greedy (temperature=0.0)
- Execution: Sandboxed EvalPlus evaluate harness with both base & extra test cases
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.provider.utility import make_raw_chat_prompt
from evalplus.sanitize import sanitize


def generate_evalplus_samples(
    *,
    dataset: str,
    model_path: str,
    adapter_path: str | None,
    output_jsonl: Path,
    device: str = "cuda:0",
    batch_size: int = 8,
    max_new_tokens: int = 1024,
    prompt_format: str = "chat",
) -> Path:
    print(f"\n=======================================================", flush=True)
    print(f"Generating samples for {dataset.upper()} | Adapter: {adapter_path or 'None (Base)'} | Format: {prompt_format}", flush=True)
    print(f"=======================================================", flush=True)

    if dataset == "humaneval":
        problems = get_human_eval_plus()
    elif dataset == "mbpp":
        problems = get_mbpp_plus()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    task_items = list(problems.items())
    print(f"Total problems in {dataset}: {len(task_items)}", flush=True)

    tok_path = adapter_path if (adapter_path and (Path(adapter_path) / "tokenizer_config.json").exists()) else model_path
    print(f"Loading tokenizer: {tok_path}...", flush=True)
    tok = AutoTokenizer.from_pretrained(tok_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading model on {device}...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)

    if adapter_path and Path(adapter_path).exists():
        print(f"Attaching adapter: {adapter_path}...", flush=True)
        model = PeftModel.from_pretrained(base, adapter_path).to(device)
    else:
        print("Using base model directly...", flush=True)
        model = base

    model.eval()

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    samples = []

    # Process in batches
    t_start = time.time()
    for batch_idx in range(0, len(task_items), batch_size):
        chunk = task_items[batch_idx : batch_idx + batch_size]
        prompts = []
        task_metas = []

        for tid, prob in chunk:
            raw_prompt = prob["prompt"]
            if prompt_format == "base":
                p = raw_prompt
            else:
                p = make_raw_chat_prompt(
                    raw_prompt,
                    "Please complete the following Python code:",
                    "Here is the completed code:",
                    tok,
                )
            prompts.append(p)
            task_metas.append((tid, prob.get("entry_point")))

        inputs = tok(prompts, return_tensors="pt", padding=True).to(device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )

        for i, (tid, entrypoint) in enumerate(task_metas):
            gen_tokens = out[i][input_len:]
            raw_code = tok.decode(gen_tokens, skip_special_tokens=True)
            if prompt_format == "base":
                # For base completion, truncate code fences and next function/class boundary
                if "```" in raw_code:
                    raw_code = raw_code.split("```")[0]
                cleaned_code = None
                try:
                    full_code = problems[tid]["prompt"] + raw_code
                    cleaned = sanitize(full_code, entrypoint)
                    if cleaned and (not entrypoint or f"def {entrypoint}" in cleaned):
                        cleaned_code = cleaned
                except Exception:
                    pass
                if not cleaned_code:
                    trimmed = raw_code.lstrip("\r\n")
                    for stop in ["\ndef ", "\nclass ", "\nif __name__", "\nprint("]:
                        if stop in trimmed:
                            trimmed = trimmed.split(stop)[0]
                    cleaned_code = problems[tid]["prompt"] + trimmed
            else:
                # Remove any trailing code fences
                if "```" in raw_code:
                    raw_code = raw_code.split("```")[0]
                cleaned_code = sanitize(raw_code, entrypoint)
                # Ensure code has the function definition
                if entrypoint and f"def {entrypoint}" not in cleaned_code:
                    cleaned_code = problems[tid]["prompt"] + "\n" + cleaned_code

            samples.append({
                "task_id": tid,
                "solution": cleaned_code,
            })

        elapsed = time.time() - t_start
        print(
            f"  Generated {len(samples)}/{len(task_items)} problems "
            f"({elapsed:.1f}s, {len(samples)/max(1, elapsed):.1f} tasks/s)...",
            flush=True,
        )

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} samples to {output_jsonl}", flush=True)

    # Free model memory
    del model
    del base
    torch.cuda.empty_cache()

    return output_jsonl


def run_evalplus_evaluation(
    dataset: str,
    samples_path: Path,
    python_bin: str = sys.executable,
) -> dict[str, Any]:
    print(f"\n[evalplus] Running official evaluation for {dataset} on {samples_path}...", flush=True)
    eval_cmd = [
        python_bin,
        "-m", "evalplus.evaluate",
        dataset,
        "--samples", str(samples_path),
        "--i_just_wanna_run",
    ]
    res = subprocess.run(eval_cmd, capture_output=True, text=True)
    print(res.stdout, flush=True)
    if res.stderr:
        print("[stderr]", res.stderr, flush=True)

    # Parse eval_results.json produced by evalplus
    results_json = samples_path.parent / f"{samples_path.stem}_eval_results.json"
    if not results_json.exists():
        # Evalplus writes eval_results.json in the same directory as samples or with sample prefix
        cand_results = list(samples_path.parent.glob("*eval_results.json"))
        if cand_results:
            results_json = cand_results[0]

    report = {"dataset": dataset, "samples": str(samples_path), "stdout": res.stdout}
    if results_json.exists():
        try:
            data = json.loads(results_json.read_text(encoding="utf-8"))
            report["eval_results"] = data.get("pass_rates", data)
        except Exception as exc:
            report["parse_error"] = str(exc)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/cxr/agentic/models/qwen2.5-coder-14b-instruct")
    parser.add_argument("--dataset", choices=["humaneval", "mbpp", "all"], default="humaneval")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--adapter", default=None, help="Custom LoRA adapter path")
    parser.add_argument("--model-tag", default="all", help="Model tag (base, sft-v0, rl-v2, rl-v3, all, or custom string)")
    parser.add_argument("--prompt-format", choices=["chat", "base"], default="chat", help="Prompt template format (chat for Instruct, base for raw pre-trained completion)")
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    all_models = [
        ("Base Model (Qwen-14B)", None, "base"),
        ("SFT Policy-v0 (Epoch 2)", "/home/cxr/agentic/checkpoints/sft-runs/qwen14b-apps-clean-v2-260906/epoch2", "sft-v0"),
        ("RL Policy-v2 (Cycle 002)", "/home/cxr/agentic/rl-runs/apps-rl-cycle-002/candidate_policy_adapter", "rl-v2"),
        ("RL Policy-v3 (Cycle 003)", "/home/cxr/agentic/rl-runs/apps-rl-cycle-003/candidate_policy_adapter", "rl-v3"),
    ]
    if args.adapter:
        tag = args.model_tag if args.model_tag != "all" else "custom"
        models = [(f"Custom ({tag})", args.adapter, tag)]
    elif args.model_tag != "all":
        models = [m for m in all_models if m[2] == args.model_tag]
    else:
        models = all_models

    datasets = ["humaneval"] if args.dataset == "humaneval" else (["mbpp"] if args.dataset == "mbpp" else ["humaneval", "mbpp"])

    all_reports = []

    for d in datasets:
        expected_len = 164 if d == "humaneval" else 378
        for m_name, adapter, m_tag in models:
            samples_file = args.output_dir / d / f"{m_tag}_samples.jsonl"
            eval_results_file = args.output_dir / d / f"{m_tag}_samples_eval_results.json"

            skip_gen = False
            if args.resume and samples_file.exists():
                lines_count = sum(1 for line in open(samples_file, encoding="utf-8") if line.strip())
                if lines_count == expected_len:
                    print(f"\n[resume] Found completed {samples_file} ({lines_count}/{expected_len}), skipping generation.", flush=True)
                    skip_gen = True

            if not skip_gen:
                generate_evalplus_samples(
                    dataset=d,
                    model_path=args.model,
                    adapter_path=adapter,
                    output_jsonl=samples_file,
                    device=args.device,
                    batch_size=args.batch_size,
                    prompt_format=args.prompt_format,
                )

            if args.resume and eval_results_file.exists():
                print(f"[resume] Found completed {eval_results_file}, loading existing evaluation results.", flush=True)
                rep = {"dataset": d, "samples": str(samples_file), "eval_results": json.loads(eval_results_file.read_text(encoding="utf-8"))}
            else:
                rep = run_evalplus_evaluation(d, samples_file)

            all_reports.append({
                "model_name": m_name,
                "model_tag": m_tag,
                "dataset": d,
                "report": rep,
            })

    summary_file = args.output_dir / "official-qwen-report-summary.json"
    summary_file.write_text(json.dumps(all_reports, indent=2, ensure_ascii=False) + "\n")
    print(f"\nAll benchmark evaluations complete! Summary written to {summary_file}")


if __name__ == "__main__":
    main()
