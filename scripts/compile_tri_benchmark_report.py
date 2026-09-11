#!/usr/bin/env python3
"""
Compile comprehensive Tri-Benchmark evaluation report:
  1. HumanEval / HumanEval+ (164 tasks)
  2. MBPP / MBPP+ (378 tasks)
  3. BigCodeBench-Hard (148 tasks, complete split)

Compares:
  - Base Model: Qwen2.5-Coder-14B-Instruct
  - SFT Policy-v0: Clean APPS SFT Epoch 2
  - RL Policy-v2: On-Policy APPS GRPO LoRA Adapter (apps-rl-cycle-002)
  - RL Policy-v3: On-Policy APPS GRPO LoRA Adapter (apps-rl-cycle-003, HumanEval only)

Models are only included in a benchmark section when their raw eval results
file exists, so partial candidates (e.g. RL Policy-v3 on HumanEval) appear
exactly where evidence exists and nowhere else.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def load_evalplus_results(json_path: Path) -> Dict[str, Any]:
    """Parse EvalPlus result json."""
    if not json_path.exists():
        return {"error": f"file not found: {json_path}"}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    eval_dict = data.get("eval", {})
    total = len(eval_dict)
    if total == 0:
        return {"error": "empty eval dict"}
    base_pass = sum(1 for v in eval_dict.values() if v[0]["base_status"] == "pass")
    plus_pass = sum(1 for v in eval_dict.values() if v[0]["plus_status"] == "pass")
    return {
        "total": total,
        "base_passed": base_pass,
        "base_pass@1": round(base_pass / total, 4),
        "plus_passed": plus_pass,
        "plus_pass@1": round(plus_pass / total, 4),
    }


def load_bigcodebench_results(json_path: Path) -> Dict[str, Any]:
    """Parse BigCodeBench result json."""
    if not json_path.exists():
        return {"error": f"file not found: {json_path}"}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    summary = data.get("summary", {})
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    return {
        "total": total,
        "passed": passed,
        "pass@1": round(passed / max(1, total), 4),
        "pass@1_percent": round(passed / max(1, total) * 100, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile Tri-Benchmark report")
    parser.add_argument(
        "--evalplus-dir",
        type=Path,
        default=Path("/home/cxr/agentic/rl-runs/official-qwen-evalplus"),
    )
    parser.add_argument(
        "--bcb-dir",
        type=Path,
        default=Path("/home/cxr/agentic/rl-runs/official-qwen-bigcodebench"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/cxr/agentic/rl-runs/official-tri-benchmark-report.json"),
    )
    args = parser.parse_args()

    models = ["base", "sft-v0", "rl-v2", "rl-v3"]
    report: Dict[str, Any] = {
        "benchmark_suite": "Official Qwen2.5-Coder Tri-Benchmark (HumanEval, MBPP, BigCodeBench)",
        "models": {
            "base": "Qwen2.5-Coder-14B-Instruct (Zero-Shot Base)",
            "sft-v0": "APPS Clean-v2 SFT Epoch 2",
            "rl-v2": "On-Policy APPS GRPO Candidate Policy Adapter (apps-rl-cycle-002)",
            "rl-v3": "On-Policy APPS GRPO Candidate Policy Adapter (apps-rl-cycle-003)",
        },
        "humaneval": {},
        "mbpp": {},
        "bigcodebench_hard": {},
    }

    def collect(section: Dict[str, Any], results_dir: Path, template: str, loader) -> None:
        for m in models:
            p = results_dir / template.format(m)
            if p.exists():
                section[m] = loader(p)

    # 1. HumanEval
    collect(report["humaneval"], args.evalplus_dir / "humaneval", "{}_samples_eval_results.json", load_evalplus_results)

    # 2. MBPP
    collect(report["mbpp"], args.evalplus_dir / "mbpp", "{}_samples_eval_results.json", load_evalplus_results)

    # 3. BigCodeBench Hard (historical runs used a hard_complete/ subdirectory;
    #    the committed repo layout stores results directly next to the suite dir)
    bcb_dir = args.bcb_dir / "hard_complete" if (args.bcb_dir / "hard_complete").exists() else args.bcb_dir
    collect(report["bigcodebench_hard"], bcb_dir, "{}_eval_results.json", load_bigcodebench_results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print clean markdown summary table
    print("\n" + "=" * 80)
    print("           OFFICIAL QWEN2.5-CODER TRI-BENCHMARK COMPARISON REPORT")
    print("=" * 80 + "\n")

    print("| Benchmark | Metric | Base 14B | SFT Policy-v0 | RL Policy-v2 | Δ (RL vs Base) | Δ (RL vs SFT) |")
    print("|---|---|---:|---:|---:|---:|---:|")

    def fmt_pct(val: Optional[float]) -> str:
        if val is None or not isinstance(val, (int, float)):
            return "N/A"
        return f"{val * 100:.2f}%"

    def fmt_delta(v_new: Optional[float], v_old: Optional[float]) -> str:
        if v_new is None or v_old is None:
            return "N/A"
        diff = (v_new - v_old) * 100
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.2f}%"

    # HumanEval
    he = report["humaneval"]
    b_base = he.get("base", {}).get("base_pass@1")
    s_base = he.get("sft-v0", {}).get("base_pass@1")
    r_base = he.get("rl-v2", {}).get("base_pass@1")
    print(f"| HumanEval (164) | Base Tests | {fmt_pct(b_base)} | {fmt_pct(s_base)} | {fmt_pct(r_base)} | {fmt_delta(r_base, b_base)} | {fmt_delta(r_base, s_base)} |")

    b_plus = he.get("base", {}).get("plus_pass@1")
    s_plus = he.get("sft-v0", {}).get("plus_pass@1")
    r_plus = he.get("rl-v2", {}).get("plus_pass@1")
    print(f"| HumanEval+ (164) | Extra Tests | {fmt_pct(b_plus)} | {fmt_pct(s_plus)} | {fmt_pct(r_plus)} | {fmt_delta(r_plus, b_plus)} | {fmt_delta(r_plus, s_plus)} |")

    # MBPP
    mb = report["mbpp"]
    mb_b_base = mb.get("base", {}).get("base_pass@1")
    mb_s_base = mb.get("sft-v0", {}).get("base_pass@1")
    mb_r_base = mb.get("rl-v2", {}).get("base_pass@1")
    print(f"| MBPP (378) | Base Tests | {fmt_pct(mb_b_base)} | {fmt_pct(mb_s_base)} | {fmt_pct(mb_r_base)} | {fmt_delta(mb_r_base, mb_b_base)} | {fmt_delta(mb_r_base, mb_s_base)} |")

    mb_b_plus = mb.get("base", {}).get("plus_pass@1")
    mb_s_plus = mb.get("sft-v0", {}).get("plus_pass@1")
    mb_r_plus = mb.get("rl-v2", {}).get("plus_pass@1")
    print(f"| MBPP+ (378) | Extra Tests | {fmt_pct(mb_b_plus)} | {fmt_pct(mb_s_plus)} | {fmt_pct(mb_r_plus)} | {fmt_delta(mb_r_plus, mb_b_plus)} | {fmt_delta(mb_r_plus, mb_s_plus)} |")

    # BigCodeBench-Hard
    bcb = report["bigcodebench_hard"]
    bcb_b = bcb.get("base", {}).get("pass@1")
    bcb_s = bcb.get("sft-v0", {}).get("pass@1")
    bcb_r = bcb.get("rl-v2", {}).get("pass@1")
    print(f"| BigCodeBench-Hard (148) | Complete pass@1 | {fmt_pct(bcb_b)} | {fmt_pct(bcb_s)} | {fmt_pct(bcb_r)} | {fmt_delta(bcb_r, bcb_b)} | {fmt_delta(bcb_r, bcb_s)} |")

    rl_v3_he = report["humaneval"].get("rl-v3")
    if rl_v3_he and "error" not in rl_v3_he:
        print(
            f"\nRL Policy-v3 (apps-rl-cycle-003): HumanEval base {fmt_pct(rl_v3_he.get('base_pass@1'))} / "
            f"plus {fmt_pct(rl_v3_he.get('plus_pass@1'))} "
            "(MBPP/BigCodeBench not run for this candidate)"
        )

    print("\n" + "=" * 80)
    print(f"Report JSON written to: {args.output}")
    print("=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    main()
