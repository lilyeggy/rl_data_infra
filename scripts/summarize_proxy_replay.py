#!/usr/bin/env python3
"""Summarize paired direct-vLLM versus Data Plane proxy replay trials."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    pairs: dict[int, list[tuple[dict, dict]]] = {}
    for direct_path in sorted(args.root.glob("trial-*/c*/direct/summary.json")):
        proxy_path = direct_path.parent.parent / "proxy" / "summary.json"
        if not proxy_path.exists():
            continue
        direct = json.loads(direct_path.read_text(encoding="utf-8"))
        proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
        if direct["success_count"] != direct["requests"] or proxy["success_count"] != proxy["requests"]:
            continue
        pairs.setdefault(int(direct["concurrency"]), []).append((direct, proxy))
    rows = []
    for concurrency, trials in sorted(pairs.items()):
        direct, proxy = zip(*trials)
        d_p95 = [item["e2e_latency_ms"]["p95"] for item in direct]
        p_p95 = [item["e2e_latency_ms"]["p95"] for item in proxy]
        d_tokens = [item["completion_token_goodput_per_second"] for item in direct]
        p_tokens = [item["completion_token_goodput_per_second"] for item in proxy]
        rows.append({
            "concurrency": concurrency,
            "paired_trials": len(trials),
            "direct_e2e_p95_ms_mean": _mean(d_p95),
            "proxy_e2e_p95_ms_mean": _mean(p_p95),
            "proxy_added_e2e_p95_ms_mean": _mean([p - d for d, p in zip(d_p95, p_p95)]),
            "direct_completion_token_goodput_per_second_mean": _mean(d_tokens),
            "proxy_completion_token_goodput_per_second_mean": _mean(p_tokens),
            "proxy_token_goodput_ratio_to_direct": _mean(p_tokens) / _mean(d_tokens) if _mean(d_tokens) else 0,
        })
    report = {
        "benchmark": "fixed-model-replay/v1",
        "comparison_contract": "same frozen prompt, vLLM model, decoding seed, max_tokens and upstream logprob request; only direct versus Data Plane proxy/evidence differs",
        "rows": rows,
    }
    output = args.output or args.root / "paired-summary.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
