#!/usr/bin/env python3
"""Summarize Evidence-on versus Evidence-off Local Data Plane trials."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict] = []
    for trial in sorted((args.root / "trials").glob("trial-*")):
        for mode in ("rl-logprobs", "observability-only"):
            for concurrency in (1, 4, 8):
                path = trial / mode / f"local-c{concurrency}.json"
                if not path.exists():
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
                wall = float(data["wall_seconds"])
                results = data.get("results", [])
                calls = sum(int(item.get("model_call_count", 0)) for item in results)
                rl_calls = sum(int(item.get("rl_usable_model_call_count", 0)) for item in results)
                rows.append({
                    "trial": trial.name,
                    "mode": mode,
                    "concurrency": concurrency,
                    "wall_seconds": wall,
                    "completed": int(data["completed"]),
                    "verified": int(data["done"]),
                    "total": int(data["total"]),
                    "verified_throughput_per_second": int(data["done"]) / wall,
                    "completed_throughput_per_second": int(data["completed"]) / wall,
                    "model_calls": calls,
                    "rl_usable_calls": rl_calls,
                })
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["mode"], row["concurrency"])].append(row)
    summary = {}
    for (mode, concurrency), group in sorted(groups.items()):
        rates = [item["verified_throughput_per_second"] for item in group]
        summary[f"{mode}/c{concurrency}"] = {
            "trials": len(group),
            "verified_throughput_mean_per_second": statistics.mean(rates),
            "verified_throughput_stdev_per_second": statistics.stdev(rates) if len(rates) > 1 else 0.0,
            "verified_rate": sum(item["verified"] for item in group) / sum(item["total"] for item in group),
            "completed_throughput_mean_per_second": statistics.mean(
                item["completed_throughput_per_second"] for item in group
            ),
            "rl_usable_call_rate": sum(item["rl_usable_calls"] for item in group) / sum(item["model_calls"] for item in group),
        }
    payload = {"report_version": "evidence-ablation/v1", "runs": rows, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# Evidence overhead ablation", "", "| mode | concurrency | trials | completed throughput (/s) | verified throughput mean ± stdev (/s) | verified rate | RL-usable call rate |", "|---|---:|---:|---:|---:|---:|---:|"]
    for (mode, concurrency), group in sorted(groups.items()):
        value = summary[f"{mode}/c{concurrency}"]
        lines.append(f"| {mode} | {concurrency} | {value['trials']} | {value['completed_throughput_mean_per_second']:.4f} | {value['verified_throughput_mean_per_second']:.4f} ± {value['verified_throughput_stdev_per_second']:.4f} | {value['verified_rate']:.2%} | {value['rl_usable_call_rate']:.2%} |")
    lines.append("")
    args.output.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
