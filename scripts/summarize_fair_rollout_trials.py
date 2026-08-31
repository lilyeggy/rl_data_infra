#!/usr/bin/env python3
"""Aggregate repeated controlled Local Data Plane versus Polar trials."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def _local(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    wall = float(data["wall_seconds"])
    done = int(data["done"])
    total = int(data["total"])
    return {
        "system": "local-data-plane+vllm",
        "trial": path.parent.parent.name,
        "concurrency": int(data["concurrency"]),
        "wall_seconds": wall,
        "verified": done,
        "total": total,
        "verified_throughput_per_second": done / wall if wall else 0.0,
        "source": str(path),
    }


def _polar(path: Path, concurrency: int) -> dict | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    wall_match = re.search(r"Wall time:?\s+([0-9.]+)s", text)
    row = re.search(r"^pi\s+(-?[0-9.]+)\s+([0-9]+)/([0-9]+)\s*$", text, re.MULTILINE)
    if not wall_match or not row:
        return None
    wall = float(wall_match.group(1))
    reward = float(row.group(1))
    completed = int(row.group(2))
    total = int(row.group(3))
    verified = max(0, min(completed, round(reward * completed)))
    return {
        "system": "polar+vllm",
        "trial": path.parent.parent.name,
        "concurrency": concurrency,
        "wall_seconds": wall,
        "verified": verified,
        "total": total,
        "verified_throughput_per_second": verified / wall if wall else 0.0,
        "source": str(path),
    }


def _stats(rows: list[dict]) -> dict:
    rates = [row["verified_throughput_per_second"] for row in rows]
    return {
        "trials": len(rows),
        "verified_throughput_mean_per_second": statistics.mean(rates) if rates else 0.0,
        "verified_throughput_median_per_second": statistics.median(rates) if rates else 0.0,
        "verified_throughput_stdev_per_second": statistics.stdev(rates) if len(rates) > 1 else 0.0,
        "verified_throughput_min_per_second": min(rates, default=0.0),
        "verified_throughput_max_per_second": max(rates, default=0.0),
        "verified_rate": (
            sum(row["verified"] for row in rows) / sum(row["total"] for row in rows)
            if rows and sum(row["total"] for row in rows)
            else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    for trial in sorted((args.root / "trials").glob("trial-*")):
        for concurrency in (1, 4, 8):
            local = _local(trial / "local" / f"local-c{concurrency}.json")
            polar = _polar(trial / "polar" / f"benchmark-c{concurrency}.log", concurrency)
            rows.extend(row for row in (local, polar) if row is not None)

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["system"], row["concurrency"])].append(row)
    summary = {
        f"{system}/c{concurrency}": _stats(group_rows)
        for (system, concurrency), group_rows in sorted(grouped.items())
    }
    payload = {"report_version": "fair-rollout-repeat/v1", "runs": rows, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = ["# Repeated Local Data Plane + vLLM vs Polar + vLLM", "", "| System | Concurrency | trials | verified throughput mean ± stdev (/s) | median (/s) | verified rate |", "|---|---:|---:|---:|---:|---:|"]
    for (system, concurrency), group_rows in sorted(grouped.items()):
        stats = _stats(group_rows)
        lines.append(
            f"| {system} | {concurrency} | {stats['trials']} | "
            f"{stats['verified_throughput_mean_per_second']:.4f} ± {stats['verified_throughput_stdev_per_second']:.4f} | "
            f"{stats['verified_throughput_median_per_second']:.4f} | {stats['verified_rate']:.2%} |"
        )
    lines.extend(["", "Only trials with matching fixed workload contract should be interpreted together. The raw per-trial data is in the adjacent JSON file.", ""])
    args.output.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
