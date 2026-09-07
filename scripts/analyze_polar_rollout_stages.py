#!/usr/bin/env python3
"""Summarize Polar session timing and recorded response rounds from saved sessions."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _mean(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return statistics.mean(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict] = []
    for trial in sorted((args.root / "trials").glob("trial-*")):
        for concurrency in (1, 4, 8):
            session_files = list((trial / "polar" / "rollout_results").rglob("ses*.json"))
            for path in session_files:
                data = json.loads(path.read_text(encoding="utf-8"))
                metadata = (data.get("trajectory") or {}).get("metadata") or {}
                task_id = str(data.get("task_id", ""))
                if not task_id:
                    continue
                # benchmark-cN submits a different task directory each time;
                # map using its task id from the matching log.
                log = trial / "polar" / f"benchmark-c{concurrency}.log"
                if not log.exists() or task_id not in log.read_text(encoding="utf-8"):
                    continue
                timing = data.get("timing") or {}
                rows.append({
                    "trial": trial.name,
                    "concurrency": concurrency,
                    "record_count": int(metadata.get("record_count", 0)),
                    "trace_count": int(metadata.get("trace_count", 0)),
                    "run_seconds": float(timing.get("run_ms", 0)) / 1000,
                    "init_seconds": float(timing.get("init_ms", 0)) / 1000,
                    "postrun_seconds": float(timing.get("postrun_ms", 0)) / 1000,
                    "outcome_reward": float(((metadata.get("evaluation") or {}).get("outcome_reward", 0))),
                })
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["concurrency"]].append(row)
    fields = ("record_count", "trace_count", "init_seconds", "run_seconds", "postrun_seconds", "outcome_reward")
    summary = {f"c{c}": {field: _mean(group, field) for field in fields} | {"sessions": len(group)} for c, group in sorted(grouped.items())}
    payload = {"report_version": "polar-rollout-stage-analysis/v1", "sessions": rows, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# Polar recorded rollout stages", "", "| concurrency | sessions | response records | traces | init(s) | run(s) | postrun(s) | reward |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for concurrency, values in sorted(summary.items()):
        lines.append(f"| {concurrency} | {values['sessions']} | {values['record_count']:.2f} | {values['trace_count']:.2f} | {values['init_seconds']:.2f} | {values['run_seconds']:.2f} | {values['postrun_seconds']:.2f} | {values['outcome_reward']:.2f} |")
    lines.append("")
    args.output.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
