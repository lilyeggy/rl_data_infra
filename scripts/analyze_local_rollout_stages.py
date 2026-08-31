#!/usr/bin/env python3
"""Derive Local Data Plane stage timings from host-owned episode evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _seconds(events: list[dict], event_type: str) -> datetime | None:
    for event in events:
        if event.get("event_type") == event_type:
            return _time(event["timestamp"])
    return None


def _one(result: dict, concurrency: int, trial: str) -> dict | None:
    output = Path(result["output_dir"])
    events_path = output / "raw-events.jsonl"
    evidence_path = output / "model-evidence.jsonl"
    if not events_path.exists() or not evidence_path.exists():
        return None
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    evidence = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines()]
    sandbox_started = _seconds(events, "SANDBOX_STARTED")
    first_request = _seconds(events, "MODEL_REQUEST")
    sandbox_finished = _seconds(events, "SANDBOX_FINISHED")
    episode_finished = _seconds(events, "EPISODE_FINISHED")
    if not all((sandbox_started, first_request, sandbox_finished, episode_finished)):
        return None
    model_seconds = sum(float(item.get("backend", {}).get("latency_ms", 0)) for item in evidence) / 1000
    sandbox_span = (sandbox_finished - sandbox_started).total_seconds()
    traced_span = (episode_finished - sandbox_started).total_seconds()
    return {
        "trial": trial,
        "concurrency": concurrency,
        "episode_index": result["index"],
        "total_wall_seconds": float(result["seconds"]),
        "bootstrap_to_first_model_seconds": (first_request - sandbox_started).total_seconds(),
        "model_seconds": model_seconds,
        "tool_queue_and_harness_seconds": max(0.0, sandbox_span - model_seconds),
        "verification_and_finalize_seconds": (episode_finished - sandbox_finished).total_seconds(),
        "outside_traced_span_seconds": max(0.0, float(result["seconds"]) - traced_span),
        "model_calls": len(evidence),
    }


def _mean(rows: list[dict], field: str) -> float:
    return statistics.mean(float(row[field]) for row in rows) if rows else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict] = []
    for trial_path in sorted((args.root / "trials").glob("trial-*")):
        for concurrency in (1, 4, 8):
            summary_path = trial_path / "local" / f"local-c{concurrency}.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.extend(
                item
                for result in summary.get("results", [])
                if (item := _one(result, concurrency, trial_path.name)) is not None
            )
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["concurrency"]].append(row)
    fields = (
        "total_wall_seconds",
        "bootstrap_to_first_model_seconds",
        "model_seconds",
        "tool_queue_and_harness_seconds",
        "verification_and_finalize_seconds",
        "outside_traced_span_seconds",
        "model_calls",
    )
    summary = {
        f"c{concurrency}": {field: _mean(group_rows, field) for field in fields}
        for concurrency, group_rows in sorted(grouped.items())
    }
    payload = {"report_version": "local-rollout-stage-analysis/v1", "episodes": rows, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# Local Data Plane stage timings", "", "Each value is the mean per Episode across all completed trials. `tool_queue_and_harness` is the sandbox span minus model-response latency, so it includes Pi tool execution and any scheduling gap.", "", "| concurrency | total wall(s) | bootstrap(s) | model(s) | tool/queue/harness(s) | verify+finalize(s) | outside trace(s) | model calls |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for concurrency, values in sorted(summary.items()):
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            concurrency,
            *(f"{values[field]:.2f}" for field in fields),
        ))
    lines.append("")
    args.output.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
