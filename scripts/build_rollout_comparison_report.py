#!/usr/bin/env python3
"""Build an auditable Polar rollout report from benchmark/session artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


FAIRNESS_FIELDS = (
    "task_revision",
    "model_id",
    "model_revision",
    "harness_revision",
    "sampling_fingerprint",
    "vllm_endpoint_fingerprint",
)


def _number(value: str) -> float:
    return float(value.strip().replace("%", ""))


def benchmark(path: Path, concurrency: int) -> dict:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    wall_match = re.search(r"Wall time:?\s+([0-9.]+)s", text)
    wall = float(wall_match.group(1)) if wall_match else None
    row = re.search(r"^pi\s+(-?[0-9.]+)\s+([0-9]+)/([0-9]+)\s*$", text, re.MULTILINE)
    reward = float(row.group(1)) if row else None
    done = int(row.group(2)) if row else 0
    total = int(row.group(3)) if row else concurrency
    task = re.search(r"->\s+(polar-smoke-pi-\S+)", text)
    return {"benchmark": str(path), "task_id": task.group(1) if task else None,
            "wall_seconds": wall, "throughput_sessions_per_second": (done / wall if wall and done else None),
            "reward": reward, "done": done, "total": total, "completion_rate": done / total if total else 0.0}


def sessions(root: Path, task_id: str | None) -> dict:
    if not task_id:
        return {"session_count": 0, "positive_reward_count": 0, "trace_count": 0, "durations_seconds": [], "statuses": {}}
    files = list((root / f"task_{task_id}").rglob("ses*.json"))
    durations: list[float] = []
    statuses: dict[str, int] = {}
    positive = 0
    traces = 0
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        status = str(data.get("status", "UNKNOWN"))
        statuses[status] = statuses.get(status, 0) + 1
        meta = (data.get("trajectory") or {}).get("metadata") or {}
        evaluation = meta.get("evaluation") or {}
        reward = evaluation.get("outcome_reward")
        if isinstance(reward, (int, float)) and reward > 0:
            positive += 1
        traces += len((data.get("trajectory") or {}).get("traces") or [])
        timing = data.get("timing") or {}
        values = [timing.get(key, 0) for key in ("init_ms", "run_ms", "postrun_ms")]
        if any(isinstance(value, (int, float)) for value in values):
            durations.append(sum(float(value or 0) for value in values) / 1000.0)
    return {"session_count": len(files), "positive_reward_count": positive, "trace_count": traces,
            "durations_seconds": durations, "statuses": statuses}


def resources(path: Path) -> dict:
    rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if not row or row[0] == "timestamp":
                    continue
                try:
                    rows.append({"gpu": _number(row[1]), "memory_used": _number(row[2]),
                                 "memory_total": _number(row[3]), "power": _number(row[4])})
                except (IndexError, ValueError):
                    continue
    return {"samples": len(rows), "gpu_mean_percent": statistics.mean(r["gpu"] for r in rows) if rows else None,
            "gpu_peak_percent": max((r["gpu"] for r in rows), default=None),
            "memory_peak_mib": max((r["memory_used"] for r in rows), default=None),
            "power_mean_watts": statistics.mean(r["power"] for r in rows) if rows else None}


def local_smoke(root: Path, concurrency: int) -> dict:
    path = root / f"local-c{concurrency}.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    wall = data.get("wall_seconds")
    done = int(data.get("done", 0))
    total = int(data.get("total", concurrency))
    completed = int(data.get("completed", sum(
        item.get("producer_status") == "COMPLETED" and item.get("integrity") == "COMPLETE"
        for item in data.get("results", [])
        if isinstance(item, dict)
    )))
    durations = [float(item["seconds"]) for item in data.get("results", []) if isinstance(item.get("seconds"), (int, float))]
    return {"system": data.get("system"),
            "is_complete_data_plane": data.get("system") == "local-data-plane+vllm",
            "wall_seconds": wall,
            "throughput_sessions_per_second": completed / wall if wall and completed else None,
            "verified_throughput_per_second": done / wall if wall and done else 0.0,
            "completed": completed, "done": done, "total": total,
            "completion_rate": completed / total if total else 0.0,
            "verified_rate": done / total if total else 0.0,
            "session_p50_seconds": statistics.median(durations) if durations else None,
            "session_p95_seconds": sorted(durations)[max(0, int(len(durations) * .95) - 1)] if durations else None,
            "resources": resources(root / f"resource-local-c{concurrency}.csv")}


def fairness(contract_path: Path | None) -> dict:
    if contract_path is None or not contract_path.exists():
        return {
            "eligible": False,
            "reason": "missing comparison contract",
            "matches": {},
        }
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    local = contract.get("local_data_plane") or {}
    polar = contract.get("polar") or {}
    matches = {
        field: bool(local.get(field)) and local.get(field) == polar.get(field)
        for field in FAIRNESS_FIELDS
    }
    return {
        "eligible": all(matches.values()),
        "reason": (
            "all controlled workload fields match"
            if all(matches.values())
            else "one or more controlled workload fields differ or are missing"
        ),
        "matches": matches,
        "contract": contract,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--polar-root", type=Path, required=True)
    parser.add_argument("--local-summary", type=Path)
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--comparison-contract", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for concurrency in (1, 4, 8):
        b = benchmark(args.polar_root / f"benchmark-c{concurrency}.log", concurrency)
        s = sessions(args.polar_root / "rollout_results", b["task_id"])
        b["session"] = s
        b["resources"] = resources(args.polar_root / f"resource-c{concurrency}.csv")
        if s["durations_seconds"]:
            b["session_p50_seconds"] = statistics.median(s["durations_seconds"])
            b["session_p95_seconds"] = sorted(s["durations_seconds"])[max(0, int(len(s["durations_seconds"]) * .95) - 1)]
        rows.append(b)
    local_smoke_rows = [local_smoke(args.local_root, c) for c in (1, 4, 8)] if args.local_root else []
    local = None
    if args.local_summary and args.local_summary.exists():
        local = json.loads(args.local_summary.read_text(encoding="utf-8"))
    fairness_result = fairness(args.comparison_contract)
    local_system_valid = bool(local_smoke_rows) and all(
        row["is_complete_data_plane"] for row in local_smoke_rows
    )
    if not local_system_valid:
        fairness_result = {
            **fairness_result,
            "eligible": False,
            "reason": "local input is not labeled as complete Local Data Plane + vLLM",
        }
    speedups = []
    for concurrency, polar_row, local_row in zip((1, 4, 8), rows, local_smoke_rows):
        polar_verified = polar_row["session"]["positive_reward_count"]
        polar_verified_throughput = (
            polar_verified / polar_row["wall_seconds"]
            if polar_row["wall_seconds"] and polar_verified
            else 0.0
        )
        local_verified_throughput = local_row["verified_throughput_per_second"]
        speedups.append({
            "concurrency": concurrency,
            "polar_verified_throughput_per_second": polar_verified_throughput,
            "local_verified_throughput_per_second": local_verified_throughput,
            "local_over_polar_verified_speedup": (
                local_verified_throughput / polar_verified_throughput
                if fairness_result["eligible"] and polar_verified_throughput > 0
                else None
            ),
        })
    payload = {"report_version": "rollout-comparison/v3", "polar": rows,
               "local_smoke": local_smoke_rows, "local_baseline": local,
               "paired_comparison": (
                   bool(local_smoke_rows)
                   and local_system_valid
                   and all(r["done"] == r["total"] for r in local_smoke_rows)
                   and fairness_result["eligible"]
               ),
               "fairness": fairness_result,
               "verified_throughput_comparison": speedups,
               "comparison_note": "The compared systems are Local Data Plane + vLLM and Polar + vLLM. Speedup is valid only when the comparison contract passes."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fairness_lines = "\n".join(
        f"- `{field}`: {'MATCH' if matched else 'MISMATCH/MISSING'}"
        for field, matched in fairness_result["matches"].items()
    ) or "- comparison contract missing"
    args.output.write_text("# Local Data Plane + vLLM vs Polar + vLLM\n\n" +
        "> 本报告比较两套完整 Rollout 系统，不把 Local Data Plane 简写成纯 vLLM。只有公平性合同全部通过时才允许计算 speedup。\n\n" +
        "## Fairness gate\n\n" + fairness_lines +
        f"\n\n**Speedup eligible:** `{fairness_result['eligible']}` — {fairness_result['reason']}\n\n" +
        "## Polar smoke\n\n| 并发 | wall(s) | completed | verified | verified throughput/s | trace | GPU mean/peak | memory peak(MiB) |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n" +
        "\n".join(f"| {c} | {r['wall_seconds']} | {r['done']}/{r['total']} | {r['session']['positive_reward_count']}/{r['total']} | {(r['session']['positive_reward_count'] / r['wall_seconds']) if r['wall_seconds'] else 0:.4f} | {r['session']['trace_count']} | {r['resources']['gpu_mean_percent']}/{r['resources']['gpu_peak_percent']} | {r['resources']['memory_peak_mib']} |" for c, r in zip((1,4,8), rows)) +
        "\n\n## Local Data Plane + vLLM\n\n| 并发 | wall(s) | completed | verified | verified throughput/s | p50/p95 session(s) | GPU mean/peak | memory peak(MiB) |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n" +
        "\n".join(f"| {c} | {r['wall_seconds']:.1f} | {r['completed']}/{r['total']} | {r['done']}/{r['total']} | {r['verified_throughput_per_second']:.4f} | {r['session_p50_seconds']:.1f}/{r['session_p95_seconds']:.1f} | {r['resources']['gpu_mean_percent']}/{r['resources']['gpu_peak_percent']} | {r['resources']['memory_peak_mib']} |" for c, r in zip((1,4,8), local_smoke_rows)) +
        "\n\n## Verified throughput comparison\n\n| 并发 | Local/s | Polar/s | observed Local / Polar |\n|---:|---:|---:|---:|\n" +
        "\n".join(f"| {r['concurrency']} | {r['local_verified_throughput_per_second']:.4f} | {r['polar_verified_throughput_per_second']:.4f} | {r['local_over_polar_verified_speedup']:.2f}x |" if r['local_over_polar_verified_speedup'] is not None else f"| {r['concurrency']} | {r['local_verified_throughput_per_second']:.4f} | {r['polar_verified_throughput_per_second']:.4f} | N/A |" for r in speedups) +
        "\n\n## 结论边界\n\n- Polar 侧是 Polar Gateway/Rollout Server + vLLM；本地侧是 Data Plane + Harness/Workspace/Evidence/Verifier + vLLM。\n- 公平性 Gate 只证明配置一致；模型实际调用轮数仍会波动，单次 observed ratio 不是隔离后的编排加速比。\n- 公平性 Gate 未通过时，只展示各系统观测值，不宣称架构 speedup。\n- 纯 vLLM microbenchmark 必须单列，不能代替端到端 Rollout 对比。\n\n```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
