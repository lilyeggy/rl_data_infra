#!/usr/bin/env python3
"""Sample host memory and per-role process footprint while a run is in flight.

Ray's node-memory OOM killer is what ended the first stage G attempt, so the
per-phase host footprint needs to be observable rather than inferred. This
loops until killed, appending one JSON object per sample.

Only processes owned by the invoking user are measured: this is read-only
instrumentation and must never reach into another user's job.
"""

import json
import os
import sys
import time

INTERVAL_SECONDS = 5.0

# Ordered longest-match-first: several of these strings appear inside others.
ROLE_PATTERNS = (
    ("ray::TaskRunner", "driver"),
    ("ray::WorkerDict", "fsdp_worker"),
    ("ray::PiAgentLoopWorker", "agent_loop"),
    ("ray::RewardLoopWorker", "reward_loop"),
    ("ray::GlobalRequestLoadBalancer", "load_balancer"),
    ("vLLMHttpServer", "vllm_http"),
    ("EngineCore", "vllm_engine"),
    ("gcs_server", "gcs"),
    ("raylet", "raylet"),
    ("ray::", "ray_other"),
)


def meminfo() -> dict:
    wanted = ("MemTotal", "MemAvailable", "MemFree", "SwapTotal", "SwapFree", "Cached")
    out = {}
    with open("/proc/meminfo") as handle:
        for line in handle:
            key, _, value = line.partition(":")
            if key in wanted:
                out[key] = int(value.split()[0]) // 1024  # MiB
    return out


def role_of(cmdline: str) -> str | None:
    for needle, role in ROLE_PATTERNS:
        if needle in cmdline:
            return role
    return "other_python" if "python" in cmdline else None


def sample() -> dict:
    uid = os.getuid()
    by_role: dict[str, list[float]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            if os.stat(f"/proc/{pid}").st_uid != uid:
                continue
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                cmdline = handle.read().replace(b"\0", b" ").decode("utf-8", "replace")
            role = role_of(cmdline)
            if role is None:
                continue
            with open(f"/proc/{pid}/status") as handle:
                rss = next(
                    int(line.split()[1]) for line in handle if line.startswith("VmRSS:")
                )
        except (OSError, StopIteration):
            continue
        by_role.setdefault(role, []).append(rss / 1024 / 1024)  # GiB

    roles = {
        role: {
            "count": len(values),
            "total_gib": round(sum(values), 2),
            "max_gib": round(max(values), 2),
        }
        for role, values in sorted(by_role.items())
    }
    return {
        "t": time.strftime("%H:%M:%S"),
        "epoch": round(time.time(), 1),
        "meminfo_mib": meminfo(),
        "roles": roles,
        "mine_total_gib": round(sum(v["total_gib"] for v in roles.values()), 2),
    }


def main() -> int:
    out_path = sys.argv[1]
    with open(out_path, "a", buffering=1) as out:
        while True:
            try:
                out.write(json.dumps(sample()) + "\n")
            except Exception as exc:  # keep sampling; never take the run down
                out.write(json.dumps({"t": time.strftime("%H:%M:%S"), "error": str(exc)}) + "\n")
            time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
