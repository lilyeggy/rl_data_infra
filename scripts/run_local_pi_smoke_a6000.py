#!/usr/bin/env python3
"""Run a Pi-direct baseline outside the Local Data Plane transaction.

This measures Pi + workspace + model latency only. It must not be labeled as
Local Data Plane end-to-end throughput; use benchmark_data_plane_a6000.py for
the complete Model Proxy/Trace/Verifier/Finalizer path.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path


PROMPT = "Edit only target.py so answer() returns 42. Do not change test_target.py. Run python3 test_target.py exactly once after editing."


def one(root: Path, pi_command: list[str], index: int) -> dict:
    workspace = root / f"workspace-{index}"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "target.py").write_text("def answer():\n    return 0\n", encoding="utf-8")
    (workspace / "test_target.py").write_text("from target import answer\nassert answer() == 42\n", encoding="utf-8")
    # Pi resolves its agent directory from HOME; keep the temporary HOME in a
    # conventional temp root so it does not inherit workspace-specific config.
    home = Path(tempfile.mkdtemp(prefix=f"pi-local-{index}-"))
    config = home / ".pi" / "agent" / "models.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"providers": {"local-vllm": {
        "baseUrl": os.environ.get("PI_BASE_URL", "http://127.0.0.1:8000/v1"), "api": "openai-completions", "apiKey": "local",
        "compat": {"supportsStrictMode": False, "supportsStore": False, "maxTokensField": "max_tokens"},
        "models": [{"id": "qwen2.5-coder-14b-instruct", "contextWindow": 32768, "maxTokens": 8192,
                    "compat": {"supportsStrictMode": False, "supportsStore": False, "maxTokensField": "max_tokens"}}]
    }}}), encoding="utf-8")
    command = pi_command + ["--provider", "local-vllm", "--model", "qwen2.5-coder-14b-instruct", "--mode", "json", "--print",
           "--no-session", "--no-context-files", "--no-extensions", "--no-skills", "--tools", "read,edit,bash",
           "--thinking", "minimal", PROMPT]
    started = time.monotonic()
    try:
        child_env = os.environ.copy()
        child_env.update({"HOME": str(home), "OPENAI_API_KEY": "local"})
        proc = subprocess.run(command, cwd=workspace, env=child_env,
                              capture_output=True, text=True, timeout=900, check=False)
        (root / f"pi-{index}.stdout.ndjson").write_text(proc.stdout, encoding="utf-8")
        (root / f"pi-{index}.stderr.log").write_text(proc.stderr, encoding="utf-8")
        verify = subprocess.run(["python3", "test_target.py"], cwd=workspace, capture_output=True, text=True, timeout=30, check=False)
        passed = verify.returncode == 0 and "return 42" in (workspace / "target.py").read_text(encoding="utf-8")
        return {"index": index, "passed": passed, "returncode": proc.returncode,
                "seconds": time.monotonic() - started, "stdout_bytes": len(proc.stdout),
                "stdout_tail": proc.stdout[-1200:], "stderr_tail": proc.stderr[-500:]}
    except subprocess.TimeoutExpired:
        return {"index": index, "passed": False, "returncode": 124, "seconds": time.monotonic() - started}
    finally:
        shutil.rmtree(home, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pi = shutil.which("pi")
    pi_command = ([pi] if pi else [
        "/home/f630/homePLUS/agentic/run/node22/bin/node",
        "/home/f630/homePLUS/agentic/run/node22/lib/node_modules/@earendil-works/pi-coding-agent/dist/cli.js",
    ])
    resource = args.output_dir / f"resource-local-c{args.concurrency}.csv"
    resource.write_text("timestamp,gpu,memory_used,memory_total,power\n", encoding="utf-8")
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            line = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw",
                                   "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False).stdout.strip()
            if line:
                with resource.open("a", encoding="utf-8") as handle:
                    handle.write(line.replace(" ", "") + "\n")
            stop.wait(2)

    started = time.monotonic()
    run_root = args.output_dir / f"run-c{args.concurrency}"
    run_root.mkdir(parents=True, exist_ok=True)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda i: one(run_root, pi_command, i), range(args.concurrency)))
    stop.set(); thread.join(timeout=3)
    wall = time.monotonic() - started
    payload = {"system": "pi-direct+vllm-baseline", "concurrency": args.concurrency, "wall_seconds": wall, "done": sum(r["passed"] for r in results),
               "total": len(results), "results": results}
    (args.output_dir / f"local-c{args.concurrency}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output_dir / f"local-c{args.concurrency}.log").write_text(
        f"Local/Pi smoke concurrency={args.concurrency}\nWall time: {wall:.0f}s\n"
        f"Done: {payload['done']}/{payload['total']}\n" + json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["done"] == payload["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
