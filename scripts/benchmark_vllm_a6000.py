#!/usr/bin/env python3
"""Measure concurrent structured tool calls against the vLLM endpoint."""
from __future__ import annotations
import argparse, concurrent.futures, json, statistics, subprocess, time
from urllib.request import Request, urlopen

def one(url: str, model: str) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": "Read target.py using the read tool."}],
            "tools": [{"type": "function", "function": {"name": "read", "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
            "tool_choice": {"type": "function", "function": {"name": "read"}}, "stream": False,
            "max_tokens": 128, "temperature": 0}
    started = time.perf_counter()
    try:
        req = Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=120) as response:
            payload = json.load(response)
        choice = payload["choices"][0]
        return {"ok": bool(choice["message"].get("tool_calls")), "seconds": time.perf_counter() - started,
                "completion_tokens": payload.get("usage", {}).get("completion_tokens", 0)}
    except Exception as exc:
        return {"ok": False, "seconds": time.perf_counter() - started, "error": str(exc)}

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--requests", type=int, default=0); ap.add_argument("--output", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="qwen2.5-coder-14b-instruct"); args = ap.parse_args()
    total = args.requests or args.concurrency * 4
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda _: one(args.url, args.model), range(total)))
    wall = time.perf_counter() - started; lat = [x["seconds"] for x in results]
    report = {"backend": "vllm", "concurrency": args.concurrency, "requests": total, "completed": sum(x["ok"] for x in results),
              "wall_seconds": wall, "requests_per_second": total / wall if wall else 0,
              "latency_seconds": {"p50": statistics.median(lat), "p95": sorted(lat)[max(0, int(.95*len(lat))-1)], "max": max(lat)},
              "total_completion_tokens": sum(x.get("completion_tokens", 0) for x in results), "results": results}
    with open(args.output, "w") as f: json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2)); return 0 if report["completed"] == total else 1

if __name__ == "__main__": raise SystemExit(main())
