#!/usr/bin/env python3
"""Replay a frozen model workload through vLLM directly or the evidence proxy.

This deliberately does not run an agent harness.  It isolates the overhead of
the Data Plane's model evidence boundary from variable trajectory length,
tool-use, sandbox startup, and verifier work.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Mapping

from src.capture.event_writer import EventWriter
from src.capture.model_proxy import ModelEndpointKind, ModelEvidenceJsonlWriter
from src.capture.model_proxy_http import ModelProxyHttpServer, ModelProxyService
from src.capture.recorder import TraceRecorder
from src.contracts._json import sha256_json
from src.contracts.execution_identity import ExecutionIdentity


FROZEN_MESSAGES = [
    {
        "role": "system",
        "content": "You are a precise coding assistant. Return only the requested code.",
    },
    {
        "role": "user",
        "content": (
            "Implement a Python function is_even(n: int) -> bool. "
            "It must return True exactly when n is divisible by two."
        ),
    },
]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _post(url: str, payload: Mapping[str, Any], authorization: str | None) -> tuple[int, dict[str, Any]]:
    headers = {"Content-Type": "application/json", "User-Agent": "agent-data-plane/1.0"}
    if authorization:
        headers["Authorization"] = authorization
    request = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode(), method="POST", headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            raw, status = response.read(), response.status
    except urllib.error.HTTPError as error:
        raw, status = error.read(), error.code
    try:
        body = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        body = {"error": {"type": "invalid_json"}}
    return status, body if isinstance(body, dict) else {"error": {"type": "invalid_shape"}}


def _request_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": FROZEN_MESSAGES,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "stream": False,
        # Keep upstream computation semantically identical in both arms.  In
        # proxy mode these fields are also what is persisted as RL evidence.
        "logprobs": True,
        "top_logprobs": 0,
    }


def _single_request(url: str, payload: Mapping[str, Any], authorization: str | None, index: int) -> dict[str, Any]:
    started = time.monotonic()
    try:
        status, body = _post(url, payload, authorization)
        seconds = time.monotonic() - started
        usage = body.get("usage") if isinstance(body.get("usage"), Mapping) else {}
        return {
            "index": index,
            "status": status,
            "seconds": seconds,
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "ok": 200 <= status < 300,
            "error": None if 200 <= status < 300 else body.get("error"),
        }
    except Exception as error:  # Preserve failures as benchmark facts.
        return {"index": index, "status": 0, "seconds": time.monotonic() - started,
                "completion_tokens": 0, "ok": False, "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("direct", "proxy"), required=True)
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.concurrency < 1 or args.requests < 1 or args.max_tokens < 1:
        parser.error("concurrency, requests and max-tokens must be positive")

    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = _request_payload(args)
    target_url, authorization, server = args.upstream_url, None, None
    if args.mode == "proxy":
        identity = ExecutionIdentity(
            run_id=f"proxy-replay-{uuid.uuid4().hex[:12]}", task_id="fixed-model-replay/v1",
            episode_id="shared-proxy-episode", attempt_id=1, producer_id="proxy-replay",
            producer_version="benchmark_proxy_replay_a6000/v1",
            policy_fingerprint=sha256_json({"model_revision": args.model_revision}),
            sampling_fingerprint=sha256_json(
                {"temperature": 0, "seed": args.seed, "max_tokens": args.max_tokens}
            ),
        )
        recorder = TraceRecorder(EventWriter(root / "events.jsonl"), run_id=identity.run_id,
                                 episode_id=identity.episode_id, trace_id=f"trace-{identity.run_id}")
        service = ModelProxyService(
            identity=identity, endpoint_kind=ModelEndpointKind.CONTROLLED,
            upstream_chat_completions_url=args.upstream_url,
            evidence_writer=ModelEvidenceJsonlWriter(root / "model-evidence.jsonl"), recorder=recorder,
            access_token="fixed-workload-benchmark", backend_model_revision=args.model_revision,
            capture_response_logprobs=True, default_sampling_seed=args.seed,
            max_calls=args.requests + 4,
        )
        server = ModelProxyHttpServer(service, host="127.0.0.1", port=0)
        server.start_in_thread()
        host, port = server.address
        target_url, authorization = f"http://{host}:{port}/v1/chat/completions", "Bearer fixed-workload-benchmark"

    started = time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(_single_request, target_url, payload, authorization, index)
                       for index in range(args.requests)]
            results = [future.result() for future in futures]
    finally:
        if server is not None:
            server.close()
    wall_seconds = time.monotonic() - started
    successful = [item for item in results if item["ok"]]
    latencies = [float(item["seconds"]) for item in successful]
    completion_tokens = sum(int(item["completion_tokens"]) for item in successful)
    summary = {
        "benchmark": "fixed-model-replay/v1", "mode": args.mode, "upstream_url": args.upstream_url,
        "model": args.model, "model_revision": args.model_revision, "seed": args.seed,
        "concurrency": args.concurrency, "requests": args.requests, "max_tokens": args.max_tokens,
        "request_contract": payload, "wall_seconds": wall_seconds,
        "success_count": len(successful), "failure_count": len(results) - len(successful),
        "request_throughput_per_second": len(successful) / wall_seconds if wall_seconds else 0,
        "completion_tokens": completion_tokens,
        "completion_token_goodput_per_second": completion_tokens / wall_seconds if wall_seconds else 0,
        "e2e_latency_ms": {
            "mean": statistics.mean(latencies) * 1000 if latencies else 0,
            "p50": _percentile(latencies, 50) * 1000,
            "p95": _percentile(latencies, 95) * 1000,
        },
        "results": results,
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"results", "request_contract"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
