#!/usr/bin/env python3
"""Benchmark the complete Local Data Plane + vLLM rollout transaction."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from src.capture import ModelEndpointKind
from src.contracts._json import sha256_json
from src.contracts.agent_episode import CaptureCapability
from src.contracts.execution_identity import ExecutionIdentity
from src.contracts.manifests import EvaluatorManifest, HarnessManifest, ModelManifest
from src.launchers import SandboxLimits, SandboxProfile, SandboxRuntime
from src.orchestration import (
    LocalExecutionOrchestrator,
    LocalExecutionSpec,
    RolloutPoolMetrics,
    RolloutPoolStage,
)


PROMPT = (
    "The complete initial contents of target.py are exactly: "
    "def answer():\\n    return 0\\n. Edit only target.py so answer() returns 42; "
    "use edit oldText `return 0` and newText `return 42`. Do not change test_target.py. "
    "Dependency policy: in your first tool-call turn issue only the edit call; after its successful "
    "result, issue exactly one bash call `python3 test_target.py` in a later turn."
)


def _run_one(
    args: argparse.Namespace,
    root: Path,
    index: int,
    *,
    stage_observer: Callable[[str], None] | None = None,
) -> dict:
    workspace = root / "workspaces" / f"workspace-{index}"
    output = root / "runs" / f"run-{index}"
    workspace.mkdir(parents=True)
    (workspace / "target.py").write_text("def answer():\n    return 0\n", encoding="utf-8")
    (workspace / "test_target.py").write_text(
        "from target import answer\nassert answer() == 42\n", encoding="utf-8"
    )
    run_id = f"data-plane-bench-{uuid.uuid4().hex[:12]}"
    manifest_digest = sha256_json({"benchmark": "deterministic-pi-smoke/v1"})
    sampling = {
        "thinking": "minimal",
        "tools": "pi-default",
        "seed": args.model_seed,
        "evidence_mode": (
            "RL_LOGPROBS" if args.capture_response_logprobs else "OBSERVABILITY_ONLY"
        ),
    }
    spec = LocalExecutionSpec(
        identity=ExecutionIdentity(
            run_id=run_id,
            task_id="deterministic-pi-smoke/v1",
            episode_id=f"episode-{run_id}",
            attempt_id=1,
            producer_id="local-docker",
            producer_version="local-docker-launcher/v1",
            policy_fingerprint=sha256_json({"policy": "pi-smoke/v1"}),
            sampling_fingerprint=sha256_json(sampling),
        ),
        harness_manifest=HarnessManifest(
            name="pi",
            version=args.pi_version,
            revision=f"pi/{args.pi_version}",
            config_digest=manifest_digest,
            hook_version="pi-json-wrapper/v1",
        ),
        model_manifest=ModelManifest(
            provider="self-hosted-vllm",
            model_id=args.model,
            revision=args.model_revision,
            sampling_config=sampling,
            tokenizer_revision=args.tokenizer_revision,
        ),
        evaluator_manifest=EvaluatorManifest(
            name="deterministic-python-verifier",
            revision="pi-smoke-verifier/v1",
            config_digest=manifest_digest,
        ),
        experiment_manifest_ref=args.experiment,
        image=args.image,
        image_digest=args.image_digest,
        workspace=str(workspace.resolve()),
        harness_argv=(
            "python3",
            "/opt/agent-data-plane/pi_data_plane_wrapper.py",
            PROMPT,
            "--model",
            args.model,
        ),
        verifier_argv=("python3", "test_target.py"),
        upstream_chat_completions_url=args.upstream_url,
        endpoint_kind=ModelEndpointKind.CONTROLLED,
        task_snapshot=sha256_json({"task": PROMPT, "files": "target-v1"}),
        capture_capabilities=frozenset({CaptureCapability.TOOL_IO}),
        capture_response_logprobs=args.capture_response_logprobs,
        default_sampling_seed=args.model_seed,
        docker_host_gateway=True,
        proxy_bind_host=args.proxy_bind_host,
        sandbox_profile=SandboxProfile(args.sandbox_profile),
        sandbox_runtime=SandboxRuntime(args.sandbox_runtime),
        limits=SandboxLimits(cpus=2, memory_mb=2048, pids=256, tmpfs_mb=512),
        harness_timeout_seconds=args.timeout,
        verifier_timeout_seconds=30,
        trace_id=f"trace-{run_id}",
    )
    started = time.monotonic()
    try:
        result = LocalExecutionOrchestrator().run(
            spec,
            output_dir=output,
            stage_observer=stage_observer,
        )
        episode = result.finalization.episode
        evidence = result.finalization.model_evidence
        completion_tokens = sum(
            int(item.backend.usage.get("completion_tokens", 0) or 0)
            for item in evidence
        )
        rl_usable_tokens = sum(
            len(item.backend.response_token_ids or ())
            for item in evidence
            if item.rl_usable_call
        )
        return {
            "index": index,
            "seconds": time.monotonic() - started,
            "passed": episode.outcome.verifier_status.value == "PASSED",
            "producer_status": result.producer_artifact.status.value,
            "integrity": episode.integrity.state.value,
            "verifier_status": episode.outcome.verifier_status.value,
            "event_count": len(episode.events),
            "model_call_count": len(evidence),
            "rl_usable_model_call_count": sum(
                item.rl_usable_call for item in evidence
            ),
            "model_completion_token_count": completion_tokens,
            "rl_usable_token_count": rl_usable_tokens,
            "execution_bundle_checksum": result.finalization.execution_bundle.checksum,
            "output_dir": str(output),
        }
    except Exception as exc:
        return {
            "index": index,
            "seconds": time.monotonic() - started,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "output_dir": str(output),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--upstream-url", default="http://127.0.0.1:8001/v1/chat/completions")
    parser.add_argument("--proxy-bind-host", default="172.17.0.1")
    parser.add_argument("--model", default="qwen2.5-coder-14b-instruct")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", default="qwen2.5-coder-tokenizer/v1")
    parser.add_argument("--pi-version", default="0.84.2")
    parser.add_argument("--sandbox-profile", choices=[item.value for item in SandboxProfile], default="FAST")
    parser.add_argument("--sandbox-runtime", choices=[item.value for item in SandboxRuntime], default="runc")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument(
        "--model-seed",
        type=int,
        default=20260831,
        help="fixed vLLM request seed for reproducible infrastructure ablations",
    )
    parser.add_argument(
        "--capture-response-logprobs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="request native token logprobs; required for RL-usable calls",
    )
    parser.add_argument(
        "--allow-verifier-failures",
        action="store_true",
        help="return success when the transaction is complete even if a task verifier fails",
    )
    parser.add_argument("--experiment", default="polar-fair-comparison/v1")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = args.output_dir / f"data-plane-c{args.concurrency}"
    root.mkdir(parents=False, exist_ok=False)
    resource_path = args.output_dir / f"resource-local-c{args.concurrency}.csv"
    resource_path.write_text("timestamp,gpu,memory_used,memory_total,power\n", encoding="utf-8")
    stopped = threading.Event()

    def sample_resources() -> None:
        while not stopped.is_set():
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.stdout.strip():
                with resource_path.open("a", encoding="utf-8") as handle:
                    handle.write(completed.stdout.strip().replace(" ", "") + "\n")
            stopped.wait(2)

    thread = threading.Thread(target=sample_resources, daemon=True)
    thread.start()
    pool_metrics = RolloutPoolMetrics()
    for index in range(args.concurrency):
        pool_metrics.register(str(index))

    def execute(index: int) -> dict:
        episode = str(index)
        pool_metrics.transition(episode, RolloutPoolStage.INIT)

        def observe(stage: str) -> None:
            pool_metrics.transition(episode, RolloutPoolStage(stage))

        result = _run_one(args, root, index, stage_observer=observe)
        pool_metrics.finish(
            episode,
            status=(
                "COMPLETED"
                if result.get("producer_status") == "COMPLETED"
                and result.get("integrity") == "COMPLETE"
                else "FAILED"
            ),
        )
        return result

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(execute, range(args.concurrency)))
    wall = time.monotonic() - started
    stopped.set()
    thread.join(timeout=3)
    durations = [item["seconds"] for item in results]
    payload = {
        "system": "local-data-plane+vllm",
        "evidence_mode": (
            "RL_LOGPROBS" if args.capture_response_logprobs else "OBSERVABILITY_ONLY"
        ),
        "concurrency": args.concurrency,
        "wall_seconds": wall,
        "completed": sum(
            item.get("producer_status") == "COMPLETED" and item.get("integrity") == "COMPLETE"
            for item in results
        ),
        "done": sum(item["passed"] for item in results),
        "total": len(results),
        "completed_model_completion_tokens": sum(
            item.get("model_completion_token_count", 0)
            for item in results
            if item.get("producer_status") == "COMPLETED" and item.get("integrity") == "COMPLETE"
        ),
        "verified_model_completion_tokens": sum(
            item.get("model_completion_token_count", 0)
            for item in results
            if item.get("passed")
        ),
        "verified_rl_usable_tokens": sum(
            item.get("rl_usable_token_count", 0)
            for item in results
            if item.get("passed")
        ),
        "session_p50_seconds": statistics.median(durations),
        "session_p95_seconds": sorted(durations)[max(0, int(len(durations) * 0.95) - 1)],
        "results": results,
    }
    payload["completed_token_goodput_per_second"] = (
        payload["completed_model_completion_tokens"] / wall if wall else 0.0
    )
    payload["verified_sft_token_goodput_per_second"] = (
        payload["verified_model_completion_tokens"] / wall if wall else 0.0
    )
    payload["verified_rl_token_goodput_per_second"] = (
        payload["verified_rl_usable_tokens"] / wall if wall else 0.0
    )
    summary = args.output_dir / f"local-c{args.concurrency}.json"
    summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    pool_metrics_path = args.output_dir / f"pool-metrics-local-c{args.concurrency}.json"
    pool_metrics_path.write_text(json.dumps(pool_metrics.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if args.allow_verifier_failures:
        return 0 if payload["completed"] == payload["total"] else 1
    return 0 if payload["done"] == payload["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
