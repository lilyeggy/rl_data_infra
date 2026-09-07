#!/usr/bin/env python3
"""Run one real-model Docker Harness trace on a Linux GPU host.

This proves the production network boundary: Docker Harness -> host-only proxy
-> already-running local OpenAI model.  The result is intentionally reported as
observation-only when the upstream does not supply RL evidence fields.

Verbose mode (default) prints every stage: proxy startup, model call evidence,
harness stdout/stderr, workspace changes, verifier output, event stream and
finalized episode/bundle.  Pass --quiet for the old terse behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

from src.capture import ModelEndpointKind
from src.contracts._json import sha256_json
from src.contracts.agent_episode import CaptureCapability, IntegrityState
from src.contracts.execution_identity import ExecutionIdentity
from src.contracts.manifests import EvaluatorManifest, HarnessManifest, ModelManifest
from src.launchers import SandboxLimits
from src.orchestration import LocalExecutionOrchestrator, LocalExecutionSpec
from src.producers import ProducerExecutionStatus

_T0 = time.monotonic()


def _stamp() -> str:
    return f"[{time.monotonic() - _T0:7.2f}s]"


def _banner(title: str) -> None:
    print(f"\n{'=' * 78}\n== {title}\n{'=' * 78}", flush=True)


def _print_json(value: object, limit: int = 4000) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    if len(text) > limit:
        text = text[:limit] + f"\n... (truncated, {len(text)} chars total)"
    print(text, flush=True)


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"(could not read {path.name}: {exc})")
        return None


def _read_text(path: Path, limit: int = 4000) -> str:
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        return f"(could not read {path.name}: {exc})"
    if len(text) > limit:
        text = text[:limit] + f"\n... (truncated, {len(text)} chars total)"
    return text or "(empty)"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        required=True,
        help="registry image name, or a local immutable sha256:<image-id>",
    )
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only print the final one-page summary",
    )
    return parser.parse_args()


def _verbose_progress(title: str) -> None:
    print(f"{_stamp()} STAGE -> {title}", flush=True)


def _report_launch_plan(root: Path) -> None:
    _banner("1. LAUNCH PLAN (execution-run-manifest / launch-plan)")
    plan = _read_json(root / "launch-plan.json")
    if isinstance(plan, dict):
        summary = {
            key: plan.get(key)
            for key in (
                "container_name",
                "proxy_base_url",
                "launch_plan_checksum",
            )
            if key in plan
        }
        _print_json(summary | {k: v for k, v in plan.items() if k not in summary}, limit=6000)
    manifest = _read_json(root / "execution-run-manifest.json")
    if isinstance(manifest, dict):
        print("\n-- identity --")
        _print_json(manifest.get("identity"), limit=1500)


def _report_producer(root: Path) -> None:
    _banner("2. PRODUCER ARTIFACT (harness execution result)")
    producer = _read_json(root / "producer-artifact.json")
    if not isinstance(producer, dict):
        return
    payload = producer.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}
    harness = payload.get("harness_result", {})
    harness_payload = harness.get("payload", {}) if isinstance(harness, dict) else {}
    _print_json(
        {
            "status": producer.get("status"),
            "issues": producer.get("issues"),
            "harness_returncode": harness_payload.get("returncode"),
            "network_policy": harness_payload.get("network_policy"),
            "sandbox": harness_payload.get("runtime_evidence"),
            "limits": harness_payload.get("limits"),
        },
        limit=3000,
    )


def _report_workspace(workspace: Path) -> None:
    _banner("3. WORKSPACE AFTER RUN")
    if not workspace.exists():
        print("(workspace missing)")
        return
    for item in sorted(workspace.rglob("*")):
        if item.is_file():
            print(f"  {item.relative_to(workspace)}  ({item.stat().st_size} bytes)")
    model_response = workspace / "model-response.json"
    if model_response.exists():
        print("\n-- model-response.json (raw upstream reply stored by the harness) --")
        response = _read_json(model_response)
        if isinstance(response, dict):
            _print_json(
                {
                    "model": response.get("model"),
                    "usage": response.get("usage"),
                    "content": (
                        response.get("choices", [{}])[0].get("message", {}).get("content")
                    ),
                    "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
                }
            )
    answer = workspace / "answer.txt"
    if answer.exists():
        print("\n-- answer.txt --")
        print(_read_text(answer))


def _report_model_evidence(root: Path) -> None:
    _banner("4. MODEL EVIDENCE (per-call, proxy-observed)")
    path = root / "model-evidence.jsonl"
    try:
        lines = [line for line in path.read_text().splitlines() if line.strip()]
    except OSError as exc:
        print(f"(could not read model-evidence.jsonl: {exc})")
        return
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            print(f"call {index}: UNPARSEABLE line")
            continue
        summary = {
            key: record.get(key)
            for key in (
                "schema_version",
                "endpoint_kind",
                "model_revision",
                "rl_usable_call",
                "issues",
                "request_event_id",
                "response_event_id",
            )
            if key in record
        }
        usage = record.get("token_usage") or record.get("usage")
        if usage:
            summary["token_usage"] = usage
        request = record.get("request") or {}
        if isinstance(request, dict):
            summary["request_keys"] = sorted(request.keys())
        response = record.get("response") or {}
        if isinstance(response, dict):
            summary["response_keys"] = sorted(response.keys())
            choices = response.get("choices") or []
            if choices:
                message = choices[0].get("message", {})
                summary["response_content"] = message.get("content")
            summary["logprobs_present"] = any(
                choice.get("logprobs") is not None for choice in choices
            )
        print(f"\n-- model call {index} --")
        _print_json(summary)


def _report_events(root: Path) -> None:
    _banner("5. RAW EVENT STREAM (immutable TraceEvents)")
    path = root / "raw-events.jsonl"
    try:
        lines = [line for line in path.read_text().splitlines() if line.strip()]
    except OSError as exc:
        print(f"(could not read raw-events.jsonl: {exc})")
        return
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            print("  UNPARSEABLE event line")
            continue
        attrs = record.get("attributes") or {}
        brief = {
            "sequence": record.get("sequence"),
            "event_id": record.get("event_id"),
            "event_type": record.get("event_type"),
            "component": record.get("component"),
            "status": record.get("status"),
            "span_id": record.get("span_id"),
        }
        interesting = {
            key: attrs[key]
            for key in ("tool_name", "task_status", "execution_validity", "termination_reason",
                        "verifier_status", "score", "verifier", "exit_code")
            if key in attrs
        }
        if interesting:
            brief["attributes"] = interesting
        print(
            json.dumps(brief, ensure_ascii=False, sort_keys=True)
        )


def _report_verifier(root: Path) -> None:
    _banner("6. VERIFIER REPORT (content-addressed artifact)")
    refs = _read_json(root / "artifact-refs.json")
    if not isinstance(refs, list):
        return
    objects = root / "objects"
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        print(
            f"  {ref.get('kind'):<18} {ref.get('sha256','')[:16]}  "
            f"{ref.get('size_bytes','-')} bytes"
        )
        if ref.get("kind") == "verifier-report":
            report_path = objects / str(ref.get("sha256"))
            print("\n-- verifier-report payload --")
            _print_json(_read_json(report_path), limit=3000)


def _report_finalized(root: Path) -> None:
    _banner("7. FINALIZED EPISODE / EXECUTION BUNDLE")
    fin = _read_json(root / "finalized" / "finalization.json")
    _print_json(fin)
    episode = _read_json(root / "finalized" / "episode.json")
    if isinstance(episode, dict):
        outcome = episode.get("outcome", {})
        print("\n-- episode outcome --")
        _print_json(
            {
                "integrity": episode.get("integrity"),
                "outcome": outcome,
                "event_count": len(episode.get("events", []) or []),
            },
            limit=2000,
        )
    bundle = _read_json(root / "finalized" / "execution-bundle.json")
    if isinstance(bundle, dict):
        print("\n-- execution bundle --")
        _print_json(
            {
                key: bundle.get(key)
                for key in ("schema_version", "identity", "episode_checksum", "decision",
                            "artifact_refs", "certification")
                if key in bundle
            },
            limit=2500,
        )


def main() -> int:
    args = _arguments()
    verbose = not args.quiet
    args.workspace.mkdir(parents=True, exist_ok=True)
    run_id = f"a6000-live-{uuid.uuid4().hex[:12]}"
    digest = sha256_json({"kind": "a6000-live-smoke"})
    policy_fingerprint = "29bd620e48e43422aba6cb545382a36ac18569111e2b9107b258ee6d23f7ed08"
    sampling_fingerprint = sha256_json({"max_tokens": 4, "temperature": 0})
    model_request = json.dumps(
        {
            "model": args.model_id,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "max_tokens": 4,
            "temperature": 0,
        },
        separators=(",", ":"),
    )
    if verbose:
        _banner("0. SMOKE PLAN")
        _print_json(
            {
                "run_id": run_id,
                "image": args.image,
                "image_digest": args.image_digest[:32] + "...",
                "upstream_url": args.upstream_url,
                "model_id": args.model_id,
                "workspace": str(args.workspace.resolve()),
                "output_dir": str(args.output_dir.resolve()),
                "model_request_body": json.loads(model_request),
                "endpoint_kind": "CONTROLLED (native token/logprob capture on)",
            }
        )
    model_headers = (
        "headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + "
        "os.environ['OPENAI_API_KEY']}"
    )
    model_call = (
        "request = urllib.request.Request("
        "os.environ['OPENAI_BASE_URL'] + '/v1/chat/completions', "
        f"data={model_request!r}.encode(), headers=headers, method='POST')"
    )
    trace_headers = (
        "trace_headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + "
        "os.environ['AGENT_TRACE_API_KEY']}"
    )
    harness = "\n".join(
        (
            "import json, os, pathlib, urllib.request",
            model_headers,
            model_call,
            "response = json.loads(urllib.request.urlopen(request, timeout=120).read())",
            (
                "pathlib.Path('/workspace/model-response.json').write_text("
                "json.dumps(response, ensure_ascii=False))"
            ),
            trace_headers,
            (
                "event = {'event_type': 'TOOL_CALL', 'component': 'TOOL', "
                "'status': 'STARTED', 'span_id': 'span-live-check', "
                "'parent_span_id': None, 'attributes': {'tool_name': 'live-model-check', "
                "'arguments': {}}, 'artifact_refs': [], "
                "'attempt': int(os.environ['AGENT_ATTEMPT_ID'])}"
            ),
            (
                "trace = urllib.request.Request(os.environ['AGENT_TRACE_URL'], "
                "data=json.dumps(event).encode(), headers=trace_headers, method='POST')"
            ),
            "urllib.request.urlopen(trace, timeout=30).read()",
            (
                "event.update({'event_type': 'TOOL_RESULT', 'status': 'SUCCEEDED', "
                "'attributes': {'tool_name': 'live-model-check', 'exit_code': 0}})"
            ),
            (
                "trace = urllib.request.Request(os.environ['AGENT_TRACE_URL'], "
                "data=json.dumps(event).encode(), headers=trace_headers, method='POST')"
            ),
            "urllib.request.urlopen(trace, timeout=30).read()",
            "pathlib.Path('/workspace/answer.txt').write_text(response['choices'][0]['message']['content'])",
        )
    )
    if verbose:
        print("\n-- harness program (runs inside Docker) --")
        print(harness)
    spec = LocalExecutionSpec(
        identity=ExecutionIdentity(
            run_id=run_id,
            task_id="live-model-connectivity",
            episode_id=f"episode-{run_id}",
            attempt_id=1,
            producer_id="local-docker",
            producer_version="local-docker-launcher/v1",
            policy_fingerprint=policy_fingerprint,
            sampling_fingerprint=sampling_fingerprint,
        ),
        harness_manifest=HarnessManifest(
            name="a6000-live-smoke-harness",
            version="v1",
            revision="live-smoke-r1",
            config_digest=digest,
            hook_version="harness-event-ingress/v1",
        ),
        model_manifest=ModelManifest(
            provider="self-hosted",
            model_id=args.model_id,
            revision="existing-a6000-service",
            sampling_config={"temperature": 0},
            tokenizer_revision="server-managed",
        ),
        evaluator_manifest=EvaluatorManifest(
            name="file-exists-verifier",
            revision="file-exists-v1",
            config_digest=digest,
        ),
        experiment_manifest_ref="a6000-live-smoke",
        image=args.image,
        image_digest=args.image_digest,
        workspace=str(args.workspace.resolve()),
        harness_argv=("python", "-c", harness),
        verifier_argv=(
            "python",
            "-c",
            "from pathlib import Path; assert Path('/workspace/answer.txt').exists()",
        ),
        upstream_chat_completions_url=args.upstream_url,
        endpoint_kind=ModelEndpointKind.CONTROLLED,
        task_snapshot=sha256_json({"task": "local-model connectivity"}),
        capture_capabilities=frozenset({CaptureCapability.TOOL_IO}),
        docker_host_gateway=True,
        proxy_bind_host="172.17.0.1",
        limits=SandboxLimits(cpus=1, memory_mb=1024, pids=128, tmpfs_mb=64),
        harness_timeout_seconds=180,
        verifier_timeout_seconds=30,
        trace_id=f"trace-{run_id}",
    )
    if verbose:
        _banner("RUN: orchestrator transaction (proxy -> docker harness -> verifier -> finalize)")
    result = LocalExecutionOrchestrator().run(
        spec,
        output_dir=args.output_dir,
        stage_observer=_verbose_progress if verbose else None,
    )
    if verbose:
        _report_launch_plan(Path(result.output_dir))
        _report_producer(Path(result.output_dir))
        _report_workspace(args.workspace)
        _report_model_evidence(Path(result.output_dir))
        _report_events(Path(result.output_dir))
        _report_verifier(Path(result.output_dir))
        _report_finalized(Path(result.output_dir))
    if result.producer_artifact.status is not ProducerExecutionStatus.COMPLETED:
        raise SystemExit(f"Harness failed: {result.producer_artifact.status.value}")
    if result.finalization.episode.integrity.state is not IntegrityState.COMPLETE:
        raise SystemExit("episode did not reach COMPLETE integrity")
    evidence = result.finalization.model_evidence
    _banner("SUMMARY")
    _print_json(
        {
            "status": "ok",
            "run_id": run_id,
            "episode_checksum": result.finalization.episode.checksum,
            "execution_bundle_checksum": result.finalization.execution_bundle.checksum,
            "verifier_status": result.finalization.episode.outcome.verifier_status.value,
            "model_call_count": len(evidence),
            "rl_usable_calls": sum(item.rl_usable_call for item in evidence),
            "model_evidence_issues": [issue for item in evidence for issue in item.issues],
            "output_dir": str(args.output_dir),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
