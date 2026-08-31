#!/usr/bin/env python3
"""Run one real-model Docker Harness trace on a Linux GPU host.

This proves the production network boundary: Docker Harness -> host-only proxy
-> already-running local OpenAI model.  The result is intentionally reported as
observation-only when the upstream does not supply RL evidence fields.
"""

from __future__ import annotations

import argparse
import json
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
    return parser.parse_args()


def main() -> int:
    args = _arguments()
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
    result = LocalExecutionOrchestrator().run(spec, output_dir=args.output_dir)
    if result.producer_artifact.status is not ProducerExecutionStatus.COMPLETED:
        raise SystemExit(f"Harness failed: {result.producer_artifact.status.value}")
    if result.finalization.episode.integrity.state is not IntegrityState.COMPLETE:
        raise SystemExit("episode did not reach COMPLETE integrity")
    evidence = result.finalization.model_evidence
    print(
        json.dumps(
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
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
