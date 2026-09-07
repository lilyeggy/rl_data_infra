#!/usr/bin/env python3
"""Exercise the complete local transaction with Docker and a fake model."""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.capture import ModelEndpointKind
from src.contracts._json import sha256_json
from src.contracts.agent_episode import CaptureCapability, IntegrityState
from src.contracts.execution_identity import ExecutionIdentity
from src.contracts.manifests import EvaluatorManifest, HarnessManifest, ModelManifest
from src.launchers import SandboxLimits
from src.orchestration import LocalExecutionOrchestrator, LocalExecutionSpec
from src.producers import ProducerExecutionStatus


class FakeControlledModel:
    def __init__(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                payload = {
                    "id": "one-shot-smoke",
                    "model": request["model"],
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "write ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2},
                    "agent_data_plane_evidence": {
                        "backend_model_revision": "fake-checkpoint-v0",
                        "prompt_token_ids": [10, 11],
                        "response_token_ids": [20, 21],
                        "response_logprobs": [-0.1, -0.2],
                    },
                }
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions"

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="alpine")
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.output_dir.parent / f"{args.output_dir.name}-workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "task.txt").write_text("write answer.txt\n", encoding="utf-8")
    digest = sha256_json({})
    model = FakeControlledModel()
    model.start()
    tool_call = json.dumps(
        {
            "event_type": "TOOL_CALL",
            "component": "TOOL",
            "status": "STARTED",
            "span_id": "span-write",
            "parent_span_id": None,
            "attributes": {"tool_name": "shell", "arguments": {"cmd": "write answer"}},
            "artifact_refs": [],
            "attempt": 1,
        },
        separators=(",", ":"),
    )
    tool_result = json.dumps(
        {
            "event_type": "TOOL_RESULT",
            "component": "TOOL",
            "status": "SUCCEEDED",
            "span_id": "span-write",
            "parent_span_id": None,
            "attributes": {"tool_name": "shell", "exit_code": 0},
            "artifact_refs": [],
            "attempt": 1,
        },
        separators=(",", ":"),
    )
    model_request = json.dumps(
        {
            "model": "example-14b",
            "messages": [{"role": "user", "content": "write answer"}],
            "temperature": 0.7,
        },
        separators=(",", ":"),
    )
    harness_script = "\n".join(
        (
            "set -eu",
            "wget -qO /tmp/model.json --header='Content-Type: application/json' "
            '--header="Authorization: Bearer $OPENAI_API_KEY" '
            f"--post-data='{model_request}' \"$OPENAI_BASE_URL/v1/chat/completions\"",
            "wget -qO- --header='Content-Type: application/json' "
            '--header="Authorization: Bearer $AGENT_TRACE_API_KEY" '
            f"--post-data='{tool_call}' \"$AGENT_TRACE_URL\" >/dev/null",
            "printf 'ok\\n' > /workspace/answer.txt",
            "wget -qO- --header='Content-Type: application/json' "
            '--header="Authorization: Bearer $AGENT_TRACE_API_KEY" '
            f"--post-data='{tool_result}' \"$AGENT_TRACE_URL\" >/dev/null",
            "cat /tmp/model.json",
        )
    )
    spec = LocalExecutionSpec(
        identity=ExecutionIdentity(
            run_id="run-local-one-shot-smoke",
            task_id="task-local-one-shot-smoke",
            episode_id="episode-local-one-shot-smoke",
            attempt_id=1,
            producer_id="local-docker",
            producer_version="local-docker-launcher/v1",
            policy_fingerprint="a" * 64,
            sampling_fingerprint="b" * 64,
        ),
        harness_manifest=HarnessManifest(
            name="shell-smoke-harness",
            version="v1",
            revision="shell-smoke-r1",
            config_digest=digest,
            hook_version="harness-event-ingress/v1",
        ),
        model_manifest=ModelManifest(
            provider="self-hosted",
            model_id="example-14b",
            revision="fake-checkpoint-v0",
            sampling_config={"temperature": 0.7},
            tokenizer_revision="fake-tokenizer-v0",
        ),
        evaluator_manifest=EvaluatorManifest(
            name="file-exists-verifier",
            revision="file-exists-v1",
            config_digest=digest,
        ),
        experiment_manifest_ref="experiment-local-one-shot-smoke",
        image=args.image,
        image_digest=args.image_digest,
        workspace=str(workspace.resolve()),
        harness_argv=("sh", "-c", harness_script),
        verifier_argv=("sh", "-c", "test -f /workspace/answer.txt"),
        upstream_chat_completions_url=model.url,
        endpoint_kind=ModelEndpointKind.CONTROLLED,
        task_snapshot=sha256_json({"task": "write answer.txt"}),
        capture_capabilities=frozenset({CaptureCapability.TOOL_IO}),
        limits=SandboxLimits(cpus=1, memory_mb=512, pids=64, tmpfs_mb=64),
        harness_timeout_seconds=60,
        verifier_timeout_seconds=30,
        trace_id="trace-local-one-shot-smoke",
    )
    try:
        result = LocalExecutionOrchestrator().run(spec, output_dir=args.output_dir)
    finally:
        model.close()
    if result.producer_artifact.status is not ProducerExecutionStatus.COMPLETED:
        raise SystemExit(f"producer failed: {result.producer_artifact.to_dict()}")
    if result.finalization.episode.integrity.state is not IntegrityState.COMPLETE:
        raise SystemExit("episode did not assemble with complete integrity")
    if not result.finalization.model_evidence[0].rl_usable_call:
        raise SystemExit("fake controlled model evidence was not RL-usable")
    print(
        json.dumps(
            {
                "status": "ok",
                "episode_checksum": result.finalization.episode.checksum,
                "execution_bundle_checksum": result.finalization.execution_bundle.checksum,
                "verifier_report_checksum": (
                    result.finalization.execution_bundle.verifier_report_checksum
                ),
                "artifact_count": len(result.finalization.episode.artifact_refs),
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
