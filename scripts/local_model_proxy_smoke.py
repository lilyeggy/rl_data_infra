#!/usr/bin/env python3
"""Run Local Docker Launcher → Model Proxy → fake controlled model end to end."""

from __future__ import annotations

import argparse
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.capture import (
    EventJsonlReader,
    EventWriter,
    ModelCallEvidence,
    ModelEndpointKind,
    ModelEvidenceJsonlWriter,
    ModelProxyHttpServer,
    ModelProxyService,
    TraceRecorder,
)
from src.contracts.execution_identity import ExecutionIdentity
from src.launchers import (
    DockerLocalLauncher,
    LocalHarnessRequest,
    SandboxLimits,
    SandboxNetworkPolicy,
)
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
                    "id": "local-smoke-completion",
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "smoke-ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                    "agent_data_plane_evidence": {
                        "backend_model_revision": "fake-checkpoint-v0",
                        "prompt_token_ids": [10, 11],
                        "response_token_ids": [20],
                        "response_logprobs": [-0.1],
                    },
                }
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions"

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="alpine")
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/local-model-proxy-smoke")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    identity = ExecutionIdentity(
        run_id="run-local-proxy-smoke",
        task_id="task-local-proxy-smoke",
        episode_id="episode-local-proxy-smoke",
        attempt_id=1,
        producer_id="local-docker",
        producer_version="local-docker-launcher/v1",
        policy_fingerprint="a" * 64,
        sampling_fingerprint="b" * 64,
    )
    events_path = args.output_dir / "raw-events.jsonl"
    evidence_path = args.output_dir / "model-evidence.jsonl"
    artifact_path = args.output_dir / "producer-artifact.json"
    if any(path.exists() for path in (events_path, evidence_path, artifact_path)):
        raise SystemExit("output directory already contains smoke evidence")
    upstream = FakeControlledModel()
    upstream.start()
    proxy_access_token = secrets.token_urlsafe(32)
    proxy = ModelProxyHttpServer(
        ModelProxyService(
            identity=identity,
            endpoint_kind=ModelEndpointKind.CONTROLLED,
            upstream_chat_completions_url=upstream.url,
            evidence_writer=ModelEvidenceJsonlWriter(evidence_path),
            recorder=TraceRecorder(
                EventWriter(events_path),
                run_id=identity.run_id,
                episode_id=identity.episode_id,
                trace_id="trace-local-proxy-smoke",
            ),
            access_token=proxy_access_token,
        ),
        host="0.0.0.0",
        port=0,
    )
    proxy.start_in_thread()
    proxy_url = f"http://host.docker.internal:{proxy.address[1]}"
    request_body = json.dumps(
        {
            "model": "example-14b",
            "messages": [{"role": "user", "content": "smoke"}],
            "temperature": 0.7,
        },
        separators=(",", ":"),
    )
    try:
        artifact = DockerLocalLauncher().run(
            LocalHarnessRequest(
                identity=identity,
                image=args.image,
                image_digest=args.image_digest,
                harness_argv=(
                    "sh",
                    "-c",
                    "wget -qO- "
                    "--header='Content-Type: application/json' "
                    "--header=\"Authorization: Bearer $OPENAI_API_KEY\" "
                    f"--post-data='{request_body}' "
                    f"{proxy_url}/v1/chat/completions",
                ),
                host_workspace=str(args.workspace.resolve()),
                timeout_seconds=60,
                platform="linux/amd64",
                model_proxy_url=proxy_url,
                model_proxy_api_key=proxy_access_token,
                network_policy=SandboxNetworkPolicy.BRIDGE_UNRESTRICTED,
                limits=SandboxLimits(cpus=1, memory_mb=512, pids=64, tmpfs_mb=64),
            )
        )
    finally:
        proxy.close()
        upstream.close()

    artifact_path.write_text(
        json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    if artifact.status is not ProducerExecutionStatus.COMPLETED:
        raise SystemExit(f"local Harness failed: {artifact.issues or artifact.payload}")
    evidence_lines = evidence_path.read_text(encoding="utf-8").splitlines()
    evidence = ModelCallEvidence.from_dict(json.loads(evidence_lines[0]))
    events = EventJsonlReader.read(events_path)
    if len(evidence_lines) != 1 or not evidence.rl_usable_call:
        raise SystemExit("model evidence is not RL-usable")
    if len(events.events) != 2 or events.issues:
        raise SystemExit("model request/response trace is incomplete")
    print(
        json.dumps(
            {
                "status": "ok",
                "producer_artifact_checksum": artifact.checksum,
                "model_evidence_checksum": evidence.checksum,
                "event_count": len(events.events),
                "container_response": json.loads(str(artifact.payload["stdout"])),
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
