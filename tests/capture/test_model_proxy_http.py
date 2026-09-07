from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.capture import (
    EventJsonlReader,
    EventWriter,
    HarnessEventIngress,
    HarnessTraceClient,
    ModelCallEvidence,
    ModelEndpointKind,
    ModelEvidenceJsonlWriter,
    ModelProxyHttpServer,
    ModelProxyService,
    TraceRecorder,
)
from src.contracts.execution_identity import ExecutionIdentity
from src.contracts.trace_event import EventComponent, EventStatus, EventType


def _identity() -> ExecutionIdentity:
    return ExecutionIdentity(
        run_id="run-http-proxy",
        task_id="task-http-proxy",
        episode_id="episode-http-proxy",
        attempt_id=1,
        producer_id="local-docker",
        producer_version="local-docker-launcher/v1",
        policy_fingerprint="a" * 64,
        sampling_fingerprint="b" * 64,
    )


class _FakeUpstream:
    def __init__(self) -> None:
        requests: list[dict] = []
        self.requests = requests

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                request = json.loads(self.rfile.read(length))
                requests.append(request)
                payload = {
                    "id": "completion-1",
                    "model": request["model"],
                    "choices": [{"message": {"role": "assistant", "content": "done"}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                    "agent_data_plane_evidence": {
                        "backend_model_revision": "checkpoint-0001",
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
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class ModelProxyHttpTest(unittest.TestCase):
    def test_sse_replay_splits_parallel_tool_calls_into_deltas(self) -> None:
        body = ModelProxyService.sse_body({
            "id": "completion-tools",
            "model": "example-14b",
            "choices": [{
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"index": 4, "id": "call-a", "type": "function",
                         "function": {"name": "read", "arguments": '{"path":"a"}'}},
                        {"index": 4, "id": "call-b", "type": "function",
                         "function": {"name": "grep", "arguments": '{"pattern":"x"}'}},
                    ],
                },
            }],
        }).decode()
        frames = [json.loads(item) for item in body.split("data: ")[1:] if item.strip() != "[DONE]"]
        tool_frames = [frame for frame in frames if frame["choices"][0]["delta"].get("tool_calls")]
        declaration_frames = [
            frame for frame in tool_frames
            if frame["choices"][0]["delta"]["tool_calls"][0].get("id")
        ]
        self.assertEqual(
            [item["index"] for item in declaration_frames[0]["choices"][0]["delta"]["tool_calls"]], [0]
        )
        self.assertEqual(
            [item["index"] for item in declaration_frames[1]["choices"][0]["delta"]["tool_calls"]], [1]
        )

    def test_forwards_openai_shape_and_persists_trace_and_native_policy_evidence(self) -> None:
        upstream = _FakeUpstream()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events_path = root / "events.jsonl"
            evidence_path = root / "model-evidence.jsonl"
            service = ModelProxyService(
                identity=_identity(),
                endpoint_kind=ModelEndpointKind.CONTROLLED,
                upstream_chat_completions_url=upstream.url,
                evidence_writer=ModelEvidenceJsonlWriter(evidence_path),
                recorder=TraceRecorder(
                    EventWriter(events_path),
                    run_id="run-http-proxy",
                    episode_id="episode-http-proxy",
                    trace_id="trace-http-proxy",
                ),
                access_token="execution-proxy-token",
            )
            proxy = ModelProxyHttpServer(
                service,
                harness_ingress=HarnessEventIngress(
                    identity=_identity(),
                    recorder=service.recorder,
                ),
            )
            proxy.start_in_thread()
            try:
                payload = {
                    "model": "example-14b",
                    "messages": [
                        {
                            "role": "user",
                            "content": "fix it",
                            "authorization": "must-not-persist",
                        }
                    ],
                    "temperature": 0.7,
                }
                request = urllib.request.Request(
                    f"http://127.0.0.1:{proxy.address[1]}/v1/chat/completions",
                    data=json.dumps(payload).encode(),
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": "Bearer execution-proxy-token",
                    },
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    result = json.loads(response.read())
                emitted = HarnessTraceClient(
                    endpoint=(f"http://127.0.0.1:{proxy.address[1]}/v1/harness/events"),
                    access_token="execution-proxy-token",
                ).emit(
                    event_type=EventType.HARNESS_DECISION,
                    component=EventComponent.HARNESS,
                    status=EventStatus.SUCCEEDED,
                    span_id="span-decision",
                    attributes={"decision": "continue", "api_key": "hidden"},
                )
            finally:
                proxy.close()
                upstream.close()

            self.assertEqual(result["choices"][0]["message"]["content"], "done")
            events = EventJsonlReader.read(events_path).events
            self.assertEqual(len(events), 3)
            self.assertEqual(events[-1].event_type, EventType.HARNESS_DECISION)
            self.assertEqual(events[-1].attributes["api_key"], "[REDACTED]")
            self.assertEqual(emitted["event_id"], events[-1].event_id)
            lines = evidence_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            evidence = ModelCallEvidence.from_dict(json.loads(lines[0]))
            self.assertTrue(evidence.rl_usable_call)
            self.assertEqual(evidence.request.messages[0]["authorization"], "[REDACTED]")
            self.assertNotIn("must-not-persist", evidence_path.read_text(encoding="utf-8"))
            self.assertTrue(upstream.requests[0]["logprobs"])
            self.assertEqual(upstream.requests[0]["top_logprobs"], 0)

    def test_observability_only_mode_does_not_request_logprobs(self) -> None:
        upstream = _FakeUpstream()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = ModelProxyService(
                identity=_identity(),
                endpoint_kind=ModelEndpointKind.CONTROLLED,
                upstream_chat_completions_url=upstream.url,
                evidence_writer=ModelEvidenceJsonlWriter(root / "evidence.jsonl"),
                recorder=TraceRecorder(
                    EventWriter(root / "events.jsonl"),
                    run_id="run-http-proxy",
                    episode_id="episode-http-proxy",
                    trace_id="trace-http-proxy",
                ),
                access_token="execution-proxy-token",
                capture_response_logprobs=False,
            )
            try:
                service.forward_chat_completions({
                    "model": "example-14b",
                    "messages": [{"role": "user", "content": "fix"}],
                })
            finally:
                upstream.close()
        self.assertNotIn("logprobs", upstream.requests[0])
        self.assertNotIn("top_logprobs", upstream.requests[0])

    def test_proxy_applies_default_seed_without_overwriting_harness_seed(self) -> None:
        upstream = _FakeUpstream()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = ModelProxyService(
                identity=_identity(),
                endpoint_kind=ModelEndpointKind.CONTROLLED,
                upstream_chat_completions_url=upstream.url,
                evidence_writer=ModelEvidenceJsonlWriter(root / "evidence.jsonl"),
                recorder=TraceRecorder(EventWriter(root / "events.jsonl"), run_id="run-http-proxy", episode_id="episode-http-proxy", trace_id="trace-http-proxy"),
                access_token="execution-proxy-token",
                default_sampling_seed=17,
            )
            try:
                service.forward_chat_completions({"model": "example-14b", "messages": [{"role": "user", "content": "a"}]})
                service.forward_chat_completions({"model": "example-14b", "messages": [{"role": "user", "content": "b"}], "seed": 18})
            finally:
                upstream.close()
        self.assertEqual([item["seed"] for item in upstream.requests], [17, 18])

    def test_replays_controlled_response_as_sse_while_persisting_evidence(self) -> None:
        upstream = _FakeUpstream()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = ModelProxyService(
                identity=_identity(),
                endpoint_kind=ModelEndpointKind.CONTROLLED,
                upstream_chat_completions_url=upstream.url,
                evidence_writer=ModelEvidenceJsonlWriter(root / "evidence.jsonl"),
                recorder=TraceRecorder(
                    EventWriter(root / "events.jsonl"),
                    run_id="run-http-proxy",
                    episode_id="episode-http-proxy",
                    trace_id="trace-http-proxy",
                ),
                access_token="execution-proxy-token",
            )
            proxy = ModelProxyHttpServer(service)
            proxy.start_in_thread()
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{proxy.address[1]}/v1/chat/completions",
                    data=json.dumps(
                        {
                            "model": "example-14b",
                            "messages": [{"role": "user", "content": "fix"}],
                            "stream": True,
                        }
                    ).encode(),
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": "Bearer execution-proxy-token",
                    },
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    body = response.read().decode()
            finally:
                proxy.close()
                upstream.close()
            self.assertIn("text/event-stream", response.headers["Content-Type"])
            self.assertIn('"content": "done"', body)
            self.assertTrue(body.endswith("data: [DONE]\n\n"))
            evidence = ModelCallEvidence.from_dict(
                json.loads((root / "evidence.jsonl").read_text())
            )
            self.assertTrue(evidence.rl_usable_call)


if __name__ == "__main__":
    unittest.main()
