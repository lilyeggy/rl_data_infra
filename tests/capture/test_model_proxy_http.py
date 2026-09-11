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


class BudgetCloseoutTest(unittest.TestCase):
    """A turned-down turn must not become a failed model call.

    The orchestrator requires *every* recorded call to carry token ids, logprobs
    and a status below 400, so one failure invalidates the whole episode. Pi
    still has to be answered, and a budget-exhausted turn has to be auditable, so
    the closeout reply is recorded in its own note instead of as evidence.
    """

    class _Transport:
        """Stands in for the native token transport, which owns episode context."""

        def __init__(self, affordable: bool = False) -> None:
            self.affordable = affordable
            self.asked: list[dict] = []
            self.generated = 0

        def can_serve(self, payload) -> str | None:
            self.asked.append(dict(payload))
            return None if self.affordable else "context_budget"

        def __call__(self, payload):
            self.generated += 1
            return 200, {
                "id": "completion-1",
                "object": "chat.completion",
                "model": payload["model"],
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "done"},
                }],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                "agent_data_plane_evidence": {
                    "backend_model_revision": "checkpoint-0001",
                    "prompt_token_ids": [10, 11],
                    "response_token_ids": [20],
                    "response_logprobs": [-0.1],
                },
            }

    def _service(self, root: Path, transport, **overrides) -> ModelProxyService:
        settings = dict(
            identity=_identity(),
            endpoint_kind=ModelEndpointKind.CONTROLLED,
            upstream_chat_completions_url="http://127.0.0.1:1/v1/chat/completions",
            evidence_writer=ModelEvidenceJsonlWriter(root / "model-evidence.jsonl"),
            recorder=TraceRecorder(
                EventWriter(root / "events.jsonl"),
                run_id="run-http-proxy",
                episode_id="episode-http-proxy",
                trace_id="trace-http-proxy",
            ),
            access_token="execution-proxy-token",
            upstream_transport=transport,
            closeout_note_path=root / "engine-closeout.jsonl",
        )
        settings.update(overrides)
        return ModelProxyService(**settings)

    def test_an_unaffordable_turn_becomes_a_closeout_with_no_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transport = self._Transport()
            service = self._service(root, transport)
            status, payload = service.forward_chat_completions({
                "model": "example-14b",
                "messages": [{"role": "user", "content": "fix"}],
            })
            self.assertEqual(status, 200)
            choice = payload["choices"][0]
            self.assertEqual(choice["finish_reason"], "stop")
            # No model output is claimed: the reply exists only to let Pi stop.
            self.assertIsNone(choice["message"].get("tool_calls"))
            self.assertTrue(choice["message"]["content"])
            self.assertEqual(len(transport.asked), 1)
            self.assertEqual(
                transport.generated, 0, "an unaffordable turn reached the engine"
            )
            self.assertFalse(
                (root / "model-evidence.jsonl").exists(),
                "a closeout is not a model call and must leave no evidence",
            )
            note = json.loads(
                (root / "engine-closeout.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(note["reason"], "context_budget")
            self.assertEqual(note["episode_id"], "episode-http-proxy")

    def test_the_call_budget_closes_out_without_asking_the_transport(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transport = self._Transport(affordable=True)
            service = self._service(root, transport, max_calls=1)
            request = {
                "model": "example-14b",
                "messages": [{"role": "user", "content": "fix"}],
            }
            self.assertEqual(service.forward_chat_completions(request)[0], 200)
            self.assertEqual(transport.generated, 1)
            status, payload = service.forward_chat_completions(request)
            self.assertEqual(status, 200)
            self.assertEqual(payload["choices"][0]["finish_reason"], "stop")
            # The over-budget request is answered here, so the episode keeps its
            # one real call and no failure is recorded.
            self.assertEqual(transport.generated, 1)
            self.assertEqual(len(transport.asked), 1)
            note = json.loads(
                (root / "engine-closeout.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(note["reason"], "model_request_budget")
            self.assertEqual(note["request"], 2)


if __name__ == "__main__":
    unittest.main()
