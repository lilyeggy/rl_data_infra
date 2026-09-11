"""Small OpenAI-compatible HTTP forwarder with durable model-call evidence."""

from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from src.capture.harness_http import HarnessEventIngress
from src.capture.tool_calls import extract_tool_calls
from src.capture.model_proxy import (
    ModelBackendResponse,
    ModelCallEvidence,
    ModelEndpointKind,
    ModelEvidenceJsonlWriter,
    ModelProxyRequest,
    capture_model_call,
)
from src.capture.recorder import TraceRecorder
from src.contracts.execution_identity import ExecutionIdentity
from src.contracts.trace_event import EventStatus
from src.errors import ContractValidationError

MODEL_PROXY_HTTP_VERSION = "model-proxy-http/v1"
_EVIDENCE_EXTENSION = "agent_data_plane_evidence"
logger = logging.getLogger(__name__)


def _recover_tool_calls(response: Any, tools: Any) -> int:
    """Fill in tool calls the backend did not parse, and report how many.

    The trained policy emits bare tool-call JSON, which stock vLLM parsers do
    not match (they require ``<tool_call>`` tags), so a request that declared
    tools comes back with content but no ``tool_calls``. The stage D/E servers
    always extracted them here; the framework's engine has to be given the same
    tolerance, and this proxy is where the protocol conversion already happens.
    An engine-provided parse always wins.

    Only the OpenAI-shaped view Pi consumes is affected -- the recorded token
    ids and logprobs still come from the engine's own response.
    """
    if not isinstance(response, Mapping):
        return 0
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return 0
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("tool_calls"):
        return 0
    recovered = extract_tool_calls(message.get("content") or "", tools)
    if not recovered:
        return 0
    message["tool_calls"] = recovered
    # Pi only acts on a tool-call turn, and the engine reported "length" or
    # "stop" because it saw no call.
    choice["finish_reason"] = "tool_calls"
    return len(recovered)


class ModelProxyService:
    """One execution-bound proxy; credentials are forwarded but never persisted."""

    def __init__(
        self,
        *,
        identity: ExecutionIdentity,
        endpoint_kind: ModelEndpointKind,
        upstream_chat_completions_url: str,
        evidence_writer: ModelEvidenceJsonlWriter,
        recorder: TraceRecorder,
        upstream_authorization: str | None = None,
        access_token: str,
        timeout_seconds: float = 300,
        max_calls: int = 128,
        backend_model_revision: str | None = None,
        capture_response_logprobs: bool = True,
        default_sampling_seed: int | None = None,
        on_first_model_request: Callable[[], None] | None = None,
        upstream_transport: Callable[[dict[str, Any]], tuple[int, dict[str, Any]]] | None = None,
    ) -> None:
        if not isinstance(identity, ExecutionIdentity):
            raise TypeError("identity must be ExecutionIdentity")
        if not isinstance(endpoint_kind, ModelEndpointKind):
            raise TypeError("endpoint_kind must be ModelEndpointKind")
        if not upstream_chat_completions_url.startswith(("http://", "https://")):
            raise ContractValidationError("upstream URL must use http or https")
        if not isinstance(evidence_writer, ModelEvidenceJsonlWriter):
            raise TypeError("evidence_writer must be ModelEvidenceJsonlWriter")
        if not isinstance(recorder, TraceRecorder):
            raise TypeError("recorder must be TraceRecorder")
        self.identity = identity
        self.endpoint_kind = endpoint_kind
        self.upstream_url = upstream_chat_completions_url
        self.evidence_writer = evidence_writer
        self.recorder = recorder
        self.upstream_authorization = upstream_authorization
        if not isinstance(access_token, str) or not access_token:
            raise ContractValidationError("proxy access_token must be non-empty")
        self.access_token = access_token
        self.timeout_seconds = float(timeout_seconds)
        self.max_calls = int(max_calls)
        self.backend_model_revision = backend_model_revision or os.environ.get(
            "AGENT_BACKEND_MODEL_REVISION"
        )
        if not isinstance(capture_response_logprobs, bool):
            raise TypeError("capture_response_logprobs must be a boolean")
        self.capture_response_logprobs = capture_response_logprobs
        if default_sampling_seed is not None and (
            isinstance(default_sampling_seed, bool)
            or not isinstance(default_sampling_seed, int)
            or default_sampling_seed < 0
        ):
            raise ContractValidationError("default_sampling_seed must be a non-negative integer or null")
        self.default_sampling_seed = default_sampling_seed
        if on_first_model_request is not None and not callable(on_first_model_request):
            raise TypeError("on_first_model_request must be callable or null")
        self.on_first_model_request = on_first_model_request
        self.upstream_transport = upstream_transport
        if self.max_calls < 1:
            raise ContractValidationError("max_calls must be positive")
        self._call_count = 0
        self._observed_first_model_request = False
        self._record_lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "service_version": MODEL_PROXY_HTTP_VERSION,
            "status": "ok",
            "run_id": self.identity.run_id,
            "episode_id": self.identity.episode_id,
            "endpoint_kind": self.endpoint_kind.value,
        }

    def forward_chat_completions(self, payload: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        if not isinstance(payload, Mapping):
            raise ContractValidationError("chat completion payload must be an object")
        model_id = payload.get("model")
        messages = payload.get("messages")
        tools = payload.get("tools", [])
        if not isinstance(model_id, str) or not model_id.strip():
            raise ContractValidationError("model must be a non-empty string")
        if not isinstance(messages, list) or not messages:
            raise ContractValidationError("messages must be a non-empty array")
        if not isinstance(tools, list):
            raise ContractValidationError("tools must be an array")
        notify_first_model_request = False
        with self._record_lock:
            if self._call_count >= self.max_calls:
                if self.upstream_transport is not None:
                    return 429, {"error": {"type": "episode_request_budget_exhausted"}}
                return 200, {
                    "id": f"budget-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Execution budget exhausted; stop and report the current result.",
                        },
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
            self._call_count += 1
            if not self._observed_first_model_request:
                self._observed_first_model_request = True
                notify_first_model_request = True
        if notify_first_model_request and self.on_first_model_request is not None:
            self.on_first_model_request()
        request_id = f"model-{uuid.uuid4().hex}"
        sampling_config = {
            str(key): value
            for key, value in payload.items()
            if key not in {"model", "messages", "tools"}
        }
        request_evidence = ModelProxyRequest(
            identity=self.identity,
            request_id=request_id,
            endpoint_kind=self.endpoint_kind,
            model_id=model_id,
            messages=tuple(messages),
            tools=tuple(tools),
            sampling_config=sampling_config,
        )
        span_id = f"span-{request_id}"
        with self._record_lock:
            self.recorder.model_request(
                span_id=span_id,
                parent_span_id=None,
                model=model_id,
                messages=messages,
                tools=tools,
                attempt=self.identity.attempt_id,
            )

        started = time.monotonic()
        upstream_payload = dict(payload)
        if self.default_sampling_seed is not None:
            upstream_payload.setdefault("seed", self.default_sampling_seed)
        # Controlled rollouts opt into the standard vLLM/OpenAI logprob field.
        # The vLLM launcher additionally requests token IDs in that field; the
        # adapter below converts them into our aligned evidence contract.
        if (
            self.endpoint_kind is ModelEndpointKind.CONTROLLED
            and self.capture_response_logprobs
        ):
            upstream_payload.setdefault("logprobs", True)
            upstream_payload.setdefault("top_logprobs", 0)
            # A framework-managed vLLM (verl's vLLMHttpServer) serves a stock
            # OpenAI surface with no bespoke evidence extension, so ask for
            # native ids explicitly: `return_token_ids` yields choices[].token_ids
            # and prompt_token_ids, `return_tokens_as_token_ids` renders each
            # logprob entry's token as "token_id:<n>". Both are ignored by
            # servers that do not implement them.
            upstream_payload.setdefault("return_token_ids", True)
            upstream_payload.setdefault("return_tokens_as_token_ids", True)
        # Pi uses SSE.  We intentionally execute a non-streaming upstream call
        # so the controlled backend can return complete token/logprob evidence,
        # then replay its finalized OpenAI response as standards-compatible SSE.
        if payload.get("stream") is True:
            upstream_payload["stream"] = False
            # stream_options is only valid for a streaming request on some
            # OpenAI-compatible gateways.  The proxy deliberately makes a
            # non-streaming upstream call so it can persist one complete
            # response before replaying SSE to Pi.
            upstream_payload.pop("stream_options", None)
        status_code, response_payload = self._send_upstream(upstream_payload)
        latency_ms = (time.monotonic() - started) * 1000
        if 200 <= status_code < 300 and self.upstream_transport is None:
            recovered = _recover_tool_calls(response_payload, request_evidence.tools)
            if recovered:
                logger.info(
                    "%s: recovered %d tool call(s) from generated text; the "
                    "backend parsed none",
                    request_id,
                    recovered,
                )
        backend, extension_issue = _backend_response(
            response_payload,
            status_code=status_code,
            latency_ms=latency_ms,
            backend_model_revision=self.backend_model_revision,
        )
        evidence = capture_model_call(request_evidence, backend)
        if extension_issue is not None:
            evidence = ModelCallEvidence(
                request=evidence.request,
                backend=evidence.backend,
                capabilities=evidence.capabilities,
                issues=evidence.issues + (extension_issue,),
            )
        event_status = EventStatus.SUCCEEDED if 200 <= status_code < 300 else EventStatus.FAILED
        with self._record_lock:
            self.evidence_writer.append(evidence)
            self.recorder.model_response(
                span_id=span_id,
                parent_span_id=None,
                response=response_payload,
                usage=(
                    response_payload.get("usage")
                    if isinstance(response_payload.get("usage"), Mapping)
                    else None
                ),
                latency_ms=latency_ms,
                token_ids=backend.response_token_ids,
                logprobs=backend.response_logprobs,
                action_mask=(
                    tuple(1 for _ in backend.response_token_ids)
                    if backend.response_token_ids is not None
                    else None
                ),
                backend_model_revision=backend.backend_model_revision,
                status=event_status,
                attempt=self.identity.attempt_id,
            )
        return status_code, response_payload

    @staticmethod
    def sse_body(response: Mapping[str, Any]) -> bytes:
        """Encode one finalized Chat Completions response as a minimal SSE stream."""

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise ContractValidationError("upstream streaming response has no choices")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise ContractValidationError("upstream streaming response has no message")
        delta: dict[str, Any] = {"role": "assistant"}
        normalized_tool_calls: list[dict[str, Any]] = []
        if message.get("tool_calls") is not None:
            tool_calls = message["tool_calls"]
            if not isinstance(tool_calls, list):
                raise ContractValidationError("tool_calls must be an array")
            normalized_tool_calls = []
            for index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, Mapping):
                    continue
                normalized = dict(tool_call)
                # Some OpenAI-compatible providers include an index in the
                # finalized tool call; SSE replay owns that field.
                normalized.pop("index", None)
                normalized_tool_calls.append({"index": index, **normalized})
            delta["tool_calls"] = normalized_tool_calls
        else:
            delta["content"] = message.get("content") or ""
        common = {
            key: response[key]
            for key in ("id", "object", "created", "model")
            if key in response
        }
        first = common | {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        frames = [f"data: {json.dumps(first, ensure_ascii=False)}\n\n"]
        # Pi is stricter than some OpenAI-compatible clients and expects each
        # parallel tool call to arrive as its own streaming delta.  Providers
        # commonly return all finalized calls in one non-streaming response,
        # so split them here while preserving their stable call indexes.
        if normalized_tool_calls:
            frames = [
                f"data: {json.dumps(common | {'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
            ]
            for tool_call in normalized_tool_calls:
                function = tool_call.get("function")
                if not isinstance(function, Mapping):
                    raise ContractValidationError("tool call function must be an object")
                declaration = {
                    "index": tool_call["index"],
                    "id": tool_call.get("id"),
                    "type": tool_call.get("type", "function"),
                    "function": {"name": function.get("name", "")},
                }
                frames.append(f"data: {json.dumps(common | {'choices': [{'index': 0, 'delta': {'tool_calls': [declaration]}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n")
                arguments = function.get("arguments", "")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
                argument_delta = {
                    "index": tool_call["index"],
                    "function": {"arguments": arguments},
                }
                frames.append(f"data: {json.dumps(common | {'choices': [{'index': 0, 'delta': {'tool_calls': [argument_delta]}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n")
        final = common | {
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": choice.get("finish_reason", "stop"),
                }
            ],
            "usage": response.get("usage", {}),
        }
        frames.append(f"data: {json.dumps(final, ensure_ascii=False)}\n\n")
        frames.append("data: [DONE]\n\n")
        return "".join(frames).encode()

    def _send_upstream(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if self.upstream_transport is not None:
            return self.upstream_transport(payload)
        headers = {
            "Content-Type": "application/json",
            # OpenCode's edge rejects urllib's default Python user agent with
            # an HTML 403 page, which cannot be represented as upstream JSON.
            "User-Agent": "agent-data-plane/1.0",
        }
        if self.upstream_authorization is not None:
            headers["Authorization"] = self.upstream_authorization
        request = urllib.request.Request(
            self.upstream_url,
            data=json.dumps(payload, ensure_ascii=False).encode(),
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                status = response.status
                body = response.read(50 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as exc:
            status = exc.code
            body = exc.read(50 * 1024 * 1024 + 1)
        except (urllib.error.URLError, TimeoutError) as exc:
            return 502, {"error": {"type": "upstream_unavailable", "message": str(exc)}}
        if len(body) > 50 * 1024 * 1024:
            return 502, {"error": {"type": "upstream_response_too_large"}}
        try:
            value = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return 502, {"error": {"type": "invalid_upstream_json"}}
        if not isinstance(value, dict):
            return 502, {"error": {"type": "invalid_upstream_shape"}}
        return status, value


def _backend_response(
    response: Mapping[str, Any], *, status_code: int, latency_ms: float,
    backend_model_revision: str | None = None,
) -> tuple[ModelBackendResponse, str | None]:
    extension = response.get(_EVIDENCE_EXTENSION)
    if extension is None:
        extension = {}
    if not isinstance(extension, Mapping):
        extension = {}
        issue = "backend training-evidence extension is not an object"
    else:
        issue = None
    try:
        backend = ModelBackendResponse(
            response=response,
            latency_ms=latency_ms,
            status_code=status_code,
            backend_model_revision=extension.get("backend_model_revision") or backend_model_revision,
            prompt_token_ids=_optional_tuple(extension.get("prompt_token_ids")) or _standard_prompt_token_ids(response),
            response_token_ids=_optional_tuple(extension.get("response_token_ids")) or _standard_response_token_ids(response),
            response_logprobs=_optional_tuple(extension.get("response_logprobs")) or _standard_response_logprobs(response),
            usage=response.get("usage", {}) if isinstance(response.get("usage"), Mapping) else {},
        )
    except (ContractValidationError, TypeError, ValueError) as exc:
        issue = f"invalid backend training-evidence extension: {exc}"
        backend = ModelBackendResponse(
            response=response,
            latency_ms=latency_ms,
            status_code=status_code,
            backend_model_revision=backend_model_revision,
            usage=response.get("usage", {}) if isinstance(response.get("usage"), Mapping) else {},
        )
    return backend, issue


def _optional_tuple(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ContractValidationError("training evidence arrays must be arrays")
    return tuple(value)


def _standard_logprob_items(response: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return []
    logprobs = choices[0].get("logprobs")
    if not isinstance(logprobs, Mapping) or not isinstance(logprobs.get("content"), list):
        return []
    return [item for item in logprobs["content"] if isinstance(item, Mapping)]


def _standard_response_token_ids(response: Mapping[str, Any]) -> tuple[int, ...] | None:
    # Preferred: `return_token_ids` gives the native ids directly.
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        raw_ids = choices[0].get("token_ids")
        if isinstance(raw_ids, list) and raw_ids:
            if all(isinstance(value, int) and not isinstance(value, bool) for value in raw_ids):
                return tuple(raw_ids)
            return None
    # Fallback: ids recovered from the "token_id:<n>" logprob key form.
    values = []
    for item in _standard_logprob_items(response):
        token = item.get("token")
        if not isinstance(token, str) or not token.startswith("token_id:"):
            return None
        try:
            values.append(int(token.removeprefix("token_id:")))
        except ValueError:
            return None
    return tuple(values) if values else None


def _standard_response_logprobs(response: Mapping[str, Any]) -> tuple[float, ...] | None:
    items = _standard_logprob_items(response)
    if not items or any(not isinstance(item.get("logprob"), (int, float)) for item in items):
        return None
    return tuple(float(item["logprob"]) for item in items)


def _standard_prompt_token_ids(response: Mapping[str, Any]) -> tuple[int, ...] | None:
    # Preferred: `return_token_ids` gives the prompt ids directly.
    raw_ids = response.get("prompt_token_ids")
    if isinstance(raw_ids, list) and raw_ids:
        if all(isinstance(value, int) and not isinstance(value, bool) for value in raw_ids):
            return tuple(raw_ids)
        return None
    raw = response.get("prompt_logprobs")
    if not isinstance(raw, list):
        return None
    values = []
    for position in raw:
        if not isinstance(position, Mapping) or not position:
            continue
        try:
            values.append(int(next(iter(position))))
        except (TypeError, ValueError):
            return None
    return tuple(values) if values else None


class ModelProxyHttpServer:
    def __init__(
        self,
        service: ModelProxyService,
        *,
        harness_ingress: HarnessEventIngress | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        service_ref = service
        ingress_ref = harness_ingress

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def _send(self, status: int, payload: Mapping[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_sse(self, status: int, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/healthz":
                    self._send(200, service_ref.health())
                else:
                    self._send(404, {"error": "not_found"})

            def do_POST(self) -> None:
                if self.path not in ("/v1/chat/completions", "/v1/harness/events"):
                    self._send(404, {"error": "not_found"})
                    return
                supplied = self.headers.get("Authorization", "")
                expected = f"Bearer {service_ref.access_token}"
                if not hmac.compare_digest(supplied, expected):
                    self._send(401, {"error": "unauthorized"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._send(400, {"error": "invalid_content_length"})
                    return
                if length < 1 or length > 10 * 1024 * 1024:
                    self._send(413, {"error": "request_too_large_or_empty"})
                    return
                try:
                    payload = json.loads(self.rfile.read(length))
                    if self.path == "/v1/harness/events":
                        if ingress_ref is None:
                            self._send(404, {"error": "not_found"})
                            return
                        event = ingress_ref.emit(payload)
                        self._send(
                            202,
                            {"event_id": event.event_id, "checksum": event.checksum},
                        )
                        return
                    status, response = service_ref.forward_chat_completions(payload)
                except (json.JSONDecodeError, UnicodeDecodeError, ContractValidationError) as exc:
                    self._send(400, {"error": "invalid_request", "message": str(exc)})
                    return
                if payload.get("stream") is True and 200 <= status < 300:
                    self._send_sse(status, service_ref.sse_body(response))
                else:
                    self._send(status, response)

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.httpd.server_address[:2]
        return str(host), int(port)

    def start_in_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
