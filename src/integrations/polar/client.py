"""Dependency-free client for Polar's documented rollout HTTP boundary.

The official stable API documents POST /rollout/task/submit and
GET /rollout/task/{task_id}. This client deliberately returns the full raw JSON
and does not guess Polar SessionResult/Trajectory fields; schema conversion is
a separate, version-pinned adapter.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from src.contracts._json import freeze_json, thaw_json
from src.contracts._validation import required_text
from src.errors import ContractValidationError

POLAR_CLIENT_VERSION = "polar-http-client/v1"


class PolarClientError(RuntimeError):
    """A transport or response-contract failure at the Polar HTTP boundary."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PolarTaskRequest:
    task_id: str
    instruction: str
    num_samples: int
    timeout_seconds: float
    agent: Mapping[str, Any]
    builder: Mapping[str, Any]
    evaluator: Mapping[str, Any] | None = None
    runtime: Mapping[str, Any] | None = None
    callback_url: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required_text(self.task_id, "task_id")
        required_text(self.instruction, "instruction")
        if (
            isinstance(self.num_samples, bool)
            or not isinstance(self.num_samples, int)
            or self.num_samples < 1
        ):
            raise ContractValidationError("num_samples must be a positive integer")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or float(self.timeout_seconds) <= 0
        ):
            raise ContractValidationError("timeout_seconds must be positive")
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        for name in ("agent", "builder", "metadata"):
            value = freeze_json(getattr(self, name))
            if not isinstance(value, Mapping):
                raise ContractValidationError(f"{name} must be an object")
            object.__setattr__(self, name, value)
        if self.evaluator is not None:
            evaluator = freeze_json(self.evaluator)
            if not isinstance(evaluator, Mapping):
                raise ContractValidationError("evaluator must be an object or null")
            object.__setattr__(self, "evaluator", evaluator)
        if self.runtime is not None:
            runtime = freeze_json(self.runtime)
            if not isinstance(runtime, Mapping):
                raise ContractValidationError("runtime must be an object")
            object.__setattr__(self, "runtime", runtime)
        if self.callback_url is not None:
            required_text(self.callback_url, "callback_url")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "num_samples": self.num_samples,
            "timeout_seconds": self.timeout_seconds,
            "agent": thaw_json(self.agent),
            "builder": thaw_json(self.builder),
            "metadata": thaw_json(self.metadata),
        }
        if self.evaluator is not None:
            payload["evaluator"] = thaw_json(self.evaluator)
        if self.runtime is not None:
            payload["runtime"] = thaw_json(self.runtime)
        if self.callback_url is not None:
            payload["callback_url"] = self.callback_url
        return payload


Transport = Callable[[str, str, Mapping[str, Any] | None, float], Mapping[str, Any]]


class PolarHttpClient:
    """Submit and inspect Polar tasks without importing Polar itself."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 30,
        headers: Mapping[str, str] | None = None,
        transport: Transport | None = None,
    ) -> None:
        required_text(base_url, "base_url")
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ContractValidationError("base_url must be an absolute HTTP(S) URL")
        if timeout_seconds <= 0:
            raise ContractValidationError("timeout_seconds must be positive")
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = float(timeout_seconds)
        self._headers = dict(headers or {})
        self._transport = transport or self._urllib_transport

    def submit_task(self, task: PolarTaskRequest) -> Mapping[str, Any]:
        if not isinstance(task, PolarTaskRequest):
            raise TypeError("submit_task requires a PolarTaskRequest")
        result = self._transport(
            "POST",
            f"{self._base_url}/rollout/task/submit",
            task.to_dict(),
            self._timeout_seconds,
        )
        if result.get("task_id") != task.task_id:
            raise PolarClientError("Polar submit response task_id mismatch")
        if not isinstance(result.get("status"), str):
            raise PolarClientError("Polar submit response has no string status")
        return result

    def get_task(self, task_id: str) -> Mapping[str, Any]:
        required_text(task_id, "task_id")
        result = self._transport(
            "GET",
            f"{self._base_url}/rollout/task/{quote(task_id, safe='')}",
            None,
            self._timeout_seconds,
        )
        response_id = result.get("task_id")
        if response_id is not None and response_id != task_id:
            raise PolarClientError("Polar task response task_id mismatch")
        return result

    def health(self) -> Mapping[str, Any]:
        return self._transport(
            "GET", f"{self._base_url}/health", None, self._timeout_seconds
        )

    def _urllib_transport(
        self,
        method: str,
        url: str,
        payload: Mapping[str, Any] | None,
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        data = None
        headers = {"Accept": "application/json", **self._headers}
        if payload is not None:
            data = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=data, method=method, headers=headers)
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                body = response.read()
        except HTTPError as exc:
            raise PolarClientError(f"Polar HTTP {exc.code} for {method} {url}") from exc
        except URLError as exc:
            raise PolarClientError(f"Polar transport failed for {method} {url}: {exc}") from exc
        try:
            value = json.loads(body)
        except (TypeError, ValueError) as exc:
            raise PolarClientError("Polar response is not valid JSON") from exc
        if not isinstance(value, Mapping):
            raise PolarClientError("Polar response must be a JSON object")
        return value
