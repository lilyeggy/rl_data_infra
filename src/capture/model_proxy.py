"""Provider-neutral Model Proxy evidence without fabricated training fields."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.capture.recorder import redact_secrets
from src.contracts._json import canonical_json_bytes, freeze_json, sha256_json, thaw_json
from src.contracts._validation import optional_text, required_text, strict_fields
from src.contracts.execution_identity import ExecutionIdentity
from src.errors import ContractValidationError

MODEL_PROXY_EVIDENCE_VERSION = "model-proxy-evidence/v1"


class ModelEndpointKind(str, Enum):
    CONTROLLED = "CONTROLLED"
    EXTERNAL_API = "EXTERNAL_API"


class ModelEvidenceCapability(str, Enum):
    REQUEST_RESPONSE = "REQUEST_RESPONSE"
    TOKEN_IDS = "TOKEN_IDS"
    BEHAVIOR_LOGPROBS = "BEHAVIOR_LOGPROBS"
    POLICY_VERSION = "POLICY_VERSION"


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelProxyRequest:
    identity: ExecutionIdentity
    request_id: str
    endpoint_kind: ModelEndpointKind
    model_id: str
    messages: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...] = ()
    sampling_config: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = MODEL_PROXY_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ExecutionIdentity):
            raise ContractValidationError("identity must be an ExecutionIdentity")
        required_text(self.request_id, "request_id")
        required_text(self.model_id, "model_id")
        if not isinstance(self.endpoint_kind, ModelEndpointKind):
            raise ContractValidationError("endpoint_kind must be ModelEndpointKind")
        for name in ("messages", "tools"):
            values = tuple(getattr(self, name))
            if any(not isinstance(item, Mapping) for item in values):
                raise ContractValidationError(f"{name} must contain objects")
            object.__setattr__(
                self,
                name,
                tuple(freeze_json(redact_secrets(item)) for item in values),
            )
        sampling = freeze_json(self.sampling_config)
        if not isinstance(sampling, Mapping):
            raise ContractValidationError("sampling_config must be an object")
        object.__setattr__(self, "sampling_config", sampling)
        if self.schema_version != MODEL_PROXY_EVIDENCE_VERSION:
            raise ContractValidationError("unsupported ModelProxyRequest schema_version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "request_id": self.request_id,
            "endpoint_kind": self.endpoint_kind.value,
            "model_id": self.model_id,
            "messages": [thaw_json(item) for item in self.messages],
            "tools": [thaw_json(item) for item in self.tools],
            "sampling_config": thaw_json(self.sampling_config),
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelProxyRequest":
        payload = strict_fields(
            value,
            {
                "schema_version",
                "identity",
                "request_id",
                "endpoint_kind",
                "model_id",
                "messages",
                "tools",
                "sampling_config",
            },
            "ModelProxyRequest",
        )
        try:
            endpoint_kind = ModelEndpointKind(payload["endpoint_kind"])
        except (TypeError, ValueError) as exc:
            raise ContractValidationError("invalid model endpoint kind") from exc
        return cls(
            schema_version=payload["schema_version"],
            identity=ExecutionIdentity.from_dict(payload["identity"]),
            request_id=payload["request_id"],
            endpoint_kind=endpoint_kind,
            model_id=payload["model_id"],
            messages=tuple(payload["messages"]),
            tools=tuple(payload["tools"]),
            sampling_config=payload["sampling_config"],
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelBackendResponse:
    response: Mapping[str, Any]
    latency_ms: float
    status_code: int
    backend_model_revision: str | None = None
    prompt_token_ids: tuple[int, ...] | None = None
    response_token_ids: tuple[int, ...] | None = None
    response_logprobs: tuple[float, ...] | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        response = freeze_json(redact_secrets(self.response))
        usage = freeze_json(self.usage)
        if not isinstance(response, Mapping) or not isinstance(usage, Mapping):
            raise ContractValidationError("response and usage must be objects")
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "usage", usage)
        if (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, (int, float))
            or not math.isfinite(float(self.latency_ms))
            or float(self.latency_ms) < 0
        ):
            raise ContractValidationError("latency_ms must be finite and non-negative")
        object.__setattr__(self, "latency_ms", float(self.latency_ms))
        if (
            isinstance(self.status_code, bool)
            or not isinstance(self.status_code, int)
            or not 100 <= self.status_code <= 599
        ):
            raise ContractValidationError("status_code must be an HTTP status code")
        optional_text(self.backend_model_revision, "backend_model_revision")
        for name in ("prompt_token_ids", "response_token_ids"):
            value = getattr(self, name)
            if value is None:
                continue
            values = tuple(value)
            if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
                raise ContractValidationError(f"{name} must contain integers")
            object.__setattr__(self, name, values)
        if self.response_logprobs is not None:
            values = tuple(float(item) for item in self.response_logprobs)
            if any(not math.isfinite(item) for item in values):
                raise ContractValidationError("response_logprobs must be finite")
            object.__setattr__(self, "response_logprobs", values)
            if self.response_token_ids is None or len(values) != len(
                self.response_token_ids
            ):
                raise ContractValidationError(
                    "response_logprobs require aligned response_token_ids"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "response": thaw_json(self.response),
            "latency_ms": self.latency_ms,
            "status_code": self.status_code,
            "backend_model_revision": self.backend_model_revision,
            "prompt_token_ids": (
                list(self.prompt_token_ids) if self.prompt_token_ids is not None else None
            ),
            "response_token_ids": (
                list(self.response_token_ids)
                if self.response_token_ids is not None
                else None
            ),
            "response_logprobs": (
                list(self.response_logprobs)
                if self.response_logprobs is not None
                else None
            ),
            "usage": thaw_json(self.usage),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelBackendResponse":
        payload = strict_fields(
            value,
            {
                "response",
                "latency_ms",
                "status_code",
                "backend_model_revision",
                "prompt_token_ids",
                "response_token_ids",
                "response_logprobs",
                "usage",
            },
            "ModelBackendResponse",
        )
        return cls(
            response=payload["response"],
            latency_ms=payload["latency_ms"],
            status_code=payload["status_code"],
            backend_model_revision=payload["backend_model_revision"],
            prompt_token_ids=(
                tuple(payload["prompt_token_ids"])
                if payload["prompt_token_ids"] is not None
                else None
            ),
            response_token_ids=(
                tuple(payload["response_token_ids"])
                if payload["response_token_ids"] is not None
                else None
            ),
            response_logprobs=(
                tuple(payload["response_logprobs"])
                if payload["response_logprobs"] is not None
                else None
            ),
            usage=payload["usage"],
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelCallEvidence:
    request: ModelProxyRequest
    backend: ModelBackendResponse
    capabilities: frozenset[ModelEvidenceCapability]
    issues: tuple[str, ...]
    schema_version: str = MODEL_PROXY_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.request, ModelProxyRequest):
            raise ContractValidationError("request must be ModelProxyRequest")
        if not isinstance(self.backend, ModelBackendResponse):
            raise ContractValidationError("backend must be ModelBackendResponse")
        capabilities = frozenset(self.capabilities)
        if any(not isinstance(item, ModelEvidenceCapability) for item in capabilities):
            raise ContractValidationError("unknown model evidence capability")
        object.__setattr__(self, "capabilities", capabilities)
        issues = tuple(self.issues)
        for index, issue in enumerate(issues):
            required_text(issue, f"issues[{index}]")
        object.__setattr__(self, "issues", issues)

    @property
    def rl_usable_call(self) -> bool:
        return {
            ModelEvidenceCapability.TOKEN_IDS,
            ModelEvidenceCapability.BEHAVIOR_LOGPROBS,
            ModelEvidenceCapability.POLICY_VERSION,
        }.issubset(self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request": self.request.to_dict(),
            "backend": self.backend.to_dict(),
            "capabilities": sorted(item.value for item in self.capabilities),
            "issues": list(self.issues),
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelCallEvidence":
        payload = strict_fields(
            value,
            {"schema_version", "request", "backend", "capabilities", "issues"},
            "ModelCallEvidence",
        )
        try:
            capabilities = frozenset(
                ModelEvidenceCapability(item) for item in payload["capabilities"]
            )
        except (TypeError, ValueError) as exc:
            raise ContractValidationError("invalid model evidence capability") from exc
        return cls(
            schema_version=payload["schema_version"],
            request=ModelProxyRequest.from_dict(payload["request"]),
            backend=ModelBackendResponse.from_dict(payload["backend"]),
            capabilities=capabilities,
            issues=tuple(payload["issues"]),
        )


def capture_model_call(
    request: ModelProxyRequest,
    backend: ModelBackendResponse,
) -> ModelCallEvidence:
    """Classify actual backend evidence; never tokenize text in the proxy."""

    if not isinstance(request, ModelProxyRequest) or not isinstance(
        backend, ModelBackendResponse
    ):
        raise TypeError("capture_model_call requires request and backend contracts")
    capabilities = {ModelEvidenceCapability.REQUEST_RESPONSE}
    issues: list[str] = []
    if backend.response_token_ids is not None:
        capabilities.add(ModelEvidenceCapability.TOKEN_IDS)
    else:
        issues.append("backend returned no response token ids")
    if backend.response_logprobs is not None:
        capabilities.add(ModelEvidenceCapability.BEHAVIOR_LOGPROBS)
    else:
        issues.append("backend returned no aligned behavior logprobs")
    if (
        request.endpoint_kind is ModelEndpointKind.CONTROLLED
        and request.identity.policy_fingerprint is not None
        and backend.backend_model_revision is not None
    ):
        capabilities.add(ModelEvidenceCapability.POLICY_VERSION)
    else:
        issues.append("model call lacks controlled immutable policy identity")
    return ModelCallEvidence(
        request=request,
        backend=backend,
        capabilities=frozenset(capabilities),
        issues=tuple(issues),
    )


class ModelEvidenceJsonlWriter:
    """Durable, idempotent model-call evidence log keyed by request_id."""

    def __init__(self, path: str | Path, *, durable: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.durable = durable
        self._index: dict[str, str] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        evidence = ModelCallEvidence.from_dict(json.loads(line))
                    except Exception as exc:
                        raise ContractValidationError(
                            f"malformed model evidence at line {line_number}: {exc}"
                        ) from exc
                    request_id = evidence.request.request_id
                    previous = self._index.setdefault(request_id, evidence.checksum)
                    if previous != evidence.checksum:
                        raise ContractValidationError(
                            f"conflicting model evidence for request_id {request_id!r}"
                        )

    def append(self, evidence: ModelCallEvidence) -> bool:
        if not isinstance(evidence, ModelCallEvidence):
            raise TypeError("append requires ModelCallEvidence")
        request_id = evidence.request.request_id
        previous = self._index.get(request_id)
        if previous is not None:
            if previous != evidence.checksum:
                raise ContractValidationError(
                    f"request_id {request_id!r} replayed with different evidence"
                )
            return False
        with self.path.open("ab") as handle:
            handle.write(canonical_json_bytes(evidence.to_dict()) + b"\n")
            handle.flush()
            if self.durable:
                os.fsync(handle.fileno())
        self._index[request_id] = evidence.checksum
        return True
