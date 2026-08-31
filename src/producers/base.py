"""Trainer-neutral interface for live harness rollout producers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from src.contracts._json import freeze_json, sha256_json, thaw_json
from src.contracts._validation import required_text, strict_fields
from src.contracts.execution_identity import ExecutionIdentity
from src.errors import ContractValidationError

PRODUCER_CONTRACT_VERSION = "rollout-producer/v1"


class ProducerCapability(str, Enum):
    RAW_HARNESS_TRACE = "RAW_HARNESS_TRACE"
    HARNESS_EVENTS = "HARNESS_EVENTS"
    TOKEN_IDS = "TOKEN_IDS"
    BEHAVIOR_LOGPROBS = "BEHAVIOR_LOGPROBS"
    ACTION_MASK = "ACTION_MASK"
    POLICY_VERSION = "POLICY_VERSION"
    VERIFIER_EVIDENCE = "VERIFIER_EVIDENCE"
    ASYNC_SUBMISSION = "ASYNC_SUBMISSION"
    WEIGHT_SYNC = "WEIGHT_SYNC"
    RESAMPLING = "RESAMPLING"


class ProducerExecutionStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INFRA_INVALID = "INFRA_INVALID"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True, slots=True, kw_only=True)
class ProducerRequest:
    identity: ExecutionIdentity
    instruction: str
    workspace: str
    timeout_seconds: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = PRODUCER_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ExecutionIdentity):
            raise ContractValidationError("identity must be an ExecutionIdentity")
        required_text(self.instruction, "instruction")
        required_text(self.workspace, "workspace")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or float(self.timeout_seconds) <= 0
        ):
            raise ContractValidationError("timeout_seconds must be positive")
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        frozen = freeze_json(self.metadata)
        if not isinstance(frozen, Mapping):
            raise ContractValidationError("metadata must be an object")
        object.__setattr__(self, "metadata", frozen)
        if self.schema_version != PRODUCER_CONTRACT_VERSION:
            raise ContractValidationError(
                f"schema_version must be {PRODUCER_CONTRACT_VERSION!r}"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ProducerArtifact:
    identity: ExecutionIdentity
    status: ProducerExecutionStatus
    capabilities: frozenset[ProducerCapability]
    payload: Mapping[str, Any]
    issues: tuple[str, ...] = ()
    schema_version: str = PRODUCER_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ExecutionIdentity):
            raise ContractValidationError("identity must be an ExecutionIdentity")
        if not isinstance(self.status, ProducerExecutionStatus):
            raise ContractValidationError("status must be a ProducerExecutionStatus")
        capabilities = frozenset(self.capabilities)
        if any(not isinstance(item, ProducerCapability) for item in capabilities):
            raise ContractValidationError("capabilities contain an unknown value")
        object.__setattr__(self, "capabilities", capabilities)
        frozen = freeze_json(self.payload)
        if not isinstance(frozen, Mapping):
            raise ContractValidationError("payload must be an object")
        object.__setattr__(self, "payload", frozen)
        issues = tuple(self.issues)
        for index, issue in enumerate(issues):
            required_text(issue, f"issues[{index}]")
        object.__setattr__(self, "issues", issues)
        if self.schema_version != PRODUCER_CONTRACT_VERSION:
            raise ContractValidationError(
                f"schema_version must be {PRODUCER_CONTRACT_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "status": self.status.value,
            "capabilities": sorted(item.value for item in self.capabilities),
            "payload": thaw_json(self.payload),
            "issues": list(self.issues),
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProducerArtifact":
        payload = strict_fields(
            value,
            {"schema_version", "identity", "status", "capabilities", "payload", "issues"},
            "ProducerArtifact",
        )
        try:
            status = ProducerExecutionStatus(payload["status"])
            capabilities = frozenset(
                ProducerCapability(item) for item in payload["capabilities"]
            )
        except (TypeError, ValueError) as exc:
            raise ContractValidationError("invalid ProducerArtifact enum value") from exc
        return cls(
            schema_version=payload["schema_version"],
            identity=ExecutionIdentity.from_dict(payload["identity"]),
            status=status,
            capabilities=capabilities,
            payload=payload["payload"],
            issues=tuple(payload["issues"]),
        )


@runtime_checkable
class RolloutProducer(Protocol):
    producer_id: str
    producer_version: str
    capabilities: frozenset[ProducerCapability]

    def run(self, request: ProducerRequest) -> ProducerArtifact:
        """Run exactly one immutable task attempt."""
