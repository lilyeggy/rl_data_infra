"""Strict attestation produced by the local verifier execution layer."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.contracts._json import sha256_json, validate_sha256
from src.contracts._validation import strict_fields, utc_instant
from src.contracts.agent_episode import EpisodeVerifierStatus
from src.contracts.manifests import EvaluatorManifest
from src.errors import ContractValidationError

LOCAL_VERIFIER_REPORT_VERSION = "local-verifier-report/v1"


class VerifierExecutionStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INFRA_INVALID = "INFRA_INVALID"
    TIMEOUT = "TIMEOUT"


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalVerifierReport:
    identity_checksum: str
    evaluator: EvaluatorManifest
    command: tuple[str, ...]
    execution_status: VerifierExecutionStatus
    verifier_status: EpisodeVerifierStatus
    score: float | None
    producer_artifact_checksum: str
    output_artifact_checksums: tuple[str, ...]
    created_at: str
    schema_version: str = LOCAL_VERIFIER_REPORT_VERSION

    def __post_init__(self) -> None:
        validate_sha256(self.identity_checksum, "identity_checksum")
        if not isinstance(self.evaluator, EvaluatorManifest):
            raise ContractValidationError("evaluator must be EvaluatorManifest")
        command = tuple(self.command)
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ContractValidationError("command must contain non-empty strings")
        object.__setattr__(self, "command", command)
        if not isinstance(self.execution_status, VerifierExecutionStatus):
            raise ContractValidationError("execution_status has invalid value")
        if not isinstance(self.verifier_status, EpisodeVerifierStatus):
            raise ContractValidationError("verifier_status has invalid value")
        if self.score is not None:
            if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
                raise ContractValidationError("score must be numeric")
            if not math.isfinite(float(self.score)):
                raise ContractValidationError("score must be finite")
            object.__setattr__(self, "score", float(self.score))
        validate_sha256(self.producer_artifact_checksum, "producer_artifact_checksum")
        checksums = tuple(self.output_artifact_checksums)
        if len(checksums) != len(set(checksums)):
            raise ContractValidationError("output_artifact_checksums must be unique")
        for index, checksum in enumerate(checksums):
            validate_sha256(checksum, f"output_artifact_checksums[{index}]")
        object.__setattr__(self, "output_artifact_checksums", checksums)
        utc_instant(self.created_at, "created_at")
        if self.schema_version != LOCAL_VERIFIER_REPORT_VERSION:
            raise ContractValidationError("unsupported LocalVerifierReport schema_version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity_checksum": self.identity_checksum,
            "evaluator": self.evaluator.to_dict(),
            "command": list(self.command),
            "execution_status": self.execution_status.value,
            "verifier_status": self.verifier_status.value,
            "score": self.score,
            "producer_artifact_checksum": self.producer_artifact_checksum,
            "output_artifact_checksums": list(self.output_artifact_checksums),
            "created_at": self.created_at,
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LocalVerifierReport":
        payload = strict_fields(
            value,
            {
                "schema_version",
                "identity_checksum",
                "evaluator",
                "command",
                "execution_status",
                "verifier_status",
                "score",
                "producer_artifact_checksum",
                "output_artifact_checksums",
                "created_at",
            },
            "LocalVerifierReport",
        )
        try:
            execution_status = VerifierExecutionStatus(payload["execution_status"])
            verifier_status = EpisodeVerifierStatus(payload["verifier_status"])
        except (TypeError, ValueError) as exc:
            raise ContractValidationError("invalid verifier report enum") from exc
        return cls(
            schema_version=payload["schema_version"],
            identity_checksum=payload["identity_checksum"],
            evaluator=EvaluatorManifest.from_dict(payload["evaluator"]),
            command=tuple(payload["command"]),
            execution_status=execution_status,
            verifier_status=verifier_status,
            score=payload["score"],
            producer_artifact_checksum=payload["producer_artifact_checksum"],
            output_artifact_checksums=tuple(payload["output_artifact_checksums"]),
            created_at=payload["created_at"],
        )
