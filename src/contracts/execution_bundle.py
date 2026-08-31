"""Immutable linkage across all artifacts derived from one execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.contracts._json import sha256_json, validate_sha256
from src.contracts._validation import strict_fields
from src.contracts.execution_identity import ExecutionIdentity
from src.errors import ContractValidationError

EXECUTION_BUNDLE_VERSION = "execution-bundle/v2"


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionBundle:
    """Content-addressed join record; it contains references, never mutable data."""

    identity: ExecutionIdentity
    episode_checksum: str
    source_artifact_checksums: tuple[str, ...]
    policy_trace_checksums: tuple[str, ...] = ()
    verifier_report_checksum: str | None = None
    schema_version: str = EXECUTION_BUNDLE_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ExecutionIdentity):
            raise ContractValidationError("identity must be an ExecutionIdentity")
        validate_sha256(self.episode_checksum, "episode_checksum")
        for field_name in ("source_artifact_checksums", "policy_trace_checksums"):
            values = tuple(getattr(self, field_name))
            if len(values) != len(set(values)):
                raise ContractValidationError(f"{field_name} must be unique")
            for index, value in enumerate(values):
                validate_sha256(value, f"{field_name}[{index}]")
            object.__setattr__(self, field_name, values)
        if not self.source_artifact_checksums:
            raise ContractValidationError("at least one source artifact checksum is required")
        validate_sha256(self.verifier_report_checksum, "verifier_report_checksum")
        if self.schema_version != EXECUTION_BUNDLE_VERSION:
            raise ContractValidationError(
                f"schema_version must be {EXECUTION_BUNDLE_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "episode_checksum": self.episode_checksum,
            "source_artifact_checksums": list(self.source_artifact_checksums),
            "policy_trace_checksums": list(self.policy_trace_checksums),
            "verifier_report_checksum": self.verifier_report_checksum,
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionBundle":
        payload = strict_fields(
            value,
            {
                "schema_version",
                "identity",
                "episode_checksum",
                "source_artifact_checksums",
                "policy_trace_checksums",
                "verifier_report_checksum",
            },
            "ExecutionBundle",
        )
        return cls(
            schema_version=payload["schema_version"],
            identity=ExecutionIdentity.from_dict(payload["identity"]),
            episode_checksum=payload["episode_checksum"],
            source_artifact_checksums=tuple(payload["source_artifact_checksums"]),
            policy_trace_checksums=tuple(payload["policy_trace_checksums"]),
            verifier_report_checksum=payload["verifier_report_checksum"],
        )
