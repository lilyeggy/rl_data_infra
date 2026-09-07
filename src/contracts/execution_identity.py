"""Stable identity shared by execution, observability, and learning views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.contracts._json import sha256_json, validate_sha256
from src.contracts._validation import (
    optional_text,
    required_text,
    strict_fields,
)
from src.errors import ContractValidationError

EXECUTION_IDENTITY_VERSION = "execution-identity/v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionIdentity:
    """Identity for exactly one task attempt produced by one rollout backend.

    The same value must be attached to the execution trace, policy trace,
    verifier report, certification decision, and every derived dataset member.
    Policy/sampling fingerprints are checksums rather than mutable model names.
    """

    run_id: str
    task_id: str
    episode_id: str
    attempt_id: int
    producer_id: str
    producer_version: str
    group_id: str | None = None
    policy_fingerprint: str | None = None
    sampling_fingerprint: str | None = None
    schema_version: str = EXECUTION_IDENTITY_VERSION

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "task_id",
            "episode_id",
            "producer_id",
            "producer_version",
        ):
            required_text(getattr(self, name), name)
        for name in ("group_id", "policy_fingerprint", "sampling_fingerprint"):
            optional_text(getattr(self, name), name)
        validate_sha256(self.policy_fingerprint, "policy_fingerprint")
        validate_sha256(self.sampling_fingerprint, "sampling_fingerprint")
        if (
            isinstance(self.attempt_id, bool)
            or not isinstance(self.attempt_id, int)
            or self.attempt_id < 1
        ):
            raise ContractValidationError("attempt_id must be a positive integer")
        if self.schema_version != EXECUTION_IDENTITY_VERSION:
            raise ContractValidationError(
                f"schema_version must be {EXECUTION_IDENTITY_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "attempt_id": self.attempt_id,
            "producer_id": self.producer_id,
            "producer_version": self.producer_version,
            "group_id": self.group_id,
            "policy_fingerprint": self.policy_fingerprint,
            "sampling_fingerprint": self.sampling_fingerprint,
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionIdentity":
        payload = strict_fields(
            value,
            {
                "schema_version",
                "run_id",
                "task_id",
                "episode_id",
                "attempt_id",
                "producer_id",
                "producer_version",
                "group_id",
                "policy_fingerprint",
                "sampling_fingerprint",
            },
            "ExecutionIdentity",
        )
        return cls(**payload)
