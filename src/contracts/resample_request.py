"""Producer-agnostic request for missing members of an on-policy group."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from src.contracts._json import freeze_json, sha256_json, thaw_json
from src.errors import ContractValidationError


RESAMPLE_SCHEMA_VERSION = "resample-request/v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class ResampleRequest:
    task_id: str
    group_id: str
    policy_version: str
    required_count: int
    reason: str
    sampling_constraints: Mapping[str, Any] = field(default_factory=dict)
    source_hint: str | None = None
    schema_version: str = RESAMPLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESAMPLE_SCHEMA_VERSION:
            raise ContractValidationError(f"schema_version must be {RESAMPLE_SCHEMA_VERSION!r}")
        for name in ("task_id", "group_id", "policy_version", "reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ContractValidationError(f"{name} must be a non-empty string")
        if isinstance(self.required_count, bool) or not isinstance(self.required_count, int):
            raise ContractValidationError("required_count must be an integer")
        if self.required_count <= 0:
            raise ContractValidationError("required_count must be positive")
        if self.source_hint is not None and not self.source_hint.strip():
            raise ContractValidationError("source_hint must be non-empty when provided")
        constraints = freeze_json(self.sampling_constraints)
        if not isinstance(constraints, Mapping):
            raise ContractValidationError("sampling_constraints must be an object")
        object.__setattr__(self, "sampling_constraints", constraints)

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())

    @property
    def request_id(self) -> str:
        return f"resample-{self.checksum[:16]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "group_id": self.group_id,
            "policy_version": self.policy_version,
            "required_count": self.required_count,
            "reason": self.reason,
            "sampling_constraints": thaw_json(self.sampling_constraints),
            "source_hint": self.source_hint,
        }
