"""Training-ready policy-consistent batch contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.contracts._json import sha256_json, validate_sha256
from src.contracts.capabilities import (
    Capability,
    TRAINING_CORE_CAPABILITIES,
    capabilities_for_record,
    require_capabilities,
)
from src.contracts.rollout_record import RolloutRecord
from src.errors import ContractValidationError


TRAINING_BATCH_SCHEMA_VERSION = "training-ready-batch/v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingReadyBatch:
    """A fixed group that is safe for a Trainer Adapter to consume."""

    records: tuple[RolloutRecord, ...]
    task_id: str
    group_id: str
    policy_version: str
    source_batch_checksum: str
    processing_versions: tuple[str, ...]
    required_capabilities: frozenset[Capability] = field(
        default_factory=lambda: TRAINING_CORE_CAPABILITIES
    )
    schema_version: str = TRAINING_BATCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "processing_versions", tuple(self.processing_versions))
        object.__setattr__(self, "required_capabilities", frozenset(self.required_capabilities))
        if self.schema_version != TRAINING_BATCH_SCHEMA_VERSION:
            raise ContractValidationError(
                f"schema_version must be {TRAINING_BATCH_SCHEMA_VERSION!r}"
            )
        if not self.records:
            raise ContractValidationError("TrainingReadyBatch requires at least one record")
        for name in ("task_id", "group_id", "policy_version"):
            if not getattr(self, name).strip():
                raise ContractValidationError(f"{name} must be non-empty")
        validate_sha256(self.source_batch_checksum, "source_batch_checksum")
        if not self.processing_versions or any(
            not isinstance(item, str) or not item.strip() for item in self.processing_versions
        ):
            raise ContractValidationError("processing_versions must contain non-empty versions")

        trajectory_ids: set[str] = set()
        for record in self.records:
            if record.trajectory_id in trajectory_ids:
                raise ContractValidationError(
                    f"duplicate trajectory_id in training batch: {record.trajectory_id}"
                )
            trajectory_ids.add(record.trajectory_id)
            if (
                record.task_id != self.task_id
                or record.group_id != self.group_id
                or record.policy_version != self.policy_version
            ):
                raise ContractValidationError(
                    "all training records must match task_id, group_id, and policy_version"
                )
            require_capabilities(
                capabilities_for_record(record),
                self.required_capabilities,
                context=f"trajectory {record.trajectory_id}",
            )

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())

    @property
    def batch_id(self) -> str:
        return f"training-{self.checksum[:16]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "group_id": self.group_id,
            "policy_version": self.policy_version,
            "source_batch_checksum": self.source_batch_checksum,
            "processing_versions": list(self.processing_versions),
            "required_capabilities": sorted(item.value for item in self.required_capabilities),
            "records": [record.to_dict() for record in self.records],
        }
