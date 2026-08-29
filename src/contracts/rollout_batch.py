"""Canonical input batch for rollout processors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from src.contracts._json import sha256_json
from src.contracts.capabilities import Capability, common_capabilities
from src.contracts.rollout_record import RolloutRecord
from src.errors import ContractValidationError


BATCH_SCHEMA_VERSION = "rollout-batch/v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class RolloutBatch:
    records: tuple[RolloutRecord, ...]
    source_adapter: str
    source_adapter_version: str
    schema_version: str = BATCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        if not self.source_adapter.strip() or not self.source_adapter_version.strip():
            raise ContractValidationError("source adapter name and version must be non-empty")
        if self.schema_version != BATCH_SCHEMA_VERSION:
            raise ContractValidationError(f"schema_version must be {BATCH_SCHEMA_VERSION!r}")
        trajectory_ids = [record.trajectory_id for record in self.records]
        duplicates = sorted(
            trajectory_id
            for trajectory_id in set(trajectory_ids)
            if trajectory_ids.count(trajectory_id) > 1
        )
        if duplicates:
            raise ContractValidationError(
                "duplicate trajectory_id values: " + ", ".join(duplicates)
            )

    @classmethod
    def from_records(
        cls,
        records: Iterable[RolloutRecord],
        *,
        source_adapter: str,
        source_adapter_version: str,
    ) -> "RolloutBatch":
        return cls(
            records=tuple(records),
            source_adapter=source_adapter,
            source_adapter_version=source_adapter_version,
        )

    @property
    def capabilities(self) -> frozenset[Capability]:
        """Capabilities guaranteed to exist on every record in the batch."""

        return common_capabilities(self.records)

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_adapter": self.source_adapter,
            "source_adapter_version": self.source_adapter_version,
            "records": [record.to_dict() for record in self.records],
        }
