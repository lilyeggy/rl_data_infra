"""Source Adapter protocol and common result envelope."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.contracts.capabilities import Capability, common_capabilities
from src.contracts.rollout_batch import RolloutBatch
from src.contracts.rollout_record import RolloutRecord
from src.errors import AdapterConversionError, AdapterIssue


@dataclass(frozen=True, slots=True, kw_only=True)
class AdapterResult:
    records: tuple[RolloutRecord, ...] = ()
    capabilities: frozenset[Capability] = frozenset()
    warnings: tuple[AdapterIssue, ...] = ()
    errors: tuple[AdapterIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "errors", tuple(self.errors))
        guaranteed = common_capabilities(self.records)
        if not self.capabilities.issubset(guaranteed):
            raise ValueError("AdapterResult cannot declare capabilities missing from a record")

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_batch(self, *, adapter_name: str, adapter_version: str) -> RolloutBatch:
        if self.errors:
            raise AdapterConversionError(
                f"{adapter_name} produced {len(self.errors)} error(s); refusing batch conversion"
            )
        return RolloutBatch.from_records(
            self.records,
            source_adapter=adapter_name,
            source_adapter_version=adapter_version,
        )


@runtime_checkable
class SourceAdapter(Protocol):
    name: str
    version: str

    def capabilities(self) -> frozenset[Capability]: ...

    def convert(self, payload: object) -> AdapterResult: ...
