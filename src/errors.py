"""Structured error types shared by contracts and source adapters."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class ErrorCode(str, Enum):
    """Stable machine-readable error semantics for the v1 pipeline."""

    ADAPTER_ERROR = "ADAPTER_ERROR"
    CAPABILITY_MISSING = "CAPABILITY_MISSING"
    CONTRACT_INVALID = "CONTRACT_INVALID"
    SOURCE_WARNING = "SOURCE_WARNING"


class PipelineError(Exception):
    """Base class for intentional, user-facing pipeline errors."""


class ContractValidationError(PipelineError, ValueError):
    """A canonical contract violates an internal invariant."""


class CapabilityMissingError(PipelineError):
    """A consumer requested capabilities that the input does not provide."""

    def __init__(self, missing: frozenset[object], context: str = "consumer") -> None:
        self.missing = missing
        self.context = context
        labels = ", ".join(sorted(str(getattr(item, "value", item)) for item in missing))
        super().__init__(f"{context} is missing required capabilities: {labels}")


class AdapterConversionError(PipelineError):
    """An AdapterResult with conversion errors cannot become a batch."""


@dataclass(frozen=True, slots=True)
class AdapterIssue:
    """One structured adapter warning or error."""

    code: ErrorCode
    message: str
    source_record_id: str | None = None
    field: str | None = None
    details: Mapping[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("AdapterIssue.message must be non-empty")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
