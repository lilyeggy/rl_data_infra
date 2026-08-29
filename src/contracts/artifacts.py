"""Content-addressed references to evidence kept outside event envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.contracts._json import sha256_json, validate_sha256
from src.contracts._validation import optional_text, required_text, strict_fields, utc_instant
from src.errors import ContractValidationError


SCHEMA_VERSION = "artifact-ref/v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactRef:
    artifact_id: str
    kind: str
    uri: str
    media_type: str
    sha256: str
    size_bytes: int
    producer_event_id: str | None
    created_at: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("artifact_id", "kind", "uri", "media_type"):
            required_text(getattr(self, name), name)
        optional_text(self.producer_event_id, "producer_event_id")
        validate_sha256(self.sha256, "sha256")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise ContractValidationError("size_bytes must be an integer")
        if self.size_bytes < 0:
            raise ContractValidationError("size_bytes must be non-negative")
        utc_instant(self.created_at, "created_at")
        if self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(f"schema_version must be {SCHEMA_VERSION!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "uri": self.uri,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "producer_event_id": self.producer_event_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRef":
        fields = {
            "schema_version",
            "artifact_id",
            "kind",
            "uri",
            "media_type",
            "sha256",
            "size_bytes",
            "producer_event_id",
            "created_at",
        }
        payload = strict_fields(value, fields, "ArtifactRef")
        return cls(**payload)

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())
