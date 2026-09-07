"""Explicit storage schema registry and migration boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from src.contracts.trace_event import SCHEMA_VERSION, TraceEvent
from src.errors import ContractValidationError


SCHEMA_REGISTRY_VERSION = "storage-schema-registry/v1"


@dataclass(frozen=True, slots=True)
class SchemaIssue:
    code: str
    schema_version: str | None
    message: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "schema_version": self.schema_version,
            "message": self.message,
        }


class StorageSchemaRegistry:
    """Reject unknown schemas unless an explicit migration is registered."""

    def __init__(self) -> None:
        self._migrations: dict[
            tuple[str, str], Callable[[Mapping[str, Any]], Mapping[str, Any]]
        ] = {}
        self._supported = {SCHEMA_VERSION}

    def register_migration(
        self,
        *,
        source_version: str,
        target_version: str,
        migrate: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> None:
        if not source_version.strip() or not target_version.strip():
            raise ValueError("schema versions must be non-empty")
        if not callable(migrate):
            raise TypeError("migrate must be callable")
        self._migrations[(source_version, target_version)] = migrate
        self._supported.add(target_version)

    def validate_event_payload(self, payload: Mapping[str, Any]) -> SchemaIssue | None:
        version = payload.get("schema_version") if isinstance(payload, Mapping) else None
        if version in self._supported:
            return None
        return SchemaIssue(
            code="UNKNOWN_SCHEMA_VERSION",
            schema_version=version if isinstance(version, str) else None,
            message="event schema version is not registered; event was not coerced",
        )

    def migrate_event_payload(
        self,
        payload: Mapping[str, Any],
        *,
        target_version: str = SCHEMA_VERSION,
    ) -> Mapping[str, Any]:
        if not isinstance(payload, Mapping):
            raise ContractValidationError("event payload must be an object")
        source_version = payload.get("schema_version")
        if source_version == target_version:
            return payload
        migration = self._migrations.get((source_version, target_version))
        if migration is None:
            raise ContractValidationError(
                f"no explicit migration from {source_version!r} to {target_version!r}"
            )
        migrated = migration(payload)
        if not isinstance(migrated, Mapping):
            raise ContractValidationError("schema migration must return an object")
        if migrated.get("schema_version") != target_version:
            raise ContractValidationError(
                "schema migration returned the wrong target schema version"
            )
        return migrated

    def parse_trace_event(self, payload: Mapping[str, Any]) -> TraceEvent:
        issue = self.validate_event_payload(payload)
        if issue is not None:
            raise ContractValidationError(issue.message)
        return TraceEvent.from_dict(payload)
