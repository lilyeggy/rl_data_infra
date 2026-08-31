"""Durable local primitives for the raw, append-only side of the data plane."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.contracts._json import canonical_json_bytes, sha256_bytes, sha256_json
from src.contracts.artifacts import ArtifactRef
from src.contracts.trace_event import TraceEvent
from src.errors import ContractValidationError


@dataclass(frozen=True, slots=True)
class AppendResult:
    appended: bool
    event_id: str
    checksum: str
    line_number: int


@dataclass(frozen=True, slots=True)
class JsonlIssue:
    line_number: int
    code: str
    message: str
    raw_sha256: str


@dataclass(frozen=True, slots=True)
class JsonlReadResult:
    events: tuple[TraceEvent, ...]
    issues: tuple[JsonlIssue, ...]
    raw_line_checksums: tuple[str, ...]


class EventJsonlReader:
    """Parse neighbors independently so one bad line does not erase valid facts."""

    @staticmethod
    def read(path: str | Path) -> JsonlReadResult:
        source = Path(path)
        events: list[TraceEvent] = []
        issues: list[JsonlIssue] = []
        checksums: list[str] = []
        if not source.exists():
            return JsonlReadResult((), (), ())
        with source.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                line_checksum = sha256_bytes(raw_line)
                checksums.append(line_checksum)
                try:
                    payload = json.loads(stripped)
                    events.append(TraceEvent.from_dict(payload))
                except (json.JSONDecodeError, UnicodeDecodeError, ContractValidationError) as exc:
                    issues.append(
                        JsonlIssue(
                            line_number=line_number,
                            code="MALFORMED_EVENT",
                            message=str(exc),
                            raw_sha256=line_checksum,
                        )
                    )
        return JsonlReadResult(tuple(events), tuple(issues), tuple(checksums))


class EventWriter:
    """Append canonical TraceEvents with idempotency on ``event_id``.

    This MVP uses a process-local writer and OS append semantics. A production
    multi-writer deployment would replace it with a transactional log or add a
    file lock; the contract and reader do not depend on that storage choice.
    """

    def __init__(self, path: str | Path, *, durable: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.durable = durable
        self._event_index: dict[str, tuple[str, int]] = {}
        read_result = EventJsonlReader.read(self.path)
        if read_result.issues:
            lines = [issue.line_number for issue in read_result.issues]
            raise ContractValidationError(
                f"cannot append to raw event log with malformed lines: {lines}"
            )
        for line_number, event in enumerate(read_result.events, start=1):
            prior = self._event_index.get(event.event_id)
            if prior is not None and prior[0] != event.checksum:
                raise ContractValidationError(
                    f"raw log has conflicting payloads for event_id {event.event_id!r}"
                )
            self._event_index.setdefault(event.event_id, (event.checksum, line_number))

    def append(self, event: TraceEvent) -> AppendResult:
        if not isinstance(event, TraceEvent):
            raise TypeError("EventWriter.append expects a TraceEvent")
        prior = self._event_index.get(event.event_id)
        if prior is not None:
            if prior[0] != event.checksum:
                raise ContractValidationError(
                    f"event_id {event.event_id!r} was replayed with different content"
                )
            return AppendResult(False, event.event_id, event.checksum, prior[1])

        line_number = len(self._event_index) + 1
        payload = canonical_json_bytes(event.to_dict()) + b"\n"
        with self.path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            if self.durable:
                os.fsync(handle.fileno())
        self._event_index[event.event_id] = (event.checksum, line_number)
        return AppendResult(True, event.event_id, event.checksum, line_number)

    def append_many(self, events: Iterable[TraceEvent]) -> tuple[AppendResult, ...]:
        return tuple(self.append(event) for event in events)


class ArtifactStore:
    """Write immutable evidence blobs under a content-addressed filename."""

    def __init__(self, root: str | Path, *, uri_prefix: str | None = None) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.uri_prefix = uri_prefix

    def put(
        self,
        content: bytes,
        *,
        kind: str,
        media_type: str,
        created_at: str,
        producer_event_id: str | None = None,
    ) -> ArtifactRef:
        if not isinstance(content, bytes):
            raise TypeError("ArtifactStore.put content must be bytes")
        digest = sha256_bytes(content)
        path = self.root / digest
        if path.exists():
            if path.read_bytes() != content:
                raise RuntimeError("SHA-256 collision or corrupted artifact store")
        else:
            with path.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        uri = (
            f"{self.uri_prefix.rstrip('/')}/{digest}"
            if self.uri_prefix is not None
            else path.as_posix()
        )
        reference_digest = sha256_json(
            {
                "sha256": digest,
                "kind": kind,
                "media_type": media_type,
                "producer_event_id": producer_event_id,
            }
        )
        return ArtifactRef(
            artifact_id=f"artifact-{reference_digest[:20]}",
            kind=kind,
            uri=uri,
            media_type=media_type,
            sha256=digest,
            size_bytes=len(content),
            producer_event_id=producer_event_id,
            created_at=created_at,
        )
