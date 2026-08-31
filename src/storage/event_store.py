"""Partitioned append-only event ingestion for the V2.3 data plane.

This is a dependency-free filesystem adapter with explicit process-local
semantics.  It is suitable for container jobs and replay tests; it does not
claim distributed exactly-once behavior.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.capture.event_writer import EventJsonlReader, EventWriter
from src.contracts._json import canonical_json_bytes, sha256_bytes, sha256_json
from src.contracts.trace_event import TraceEvent
from src.errors import ContractValidationError


EVENT_INGESTION_VERSION = "event-ingestion/v1"
PARTITION_MANIFEST_VERSION = "storage-partition/v1"
STORAGE_RECOVERY_VERSION = "storage-recovery/v1"
COMPACTION_VERSION = "storage-compaction/v1"


@dataclass(frozen=True, slots=True)
class QuarantinedIngestion:
    event_id: str
    reason_code: str
    event_checksum: str
    existing_checksum: str | None = None
    partition: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "reason_code": self.reason_code,
            "event_checksum": self.event_checksum,
            "existing_checksum": self.existing_checksum,
            "partition": self.partition,
        }


@dataclass(frozen=True, slots=True)
class CompactionResult:
    run_id: str
    episode_id: str
    input_event_count: int
    output_event_count: int
    raw_checksum: str
    compacted_checksum: str
    compacted_path: str
    compaction_version: str = COMPACTION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "compaction_version": self.compaction_version,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "input_event_count": self.input_event_count,
            "output_event_count": self.output_event_count,
            "raw_checksum": self.raw_checksum,
            "compacted_checksum": self.compacted_checksum,
            "compacted_path": self.compacted_path,
        }


@dataclass(frozen=True, slots=True)
class StorageRecoveryResult:
    partitions_scanned: int
    manifests_rebuilt: int
    event_count: int
    issues: tuple[str, ...]
    recovery_version: str = STORAGE_RECOVERY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "recovery_version": self.recovery_version,
            "partitions_scanned": self.partitions_scanned,
            "manifests_rebuilt": self.manifests_rebuilt,
            "event_count": self.event_count,
            "issues": list(self.issues),
        }


@dataclass(frozen=True, slots=True)
class IngestionBatchResult:
    input_checksum: str
    appended_event_ids: tuple[str, ...]
    duplicate_event_ids: tuple[str, ...]
    quarantined: tuple[QuarantinedIngestion, ...]
    partition_manifests: tuple[MappingLike, ...]
    ingestion_version: str = EVENT_INGESTION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "ingestion_version": self.ingestion_version,
            "input_checksum": self.input_checksum,
            "appended_event_ids": list(self.appended_event_ids),
            "duplicate_event_ids": list(self.duplicate_event_ids),
            "quarantined": [item.to_dict() for item in self.quarantined],
            "partition_manifests": [dict(item) for item in self.partition_manifests],
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


class _ProcessFileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


MappingLike = dict[str, Any]


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


class PartitionedEventStore:
    """Run/Episode-partitioned append-only storage with global event idempotency.

    The implementation uses process-local locks and one ``EventWriter`` per
    partition.  A future object-store or transactional-log adapter can replace
    this class while preserving the ingestion result and quarantine semantics.
    """

    def __init__(self, root: str | Path, *, durable: bool = True) -> None:
        self.root = Path(root)
        self.partitions_root = self.root / "partitions"
        self.partitions_root.mkdir(parents=True, exist_ok=True)
        self.quarantine_path = self.root / "quarantine.jsonl"
        self.lock_path = self.root / ".ingestion.lock"
        self.durable = durable
        self._lock = threading.RLock()
        self._writers: dict[Path, EventWriter] = {}
        self._event_index: dict[str, tuple[str, Path]] = {}
        self._load_existing_index()

    def ingest(self, events: Iterable[TraceEvent]) -> IngestionBatchResult:
        batch = tuple(events)
        if any(not isinstance(event, TraceEvent) for event in batch):
            raise TypeError("PartitionedEventStore accepts TraceEvent values")
        input_checksum = sha256_json([event.to_dict() for event in batch])
        appended: list[str] = []
        duplicates: list[str] = []
        quarantined: list[QuarantinedIngestion] = []
        grouped: dict[Path, list[TraceEvent]] = {}
        with self._lock, self._process_lock():
            self._refresh_index()
            pending: dict[str, str] = {}
            for event in batch:
                checksum = event.checksum
                prior = self._event_index.get(event.event_id)
                if prior is not None:
                    if prior[0] == checksum:
                        duplicates.append(event.event_id)
                    else:
                        item = QuarantinedIngestion(
                            event_id=event.event_id,
                            reason_code="CONFLICTING_EVENT_ID",
                            event_checksum=checksum,
                            existing_checksum=prior[0],
                            partition=prior[1].as_posix(),
                        )
                        quarantined.append(item)
                    continue
                prior_pending = pending.get(event.event_id)
                if prior_pending is not None:
                    if prior_pending == checksum:
                        duplicates.append(event.event_id)
                    else:
                        quarantined.append(
                            QuarantinedIngestion(
                                event_id=event.event_id,
                                reason_code="CONFLICTING_BATCH_EVENT_ID",
                                event_checksum=checksum,
                                existing_checksum=prior_pending,
                                partition=self._partition_path(event).as_posix(),
                            )
                        )
                    continue
                pending[event.event_id] = checksum
                grouped.setdefault(self._partition_path(event), []).append(event)

            for partition, partition_events in sorted(
                grouped.items(), key=lambda item: item[0].as_posix()
            ):
                writer = self._writer(partition)
                results = writer.append_many(partition_events)
                for event, result in zip(partition_events, results):
                    if result.appended:
                        appended.append(event.event_id)
                        self._event_index[event.event_id] = (
                            event.checksum,
                            partition,
                        )
                self._write_partition_manifest(partition)

            if quarantined:
                self._append_quarantine(quarantined)
            manifests = tuple(
                self._read_partition_manifest(partition)
                for partition in sorted(grouped, key=lambda item: item.as_posix())
            )
        appended_set = set(appended)
        duplicate_set = set(duplicates)
        return IngestionBatchResult(
            input_checksum=input_checksum,
            appended_event_ids=tuple(
                event_id
                for event_id in _ordered_unique(event.event_id for event in batch)
                if event_id in appended_set
            ),
            duplicate_event_ids=tuple(
                event_id
                for event_id in _ordered_unique(event.event_id for event in batch)
                if event_id in duplicate_set
            ),
            quarantined=tuple(quarantined),
            partition_manifests=manifests,
        )

    def compact_partition(self, *, run_id: str, episode_id: str) -> CompactionResult:
        """Write a deterministic derived compacted view without touching raw events."""

        with self._lock, self._process_lock():
            partition = self._partition_path_for_ids(run_id, episode_id)
            result = EventJsonlReader.read(partition / "raw-events.jsonl")
            if result.issues:
                raise ContractValidationError(
                    f"cannot compact malformed partition: {partition}"
                )
            ordered = tuple(
                sorted(
                    {event.event_id: event for event in result.events}.values(),
                    key=lambda event: (event.sequence, event.timestamp, event.event_id),
                )
            )
            payload = b"".join(
                canonical_json_bytes(event.to_dict()) + b"\n" for event in ordered
            )
            compacted_path = partition / "compacted-events.jsonl"
            self._atomic_write_bytes(compacted_path, payload)
            compacted_checksum = sha256_bytes(payload)
            manifest = {
                "schema_version": COMPACTION_VERSION,
                "run_id": run_id,
                "episode_id": episode_id,
                "raw_checksum": sha256_json([event.to_dict() for event in result.events]),
                "compacted_checksum": compacted_checksum,
                "input_event_count": len(result.events),
                "output_event_count": len(ordered),
                "compacted_path": compacted_path.as_posix(),
            }
            self._atomic_write_json(partition / "compaction-manifest.json", manifest)
            return CompactionResult(
                run_id=run_id,
                episode_id=episode_id,
                input_event_count=len(result.events),
                output_event_count=len(ordered),
                raw_checksum=manifest["raw_checksum"],
                compacted_checksum=compacted_checksum,
                compacted_path=compacted_path.as_posix(),
            )

    def read_compacted_partition(
        self, *, run_id: str, episode_id: str
    ) -> tuple[TraceEvent, ...]:
        partition = self._partition_path_for_ids(run_id, episode_id)
        return EventJsonlReader.read(partition / "compacted-events.jsonl").events

    def recover(self) -> StorageRecoveryResult:
        """Rebuild derived indexes/manifests from append-only raw partitions."""

        with self._lock, self._process_lock():
            issues = self._refresh_index(allow_malformed=True)
            partitions = sorted(self.partitions_root.rglob("raw-events.jsonl"))
            rebuilt = 0
            event_count = 0
            for raw_path in partitions:
                result = EventJsonlReader.read(raw_path)
                event_count += len(result.events)
                if result.issues:
                    message = f"{raw_path}: malformed raw events"
                    if message not in issues:
                        issues.append(message)
                    continue
                self._write_partition_manifest(raw_path.parent)
                rebuilt += 1
            return StorageRecoveryResult(
                partitions_scanned=len(partitions),
                manifests_rebuilt=rebuilt,
                event_count=event_count,
                issues=tuple(issues),
            )

    def read_partition(self, *, run_id: str, episode_id: str) -> tuple[TraceEvent, ...]:
        partition = self._partition_path_for_ids(run_id, episode_id)
        return EventJsonlReader.read(partition / "raw-events.jsonl").events

    def read_all(self) -> tuple[TraceEvent, ...]:
        events: list[TraceEvent] = []
        for path in sorted(self.partitions_root.rglob("raw-events.jsonl")):
            result = EventJsonlReader.read(path)
            if result.issues:
                raise ContractValidationError(
                    f"partition has malformed events: {path}: {result.issues}"
                )
            events.extend(result.events)
        return tuple(events)

    def partition_manifest(self, *, run_id: str, episode_id: str) -> MappingLike:
        return self._read_partition_manifest(
            self._partition_path_for_ids(run_id, episode_id)
        )

    def _refresh_index(self, *, allow_malformed: bool = False) -> list[str]:
        self._event_index.clear()
        self._writers.clear()
        issues: list[str] = []
        for path in sorted(self.partitions_root.rglob("raw-events.jsonl")):
            result = EventJsonlReader.read(path)
            if result.issues:
                if not allow_malformed:
                    lines = [issue.line_number for issue in result.issues]
                    raise ContractValidationError(
                        f"cannot append to malformed partition {path}: {lines}"
                    )
                issues.append(f"{path}: malformed raw events")
            for event in result.events:
                prior = self._event_index.get(event.event_id)
                if prior is not None and prior[0] != event.checksum:
                    issue = f"{path}: global event_id conflict {event.event_id}"
                    if not allow_malformed:
                        raise ContractValidationError(issue)
                    issues.append(issue)
                    continue
                self._event_index.setdefault(event.event_id, (event.checksum, path.parent))
        return issues

    def _load_existing_index(self) -> None:
        for path in sorted(self.partitions_root.rglob("raw-events.jsonl")):
            result = EventJsonlReader.read(path)
            if result.issues:
                raise ContractValidationError(
                    f"cannot open storage with malformed partition: {path}"
                )
            for event in result.events:
                prior = self._event_index.get(event.event_id)
                if prior is not None and prior[0] != event.checksum:
                    raise ContractValidationError(
                        f"global event_id conflict while loading storage: {event.event_id}"
                    )
                self._event_index.setdefault(event.event_id, (event.checksum, path.parent))

    def _writer(self, partition: Path) -> EventWriter:
        writer = self._writers.get(partition)
        if writer is None:
            writer = EventWriter(partition / "raw-events.jsonl", durable=self.durable)
            self._writers[partition] = writer
        return writer

    def _partition_path(self, event: TraceEvent) -> Path:
        return self._partition_path_for_ids(event.run_id, event.episode_id)

    def _partition_path_for_ids(self, run_id: str, episode_id: str) -> Path:
        run_key = sha256_bytes(run_id.encode("utf-8"))[:20]
        episode_key = sha256_bytes(episode_id.encode("utf-8"))[:20]
        partition = self.partitions_root / f"run-{run_key}" / f"episode-{episode_key}"
        partition.mkdir(parents=True, exist_ok=True)
        metadata = partition / "identity.json"
        if not metadata.exists():
            self._atomic_write_json(
                metadata,
                {
                    "schema_version": PARTITION_MANIFEST_VERSION,
                    "run_id": run_id,
                    "episode_id": episode_id,
                },
            )
        else:
            identity = json.loads(metadata.read_text())
            if identity.get("run_id") != run_id or identity.get("episode_id") != episode_id:
                raise ContractValidationError("partition identity collision")
        return partition

    def _write_partition_manifest(self, partition: Path) -> None:
        events = EventJsonlReader.read(partition / "raw-events.jsonl").events
        payload = {
            "schema_version": PARTITION_MANIFEST_VERSION,
            "storage_mode": "process-local-durable-filesystem",
            "run_id": events[0].run_id if events else None,
            "episode_id": events[0].episode_id if events else None,
            "event_count": len(events),
            "event_ids": [event.event_id for event in events],
            "event_checksums": [event.checksum for event in events],
            "events_checksum": sha256_json([event.to_dict() for event in events]),
            "raw_events_path": (partition / "raw-events.jsonl").as_posix(),
        }
        self._atomic_write_json(partition / "partition-manifest.json", payload)

    def _process_lock(self):
        return _ProcessFileLock(self.lock_path)

    @staticmethod
    def _read_partition_manifest(partition: Path) -> MappingLike:
        path = partition / "partition-manifest.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def _append_quarantine(self, items: Iterable[QuarantinedIngestion]) -> None:
        payload = b"".join(
            canonical_json_bytes(item.to_dict()) + b"\n" for item in items
        )
        with self.quarantine_path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            if self.durable:
                os.fsync(handle.fileno())

    def _atomic_write_bytes(self, path: Path, payload: bytes) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            if self.durable:
                os.fsync(handle.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _atomic_write_json(path: Path, value: MappingLike) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        payload = canonical_json_bytes(value) + b"\n"
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
