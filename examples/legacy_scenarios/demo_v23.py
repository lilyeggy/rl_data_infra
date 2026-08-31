"""HISTORICAL EXAMPLE: V2.3 partitioned-storage ingestion smoke package."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from src.capture.event_writer import EventJsonlReader
from examples.legacy_scenarios.demo_v21_runtime import generate_v21_runtime_evidence
from src.storage import (
    BoundedEventQueue,
    IngestionWorker,
    PartitionedEventStore,
    StorageSchemaRegistry,
    summarize_ingestion_latencies,
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def generate_v23_storage_smoke(output_dir: str | Path) -> dict[str, Any]:
    """Ingest complete V2.1 local runtime events into partitioned storage."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    storage_root = output / "storage"
    if storage_root.exists():
        shutil.rmtree(storage_root)
    for name in (
        "first-ingestion.json",
        "replay-ingestion.json",
        "storage-summary.json",
        "summary.json",
    ):
        path = output / name
        if path.exists():
            path.unlink()
    with tempfile.TemporaryDirectory() as temporary:
        source_dir = Path(temporary) / "source"
        generate_v21_runtime_evidence(source_dir)
        events = EventJsonlReader.read(source_dir / "raw-events.jsonl").events

        storage = PartitionedEventStore(storage_root, durable=True)
        registry = StorageSchemaRegistry()
        for event in events:
            if registry.validate_event_payload(event.to_dict()) is not None:
                raise RuntimeError("generated event failed schema registry validation")

        queue = BoundedEventQueue(max_events=len(events))
        queue.put(events)
        first_started = time.perf_counter()
        first_drain = IngestionWorker(queue, storage).drain()
        first_latency_ms = (time.perf_counter() - first_started) * 1000
        first = first_drain.ingestion_results[0]

        replay_queue = BoundedEventQueue(max_events=len(events))
        replay_queue.put(events)
        replay_started = time.perf_counter()
        replay_drain = IngestionWorker(replay_queue, storage).drain()
        replay_latency_ms = (time.perf_counter() - replay_started) * 1000
        replay = replay_drain.ingestion_results[0]
        read_back = storage.read_all()
        partition_ids = tuple(
            dict.fromkeys((event.run_id, event.episode_id) for event in events)
        )
        compactions = tuple(
            storage.compact_partition(run_id=run_id, episode_id=episode_id)
            for run_id, episode_id in partition_ids
        )
        latency_summary = summarize_ingestion_latencies(
            (first_latency_ms, replay_latency_ms)
        )

    manifests = [
        storage.partition_manifest(run_id=event.run_id, episode_id=event.episode_id)
        for event in events
    ]
    unique_manifests = {
        (item.get("run_id"), item.get("episode_id")): item for item in manifests
    }
    _write_json(output / "first-ingestion.json", first.to_dict())
    _write_json(output / "replay-ingestion.json", replay.to_dict())
    _write_json(output / "compaction.json", [item.to_dict() for item in compactions])
    _write_json(output / "ingestion-latency.json", latency_summary.to_dict())
    _write_json(
        output / "storage-summary.json",
        {
            "storage_mode": "process-local-durable-filesystem",
            "partition_manifest_version": "storage-partition/v1",
            "event_ingestion_version": "event-ingestion/v1",
            "partition_count": len(unique_manifests),
            "partition_manifests": list(unique_manifests.values()),
            "read_back_event_count": len(read_back),
            "distributed_exactly_once": False,
            "compaction_count": len(compactions),
            "compaction_output_event_count": sum(
                item.output_event_count for item in compactions
            ),
            "ingestion_latency": latency_summary.to_dict(),
        },
    )
    summary = {
        "release": "v2.3",
        "scope": "partitioned append-only storage and ingestion smoke package",
        "storage_mode": "process-local-durable-filesystem",
        "input_event_count": len(events),
        "first_append_count": len(first.appended_event_ids),
        "replay_duplicate_count": len(replay.duplicate_event_ids),
        "quarantine_count": len(first.quarantined) + len(replay.quarantined),
        "partition_count": len(unique_manifests),
        "read_back_event_count": len(read_back),
        "distributed_exactly_once": False,
        "compaction_count": len(compactions),
        "schema_validation": "explicit-supported-version-only",
        "ingestion_latency": latency_summary.to_dict(),
        "claim_boundary": (
            "filesystem adapter validates partitioning, durability, idempotency and "
            "quarantine; it is not a distributed production storage service"
        ),
    }
    _write_json(output / "summary.json", summary)
    return summary
