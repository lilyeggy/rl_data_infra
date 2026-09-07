from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.contracts.trace_event import EventType
from src.errors import ContractValidationError
from src.storage import (
    BackpressureError,
    BoundedEventQueue,
    IngestionWorker,
    PartitionedEventStore,
    StorageSchemaRegistry,
    summarize_ingestion_latencies,
)
from tests.execution_fixtures import make_trace_event


class QueueSchemaAndCompactionTest(unittest.TestCase):
    def test_bounded_queue_applies_backpressure_and_worker_drains(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            events = tuple(
                make_trace_event(
                    event_id=f"evt-queue-{index}",
                    run_id="run-queue",
                    episode_id="episode-queue",
                    sequence=index,
                )
                for index in range(3)
            )
            queue = BoundedEventQueue(max_events=3)
            queue.put(events[:2])
            with self.assertRaises(BackpressureError):
                queue.put(events)
            store = PartitionedEventStore(temporary, durable=False)
            drained = IngestionWorker(queue, store).drain()

            self.assertEqual(drained.batches, 1)
            self.assertEqual(drained.events, 2)
            self.assertEqual(queue.snapshot().pending_events, 0)
            self.assertEqual(len(store.read_all()), 2)

    def test_latency_percentiles_are_deterministic(self) -> None:
        summary = summarize_ingestion_latencies((1, 2, 3, 4, 5, 100))

        self.assertEqual(summary.count, 6)
        self.assertEqual(summary.mean_ms, 19.166666666666668)
        self.assertEqual(summary.p95_ms, 100)
        self.assertEqual(summary.p99_ms, 100)
        self.assertEqual(summarize_ingestion_latencies(()).count, 0)

    def test_schema_registry_requires_explicit_migration(self) -> None:
        registry = StorageSchemaRegistry()
        payload = make_trace_event().to_dict()
        self.assertIsNone(registry.validate_event_payload(payload))
        unknown = dict(payload, schema_version="trace-event/v999")
        self.assertEqual(
            registry.validate_event_payload(unknown).code,
            "UNKNOWN_SCHEMA_VERSION",
        )
        with self.assertRaises(ContractValidationError):
            registry.migrate_event_payload(unknown)
        registry.register_migration(
            source_version="trace-event/v999",
            target_version="trace-event/v1",
            migrate=lambda value: dict(value, schema_version="trace-event/v1"),
        )
        migrated = registry.migrate_event_payload(unknown)
        self.assertEqual(registry.parse_trace_event(migrated).event_type, EventType.MODEL_REQUEST)

    def test_compaction_is_derived_and_does_not_rewrite_raw_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = PartitionedEventStore(temporary, durable=False)
            events = tuple(
                make_trace_event(
                    event_id=f"evt-compact-{index}",
                    run_id="run-compact",
                    episode_id="episode-compact",
                    sequence=sequence,
                )
                for index, sequence in enumerate((2, 0, 1))
            )
            store.ingest(events)
            raw_path = next((Path(temporary) / "partitions").rglob("raw-events.jsonl"))
            raw_before = raw_path.read_bytes()

            result = store.compact_partition(
                run_id="run-compact", episode_id="episode-compact"
            )
            compacted = store.read_compacted_partition(
                run_id="run-compact", episode_id="episode-compact"
            )

            self.assertEqual(result.input_event_count, 3)
            self.assertEqual(result.output_event_count, 3)
            self.assertEqual([event.sequence for event in compacted], [0, 1, 2])
            self.assertEqual(raw_path.read_bytes(), raw_before)
            self.assertTrue(result.compacted_path.endswith("compacted-events.jsonl"))


if __name__ == "__main__":
    unittest.main()
