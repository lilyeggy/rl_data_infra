from __future__ import annotations

import multiprocessing
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.contracts.trace_event import TraceEvent
from src.storage import PartitionedEventStore
from tests.execution_fixtures import make_trace_event


def _ingest_process(root: str, payloads: tuple[dict, ...], queue) -> None:
    store = PartitionedEventStore(root, durable=False)
    events = tuple(TraceEvent.from_dict(payload) for payload in payloads)
    result = store.ingest(events)
    queue.put((len(result.appended_event_ids), len(result.duplicate_event_ids)))


class PartitionedEventStoreTest(unittest.TestCase):
    def test_partitions_by_run_and_episode_and_reloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = PartitionedEventStore(temporary, durable=False)
            first = make_trace_event(
                event_id="evt-store-1",
                run_id="run-a",
                episode_id="episode-a",
            )
            second = make_trace_event(
                event_id="evt-store-2",
                run_id="run-a",
                episode_id="episode-b",
            )
            result = store.ingest((first, second))

            self.assertEqual(result.appended_event_ids, ("evt-store-1", "evt-store-2"))
            self.assertEqual(len(store.read_all()), 2)
            manifest = store.partition_manifest(run_id="run-a", episode_id="episode-a")
            self.assertEqual(manifest["event_count"], 1)
            self.assertEqual(manifest["run_id"], "run-a")
            self.assertEqual(manifest["episode_id"], "episode-a")
            self.assertTrue((Path(temporary) / "partitions").is_dir())

            reloaded = PartitionedEventStore(temporary, durable=False)
            duplicate = reloaded.ingest((first,))
            self.assertEqual(duplicate.duplicate_event_ids, ("evt-store-1",))
            self.assertFalse(duplicate.appended_event_ids)

    def test_conflicting_event_is_quarantined_not_appended(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = PartitionedEventStore(temporary, durable=False)
            original = make_trace_event(
                event_id="evt-conflict", run_id="run-a", episode_id="episode-a"
            )
            store.ingest((original,))
            conflict = replace(original, attributes={"changed": True})

            result = store.ingest((conflict,))

            self.assertFalse(result.appended_event_ids)
            self.assertEqual(len(result.quarantined), 1)
            self.assertEqual(result.quarantined[0].reason_code, "CONFLICTING_EVENT_ID")
            self.assertEqual(len(store.read_all()), 1)
            quarantine = Path(temporary) / "quarantine.jsonl"
            self.assertTrue(quarantine.is_file())
            self.assertEqual(len(quarantine.read_text().splitlines()), 1)
            raw_files = list((Path(temporary) / "partitions").rglob("raw-events.jsonl"))
            self.assertEqual(sum(len(path.read_text().splitlines()) for path in raw_files), 1)

    def test_batch_ingestion_is_deterministic_and_supports_throughput_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = PartitionedEventStore(temporary, durable=False)
            events = tuple(
                make_trace_event(
                    event_id=f"evt-batch-{index}",
                    run_id="run-batch",
                    episode_id=f"episode-{index % 4}",
                    sequence=index,
                )
                for index in range(200)
            )

            result = store.ingest(events)

            self.assertEqual(len(result.appended_event_ids), 200)
            self.assertEqual(len(store.read_all()), 200)
            replay = store.ingest(events)
            self.assertEqual(result.input_checksum, replay.input_checksum)
            self.assertEqual(len(replay.duplicate_event_ids), 200)
            self.assertEqual(len(result.partition_manifests), 4)
            self.assertTrue(all(item["event_count"] == 50 for item in result.partition_manifests))

    def test_multiple_processes_share_event_idempotency_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            events = tuple(
                make_trace_event(
                    event_id=f"evt-process-{index}",
                    run_id="run-process",
                    episode_id="episode-process",
                    sequence=index,
                )
                for index in range(20)
            )
            payloads = tuple(event.to_dict() for event in events)
            context = multiprocessing.get_context("fork")
            queue = context.Queue()
            processes = [
                context.Process(target=_ingest_process, args=(temporary, payloads, queue))
                for _ in range(2)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)
            results = [queue.get(timeout=5) for _ in processes]

            store = PartitionedEventStore(temporary, durable=False)
            self.assertEqual(len(store.read_all()), 20)
            self.assertEqual(sum(item[0] for item in results), 20)
            self.assertEqual(sum(item[1] for item in results), 20)

    def test_recover_rebuilds_manifest_from_raw_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = PartitionedEventStore(temporary, durable=False)
            event = make_trace_event(
                event_id="evt-recover",
                run_id="run-recover",
                episode_id="episode-recover",
            )
            store.ingest((event,))
            partition = next((Path(temporary) / "partitions").rglob("partition-manifest.json"))
            partition.unlink()

            recovery = store.recover()

            self.assertEqual(recovery.partitions_scanned, 1)
            self.assertEqual(recovery.manifests_rebuilt, 1)
            self.assertEqual(recovery.event_count, 1)
            self.assertFalse(recovery.issues)
            self.assertTrue(partition.is_file())
            self.assertEqual(len(store.read_all()), 1)

    def test_recover_reports_malformed_partition_without_rewriting_raw_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = PartitionedEventStore(temporary, durable=False)
            event = make_trace_event(
                event_id="evt-malformed",
                run_id="run-malformed",
                episode_id="episode-malformed",
            )
            store.ingest((event,))
            raw_path = next((Path(temporary) / "partitions").rglob("raw-events.jsonl"))
            raw_path.write_text(raw_path.read_text() + "{broken\n")
            before = raw_path.read_bytes()

            recovery = store.recover()

            self.assertEqual(recovery.event_count, 1)
            self.assertEqual(recovery.manifests_rebuilt, 0)
            self.assertTrue(recovery.issues)
            self.assertEqual(raw_path.read_bytes(), before)

    def test_partition_identity_uses_safe_digest_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = PartitionedEventStore(temporary, durable=False)
            event = make_trace_event(
                event_id="evt-path-safe",
                run_id="run/with/slash",
                episode_id="episode/with/slash",
            )
            store.ingest((event,))

            paths = list((Path(temporary) / "partitions").rglob("identity.json"))
            self.assertEqual(len(paths), 1)
            self.assertNotIn("/with/", paths[0].as_posix())


if __name__ == "__main__":
    unittest.main()
