from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.capture.event_writer import ArtifactStore, EventJsonlReader, EventWriter
from src.capture.recorder import TraceRecorder
from src.contracts.trace_event import EventComponent, EventStatus, EventType
from src.errors import ContractValidationError
from tests.execution_fixtures import make_trace_event


class EventWriterTest(unittest.TestCase):
    def test_append_is_durable_idempotent_and_rejects_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw-events.jsonl"
            writer = EventWriter(path)
            event = make_trace_event()

            self.assertTrue(writer.append(event).appended)
            self.assertFalse(writer.append(event).appended)
            self.assertEqual(len(path.read_text().splitlines()), 1)

            reloaded = EventWriter(path)
            self.assertFalse(reloaded.append(event).appended)
            with self.assertRaisesRegex(ContractValidationError, "different content"):
                reloaded.append(replace(event, attributes={"changed": True}))

    def test_reader_quarantines_bad_line_without_losing_neighbors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw-events.jsonl"
            valid = json.dumps(make_trace_event().to_dict(), separators=(",", ":"))
            path.write_text(valid + "\n{broken\n" + valid + "\n")

            result = EventJsonlReader.read(path)

            self.assertEqual(len(result.events), 2)
            self.assertEqual(len(result.issues), 1)
            self.assertEqual(result.issues[0].line_number, 2)

    def test_recorder_assigns_sequence_and_redacts_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            writer = EventWriter(Path(temporary) / "events.jsonl")
            ids = iter(("1", "2"))
            recorder = TraceRecorder(
                writer,
                run_id="run-001",
                episode_id="episode-001",
                trace_id="trace-001",
                clock=lambda: "2026-08-14T00:00:00Z",
                id_factory=lambda: next(ids),
            )
            first = recorder.emit(
                EventType.MODEL_REQUEST,
                EventComponent.MODEL,
                EventStatus.STARTED,
                span_id="span-1",
                parent_span_id=None,
                attributes={"headers": {"Authorization": "Bearer secret"}},
            )
            second = recorder.emit(
                EventType.MODEL_RESPONSE,
                EventComponent.MODEL_BACKEND,
                EventStatus.SUCCEEDED,
                span_id="span-1",
                parent_span_id=None,
            )

            self.assertEqual((first.sequence, second.sequence), (0, 1))
            self.assertEqual(first.attributes["headers"]["Authorization"], "[REDACTED]")

    def test_artifact_store_is_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ArtifactStore(temporary)
            first = store.put(
                b"same bytes",
                kind="stdout",
                media_type="text/plain",
                created_at="2026-08-14T00:00:00Z",
            )
            second = store.put(
                b"same bytes",
                kind="stdout",
                media_type="text/plain",
                created_at="2026-08-14T00:00:00Z",
            )

            self.assertEqual(first.artifact_id, second.artifact_id)
            self.assertTrue(Path(first.uri).is_file())


if __name__ == "__main__":
    unittest.main()
