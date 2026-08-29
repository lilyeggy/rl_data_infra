from __future__ import annotations

import unittest

from src.contracts.trace_event import (
    EventComponent,
    EventStatus,
    EventType,
    SCHEMA_VERSION,
    TraceEvent,
)
from src.errors import ContractValidationError
from tests.execution_fixtures import make_trace_event


class TraceEventContractTest(unittest.TestCase):
    def test_valid_event_is_immutable_and_freezes_nested_json(self) -> None:
        event = make_trace_event(
            attributes={"nested": {"items": [1, 2], "enabled": True}}
        )

        with self.assertRaises(TypeError):
            event.attributes["new"] = "value"  # type: ignore[index]
        with self.assertRaises(TypeError):
            event.attributes["nested"]["items"][0] = 9  # type: ignore[index]

    def test_roundtrip_and_checksum_are_deterministic(self) -> None:
        first = make_trace_event(attributes={"b": 2, "a": 1})
        second = make_trace_event(attributes={"a": 1, "b": 2})

        restored = TraceEvent.from_dict(first.to_dict())

        self.assertEqual(restored, first)
        self.assertEqual(first.checksum, second.checksum)
        self.assertEqual(restored.checksum, first.checksum)

    def test_to_dict_uses_json_values_and_preserves_the_exact_envelope(self) -> None:
        payload = make_trace_event().to_dict()

        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "event_id",
                "run_id",
                "episode_id",
                "trace_id",
                "span_id",
                "parent_span_id",
                "sequence",
                "timestamp",
                "event_type",
                "component",
                "status",
                "attempt",
                "attributes",
                "artifact_refs",
            },
        )
        self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
        self.assertEqual(payload["event_type"], "MODEL_REQUEST")
        self.assertEqual(payload["component"], "MODEL")
        self.assertEqual(payload["status"], "STARTED")
        self.assertIsInstance(payload["attributes"], dict)
        self.assertIsInstance(payload["artifact_refs"], list)

    def test_from_dict_rejects_unknown_top_level_fields(self) -> None:
        payload = self._valid_payload()
        payload["private_harness_state"] = {"decision": "hidden"}

        with self.assertRaisesRegex(ContractValidationError, "attributes"):
            TraceEvent.from_dict(payload)

    def test_requires_non_empty_identity_fields(self) -> None:
        for field_name in (
            "event_id",
            "run_id",
            "episode_id",
            "trace_id",
            "span_id",
        ):
            with self.subTest(field=field_name):
                with self.assertRaisesRegex(ContractValidationError, field_name):
                    make_trace_event(**{field_name: "  "})

        with self.assertRaisesRegex(ContractValidationError, "parent_span_id"):
            make_trace_event(parent_span_id="")

    def test_rejects_a_span_that_is_its_own_parent(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "parent_span_id"):
            make_trace_event(parent_span_id="span-model-001")

    def test_sequence_is_zero_based_and_attempt_is_positive(self) -> None:
        self.assertEqual(make_trace_event(sequence=0).sequence, 0)

        for invalid in (-1, True, 1.5):
            with self.subTest(sequence=invalid):
                with self.assertRaisesRegex(ContractValidationError, "sequence"):
                    make_trace_event(sequence=invalid)
        for invalid in (0, -1, True, 1.5):
            with self.subTest(attempt=invalid):
                with self.assertRaisesRegex(ContractValidationError, "attempt"):
                    make_trace_event(attempt=invalid)

    def test_timestamp_must_be_a_valid_utc_instant(self) -> None:
        for invalid in (
            "2026-08-14T00:00:00",
            "2026-08-14T08:00:00+08:00",
            "not-a-timestamp",
            "",
        ):
            with self.subTest(timestamp=invalid):
                with self.assertRaisesRegex(ContractValidationError, "timestamp"):
                    make_trace_event(timestamp=invalid)

        self.assertEqual(
            make_trace_event(timestamp="2026-08-14T00:00:00+00:00").timestamp,
            "2026-08-14T00:00:00+00:00",
        )

    def test_direct_constructor_requires_enum_members(self) -> None:
        for field_name, value in (
            ("event_type", "MODEL_REQUEST"),
            ("component", "MODEL"),
            ("status", "STARTED"),
        ):
            with self.subTest(field=field_name):
                with self.assertRaisesRegex(ContractValidationError, field_name):
                    make_trace_event(**{field_name: value})

    def test_from_dict_restores_enums_and_rejects_invalid_values(self) -> None:
        restored = TraceEvent.from_dict(self._valid_payload())

        self.assertIs(restored.event_type, EventType.MODEL_REQUEST)
        self.assertIs(restored.component, EventComponent.MODEL)
        self.assertIs(restored.status, EventStatus.STARTED)

        payload = self._valid_payload()
        payload["status"] = "NOT_A_STATUS"
        with self.assertRaises(ContractValidationError):
            TraceEvent.from_dict(payload)

    def test_attributes_and_artifact_refs_are_strict(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "attributes"):
            make_trace_event(attributes=["not", "an", "object"])
        with self.assertRaisesRegex(ContractValidationError, "non-finite"):
            make_trace_event(attributes={"latency_ms": float("nan")})
        with self.assertRaisesRegex(ContractValidationError, "artifact_refs"):
            make_trace_event(artifact_refs=("artifact-001", ""))
        with self.assertRaisesRegex(ContractValidationError, "duplicate"):
            make_trace_event(artifact_refs=("artifact-001", "artifact-001"))

    def test_rejects_an_unknown_schema_version(self) -> None:
        payload = self._valid_payload()
        payload["schema_version"] = "trace-event/v999"

        with self.assertRaisesRegex(ContractValidationError, "schema_version"):
            TraceEvent.from_dict(payload)

    @staticmethod
    def _valid_payload() -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "event_id": "evt-001",
            "run_id": "run-001",
            "episode_id": "episode-001",
            "trace_id": "trace-001",
            "span_id": "span-model-001",
            "parent_span_id": "span-root-001",
            "sequence": 1,
            "timestamp": "2026-08-14T00:00:01Z",
            "event_type": "MODEL_REQUEST",
            "component": "MODEL",
            "status": "STARTED",
            "attempt": 1,
            "attributes": {"model": "fixture-model"},
            "artifact_refs": ["artifact-request-001"],
        }


if __name__ == "__main__":
    unittest.main()
