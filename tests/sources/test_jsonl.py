from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.contracts import Capability
from src.errors import AdapterConversionError, ErrorCode
from src.sources import JsonlSourceAdapter, SourceAdapter, dumps_jsonl
from tests.contract_fixtures import make_record


class JsonlSourceAdapterTest(unittest.TestCase):
    def test_implements_source_adapter_protocol(self) -> None:
        self.assertIsInstance(JsonlSourceAdapter(), SourceAdapter)

    def test_roundtrip_preserves_semantics_but_replaces_source_envelope(self) -> None:
        original = make_record()
        payload = dumps_jsonl((original,))

        result = JsonlSourceAdapter().convert(payload)

        self.assertTrue(result.ok)
        self.assertEqual(len(result.records), 1)
        restored = result.records[0]
        self.assertEqual(restored.semantic_dict(), original.semantic_dict())
        self.assertEqual(restored.source_type, "jsonl")
        self.assertEqual(restored.source_record_id, "line:1")
        self.assertNotEqual(restored.source_payload_sha256, original.source_payload_sha256)

    def test_dump_is_stable_and_does_not_mutate_source(self) -> None:
        original = make_record(opaque_metadata={"z": [2, 1], "a": {"x": True}})
        before = original.to_dict()

        first = dumps_jsonl((original,))
        second = dumps_jsonl((original,))

        self.assertEqual(first, second)
        self.assertEqual(original.to_dict(), before)
        self.assertTrue(first.endswith("\n"))

    def test_file_conversion_records_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollouts.jsonl"
            path.write_text(dumps_jsonl((make_record(),)), encoding="utf-8")

            result = JsonlSourceAdapter().convert_file(path)

        self.assertTrue(result.ok)
        self.assertEqual(result.records[0].source_payload_ref, f"{path}#line:1")

    def test_reports_parse_errors_without_discarding_valid_lines(self) -> None:
        valid_line = dumps_jsonl((make_record(),)).strip()
        result = JsonlSourceAdapter().convert("{not-json}\n" + valid_line + "\n")

        self.assertFalse(result.ok)
        self.assertEqual(len(result.records), 1)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0].code, ErrorCode.ADAPTER_ERROR)
        self.assertIn("line 1", result.errors[0].message)
        with self.assertRaises(AdapterConversionError):
            result.to_batch(adapter_name="jsonl", adapter_version="v1")

    def test_required_capability_is_an_explicit_error_not_a_default(self) -> None:
        incomplete = make_record(token_ids=None, loss_mask=None, old_logprobs=None)
        result = JsonlSourceAdapter(
            required_capabilities={Capability.TOKEN_IDS, Capability.ACTION_MASK}
        ).convert(dumps_jsonl((incomplete,)))

        self.assertFalse(result.ok)
        self.assertIsNone(result.records[0].token_ids)
        self.assertEqual(result.errors[0].code, ErrorCode.CAPABILITY_MISSING)
        self.assertCountEqual(
            result.errors[0].details["missing"],
            [Capability.TOKEN_IDS.value, Capability.ACTION_MASK.value],
        )

    def test_capabilities_are_only_those_common_to_every_record(self) -> None:
        complete = make_record()
        incomplete = make_record(
            trajectory_id="trajectory-002",
            source_record_id="fixture-record-002",
            old_logprobs=None,
            verifier_evidence_ref=None,
            tool_events=None,
        )

        result = JsonlSourceAdapter().convert(dumps_jsonl((complete, incomplete)))

        self.assertTrue(result.ok)
        self.assertIn(Capability.TOKEN_IDS, result.capabilities)
        self.assertNotIn(Capability.OLD_LOGPROBS, result.capabilities)
        self.assertNotIn(Capability.VERIFIER_EVIDENCE, result.capabilities)

    def test_rejects_unknown_envelope_and_record_fields(self) -> None:
        envelope = json.loads(dumps_jsonl((make_record(),)))
        envelope["gateway"] = {"internal": True}
        result = JsonlSourceAdapter().convert(json.dumps(envelope) + "\n")
        self.assertEqual(result.errors[0].code, ErrorCode.ADAPTER_ERROR)

        envelope = json.loads(dumps_jsonl((make_record(),)))
        envelope["record"]["gateway"] = {"internal": True}
        result = JsonlSourceAdapter().convert(json.dumps(envelope) + "\n")
        self.assertIn("opaque_metadata", result.errors[0].message)
        self.assertEqual(result.errors[0].code, ErrorCode.CONTRACT_INVALID)

    def test_empty_payload_is_a_warning(self) -> None:
        result = JsonlSourceAdapter().convert("\n")

        self.assertTrue(result.ok)
        self.assertFalse(result.records)
        self.assertEqual(result.warnings[0].code, ErrorCode.SOURCE_WARNING)

    def test_non_text_payload_is_rejected(self) -> None:
        result = JsonlSourceAdapter().convert({"record": "not-jsonl"})

        self.assertFalse(result.ok)
        self.assertEqual(result.errors[0].code, ErrorCode.ADAPTER_ERROR)


if __name__ == "__main__":
    unittest.main()
