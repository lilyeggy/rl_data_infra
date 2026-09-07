from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from src.contracts import (
    Capability,
    ComponentStatus,
    RolloutStatus,
    VerifierStatus,
)
from src.errors import ErrorCode
from src.sources import JsonlSourceAdapter, PolarFixtureImporter, SourceAdapter, dumps_jsonl


REAL_SUCCESS = Path("tests/fixtures/polar/calculator_success")
REAL_FAULT = Path("tests/fixtures/polar/calculator_fault")


def write_fixture(root: Path, *, reward: float = 1.0, failed_runtime: bool = False) -> None:
    root.mkdir(parents=True)
    if failed_runtime:
        summary = {
            "session_id": "session-fault",
            "task_id": "task-fault",
            "status": "ERROR",
            "trajectory": {"metadata": {}, "traces": []},
            "timing": {"run_ms": 0.0},
            "error": "runtime initialization failed: injected",
        }
        fixture_type = "calculator_fault"
    else:
        summary = {
            "session_id": "session-success",
            "task_id": "task-success",
            "status": "COMPLETED",
            "trajectory": {
                "metadata": {
                    "evaluation": {
                        "outcome_reward": reward,
                        "report": {
                            "resolved": reward == 1,
                            "empty_generation": False,
                            "error_eval": False,
                            "test_timeout": False,
                        },
                    }
                },
                "traces": [
                    {
                        "prompt_ids": [10, 11],
                        "response_ids": [20, 21, 22],
                        "loss_mask": [1, 0, 1],
                        "response_logprobs": [-0.1, 0.0, -0.2],
                        "reward": reward,
                        "finish_reason": "stop",
                        "response_messages": [
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {"id": "call-1", "function": {"name": "read_file"}}
                                ],
                            }
                        ],
                    }
                ],
            },
        }
        fixture_type = "calculator_success"
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    summary_bytes = summary_path.read_bytes()
    manifest = {
        "schema_version": "polar-fixture-manifest/v1",
        "fixture_id": f"fixture-{fixture_type}",
        "fixture_type": fixture_type,
        "synthetic_fault": failed_runtime,
        "source": {
            "polar_commit": "a" * 40,
            "model_id": "Qwen/Qwen3-4B-Instruct-2507",
            "model_revision": "b" * 40,
            "tokenizer_revision": "b" * 40,
            "runtime_image_identity": "sha256:" + "c" * 64,
            "harness": "qwen_code",
            "policy_version": None,
        },
        "files": [
            {
                "path": "summary.json",
                "role": "summary",
                "media_type": "application/json",
                "bytes": len(summary_bytes),
                "sha256": hashlib.sha256(summary_bytes).hexdigest(),
            }
        ],
    }
    (root / "source-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class PolarFixtureImporterTest(unittest.TestCase):
    def test_implements_protocol_without_importing_upstream_polar(self) -> None:
        self.assertIsInstance(PolarFixtureImporter(), SourceAdapter)
        self.assertNotIn("polar", sys.modules)

    def test_concatenates_native_tokens_and_preserves_response_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir) / "fixture"
            write_fixture(fixture)

            result = PolarFixtureImporter().convert(fixture)

        self.assertTrue(result.ok)
        record = result.records[0]
        self.assertEqual(record.token_ids, (10, 11, 20, 21, 22))
        self.assertEqual(record.prompt_token_count, 2)
        self.assertEqual(record.loss_mask, (0, 0, 1, 0, 1))
        self.assertEqual(record.old_logprobs, (-0.1, 0.0, -0.2))
        self.assertEqual(record.reward, 1.0)
        self.assertEqual(record.verifier_status, VerifierStatus.PASSED)
        self.assertIn(Capability.OLD_LOGPROBS, result.capabilities)
        self.assertNotIn(Capability.POLICY_VERSION, result.capabilities)
        self.assertEqual(record.tool_events[0]["id"], "call-1")  # type: ignore[index]

    def test_represents_pre_run_infrastructure_failure_without_fake_reward(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir) / "fixture"
            write_fixture(fixture, failed_runtime=True)

            result = PolarFixtureImporter().convert(fixture)

        self.assertTrue(result.ok)
        record = result.records[0]
        self.assertEqual(record.rollout_status, RolloutStatus.FAILED)
        self.assertEqual(record.runtime_status, ComponentStatus.FAILED)
        self.assertEqual(record.verifier_status, VerifierStatus.NOT_RUN)
        self.assertIsNone(record.reward)
        self.assertIsNone(record.token_ids)
        self.assertNotIn(Capability.REWARD, result.capabilities)

    def test_reports_execution_success_separately_from_task_reward(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir) / "fixture"
            write_fixture(fixture, reward=0.0)

            result = PolarFixtureImporter().convert(fixture)

        self.assertTrue(result.ok)
        self.assertEqual(result.records[0].reward, 0.0)
        self.assertEqual(result.warnings[0].code, ErrorCode.SOURCE_WARNING)
        self.assertIn("task-level success", result.warnings[0].message)

    def test_required_capabilities_fail_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir) / "fixture"
            write_fixture(fixture)

            result = PolarFixtureImporter(
                required_capabilities={Capability.POLICY_VERSION, Capability.GROUP_ID}
            ).convert(fixture)

        self.assertFalse(result.ok)
        self.assertEqual(result.errors[0].code, ErrorCode.CAPABILITY_MISSING)
        self.assertCountEqual(
            result.errors[0].details["missing"],
            [Capability.POLICY_VERSION.value, Capability.GROUP_ID.value],
        )

    def test_rejects_summary_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir) / "fixture"
            write_fixture(fixture)
            (fixture / "summary.json").write_text("{}", encoding="utf-8")

            result = PolarFixtureImporter().convert(fixture)

        self.assertFalse(result.ok)
        self.assertEqual(result.errors[0].code, ErrorCode.CONTRACT_INVALID)

    def test_polar_to_jsonl_roundtrip_preserves_record_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir) / "fixture"
            write_fixture(fixture)
            polar_result = PolarFixtureImporter().convert(fixture)

            jsonl_result = JsonlSourceAdapter().convert(dumps_jsonl(polar_result.records))

        self.assertTrue(jsonl_result.ok)
        self.assertEqual(
            jsonl_result.records[0].semantic_dict(),
            polar_result.records[0].semantic_dict(),
        )

    @unittest.skipUnless(
        (REAL_SUCCESS / "source-manifest.json").is_file(),
        "real Day 2 fixture is not present in this checkout",
    )
    def test_real_day2_fixture_maps_native_token_arrays(self) -> None:
        result = PolarFixtureImporter().convert(REAL_SUCCESS)

        self.assertTrue(result.ok)
        record = result.records[0]
        self.assertEqual(record.prompt_token_count, 14846)
        self.assertEqual(len(record.token_ids or ()), 14945)
        self.assertEqual(len(record.loss_mask or ()), 14945)
        self.assertEqual(len(record.old_logprobs or ()), 99)
        self.assertEqual(record.reward, 0.0)
        self.assertTrue(result.warnings)

    @unittest.skipUnless(
        (REAL_FAULT / "source-manifest.json").is_file(),
        "real Day 2 fixture is not present in this checkout",
    )
    def test_real_day2_fault_remains_untrainable(self) -> None:
        result = PolarFixtureImporter().convert(REAL_FAULT)

        self.assertTrue(result.ok)
        record = result.records[0]
        self.assertEqual(record.runtime_status, ComponentStatus.FAILED)
        self.assertIsNone(record.reward)
        self.assertFalse(result.capabilities)


if __name__ == "__main__":
    unittest.main()
