import json
import tempfile
import unittest
from pathlib import Path

from scripts.package_polar_fixture import (
    PackageError,
    PackageMetadata,
    load_redactions,
    package_fixture,
)
from scripts.verify_polar_fixture import FAIL, verify_fixture


SCHEMA_PATH = Path(__file__).parent / "fixtures/polar/source-manifest.schema.json"


def write_source(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "request.json": {"messages": [{"role": "user", "content": "17 * 23"}]},
        "response.json": {"answer": "391"},
        "summary.json": {
            "status": "COMPLETED",
            "trajectory": {
                "metadata": {
                    "evaluation": {
                        "outcome_reward": 1.0,
                        "report": {
                            "resolved": True,
                            "empty_generation": False,
                            "failed_apply_patch": False,
                            "error_eval": False,
                            "test_timeout": False,
                        },
                    }
                }
            },
        },
    }
    for filename, payload in payloads.items():
        (root / filename).write_text(json.dumps(payload), encoding="utf-8")


def write_coding_source(root: Path, fixture_type: str, *, synthetic_fault: bool = False) -> None:
    write_source(root)
    patch_path = root / "patch.diff"
    patch_path.write_text("diff --git a/example.py b/example.py\n", encoding="utf-8")
    outcome = {
        "coding_success": "VALID_SUCCESS",
        "coding_valid_failure": "VALID_FAILURE",
        "coding_invalid_infra": "INVALID_INFRASTRUCTURE",
    }[fixture_type]
    if fixture_type == "coding_success":
        result = {
            "verifier_completed": True,
            "timed_out": False,
            "crashed": False,
            "exit_code": 0,
            "resolved": True,
            "reward": 1.0,
            "error_type": None,
        }
    elif fixture_type == "coding_valid_failure":
        result = {
            "verifier_completed": True,
            "timed_out": False,
            "crashed": False,
            "exit_code": 1,
            "resolved": False,
            "reward": 0.0,
            "error_type": None,
        }
    else:
        result = {
            "verifier_completed": False,
            "timed_out": True,
            "crashed": False,
            "exit_code": None,
            "resolved": None,
            "reward": None,
            "error_type": "verifier_timeout",
        }
    (root / "verifier-evidence.json").write_text(
        json.dumps(
            {
                "schema_version": "coding-verifier-evidence/v1",
                "outcome_class": outcome,
                "task_id": "swebench-qwen-code-example",
                "instance_id": "example__repo-1",
                "base_commit": "d" * 40,
                "runtime_image_identity": "sha256:" + "c" * 64,
                "synthetic_fault": synthetic_fault,
                "test": {"command": "pytest -q", "timeout_seconds": 1200},
                "result": result,
                "provenance": {
                    "source_file": "summary.json",
                    "json_path": "$.trajectory.traces[-1].evaluator",
                },
                "patch_sha256": __import__("hashlib").sha256(patch_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    if fixture_type in {"coding_success", "coding_valid_failure"}:
        (root / "replay.json").write_text(
            json.dumps(
                {
                    "schema_version": "coding-clean-replay/v1",
                    "clean_workspace": True,
                    "patch_applied": True,
                    "verifier_completed": True,
                    "resolved": fixture_type == "coding_success",
                }
            ),
            encoding="utf-8",
        )


def valid_metadata(fixture_type: str = "calculator_success") -> PackageMetadata:
    return PackageMetadata(
        fixture_id=f"polar-{fixture_type}-001",
        fixture_type=fixture_type,
        polar_commit="a" * 40,
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        model_revision="b" * 40,
        tokenizer_revision="b" * 40,
        runtime_image_identity="sha256:" + "c" * 64,
        harness="polar-calculator-built-in",
        created_at_utc="2026-08-09T12:00:00Z",
        known_missing_fields=("policy_version", "old_logprobs"),
    )


class PackagePolarFixtureTest(unittest.TestCase):
    def test_packages_and_verifies_success_fixture_without_mutating_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "fixture"
            write_source(source)
            original_response = (source / "response.json").read_bytes()
            output.mkdir()
            (output / "README.md").write_text("fixture docs", encoding="utf-8")

            manifest = package_fixture(
                source,
                output,
                SCHEMA_PATH,
                valid_metadata(),
            )
            results = verify_fixture(output, SCHEMA_PATH)

            self.assertFalse(any(result.status == FAIL for result in results))
            self.assertEqual((output / "response.json").read_bytes(), original_response)
            self.assertEqual((source / "response.json").read_bytes(), original_response)
            self.assertEqual((output / "README.md").read_text(), "fixture docs")
            self.assertEqual(manifest["known_missing_fields"], ["old_logprobs", "policy_version"])
            self.assertNotIn("old_logprobs", json.loads((output / "response.json").read_text()))

    def test_packages_fault_with_synthetic_fault_true(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "fixture"
            write_source(source)
            (source / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "ERROR",
                        "error": "runtime initialization failed: injected failure",
                    }
                ),
                encoding="utf-8",
            )

            manifest = package_fixture(
                source,
                output,
                SCHEMA_PATH,
                valid_metadata("calculator_fault"),
            )

        self.assertTrue(manifest["synthetic_fault"])

    def test_accepts_completed_calculator_path_with_reward_zero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "fixture"
            write_source(source)
            (source / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETED",
                        "trajectory": {
                            "metadata": {
                                "evaluation": {
                                    "outcome_reward": 0.0,
                                    "report": {"resolved": False, "empty_generation": True},
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            manifest = package_fixture(source, output, SCHEMA_PATH, valid_metadata())

            self.assertEqual(manifest["fixture_type"], "calculator_success")
            self.assertTrue((output / "summary.json").is_file())

    def test_packages_coding_success_with_verifier_and_replay_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "fixture"
            write_coding_source(source, "coding_success")

            manifest = package_fixture(
                source,
                output,
                SCHEMA_PATH,
                valid_metadata("coding_success"),
            )

        roles = {entry["path"]: entry["role"] for entry in manifest["files"]}
        self.assertEqual(roles["patch.diff"], "patch")
        self.assertEqual(roles["verifier-evidence.json"], "verifier_evidence")
        self.assertEqual(roles["replay.json"], "replay_evidence")

    def test_packages_explicit_synthetic_coding_infrastructure_fault(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "fixture"
            write_coding_source(source, "coding_invalid_infra", synthetic_fault=True)
            metadata = valid_metadata("coding_invalid_infra")
            metadata = PackageMetadata(**{**metadata.__dict__, "synthetic_fault": True})

            manifest = package_fixture(source, output, SCHEMA_PATH, metadata)

        self.assertTrue(manifest["synthetic_fault"])

    def test_packages_allowlisted_raw_and_normalized_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "fixture"
            write_source(source)
            (source / "gateway.log").write_text("gateway ok", encoding="utf-8")
            normalized = source / "normalized-logs"
            normalized.mkdir()
            (normalized / "evaluator.log").write_text("reward=1", encoding="utf-8")

            manifest = package_fixture(
                source,
                output,
                SCHEMA_PATH,
                valid_metadata(),
            )

            roles = {entry["path"]: entry["role"] for entry in manifest["files"]}
            self.assertEqual(roles["gateway.log"], "raw_log")
            self.assertEqual(roles["normalized-logs/evaluator.log"], "normalized_log")
            self.assertTrue((output / "normalized-logs/evaluator.log").is_file())

    def test_rejects_missing_required_source_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            write_source(source)
            (source / "summary.json").unlink()

            with self.assertRaisesRegex(PackageError, "missing required"):
                package_fixture(
                    source,
                    root / "fixture",
                    SCHEMA_PATH,
                    valid_metadata(),
                )

    def test_rejects_invalid_source_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            write_source(source)
            (source / "response.json").write_text("{broken", encoding="utf-8")

            with self.assertRaisesRegex(PackageError, "invalid UTF-8 JSON"):
                package_fixture(
                    source,
                    root / "fixture",
                    SCHEMA_PATH,
                    valid_metadata(),
                )

    def test_rejects_unclassified_staging_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            write_source(source)
            (source / "mystery.bin").write_bytes(b"unknown")

            with self.assertRaisesRegex(PackageError, "unexpected staging files"):
                package_fixture(
                    source,
                    root / "fixture",
                    SCHEMA_PATH,
                    valid_metadata(),
                )

    def test_refuses_to_overwrite_existing_fixture(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "fixture"
            write_source(source)
            output.mkdir()
            (output / "response.json").write_text("existing", encoding="utf-8")

            with self.assertRaisesRegex(PackageError, "refusing to overwrite"):
                package_fixture(
                    source,
                    output,
                    SCHEMA_PATH,
                    valid_metadata(),
                )

            self.assertEqual((output / "response.json").read_text(), "existing")

    def test_rejects_output_inside_source_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source"
            write_source(source)

            with self.assertRaisesRegex(PackageError, "must not be inside"):
                package_fixture(
                    source,
                    source / "fixture",
                    SCHEMA_PATH,
                    valid_metadata(),
                )

    def test_rejects_file_over_repository_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            write_source(source)

            with self.assertRaisesRegex(PackageError, "exceed repository limit"):
                package_fixture(
                    source,
                    root / "fixture",
                    SCHEMA_PATH,
                    valid_metadata(),
                    max_file_bytes=10,
                )

    def test_rejects_obvious_secret_before_writing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "fixture"
            write_source(source)
            (source / "request.json").write_text(
                json.dumps({"headers": {"authorization": "Bearer secret"}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(PackageError, "obvious_sensitive_fields"):
                package_fixture(
                    source,
                    output,
                    SCHEMA_PATH,
                    valid_metadata(),
                )

            self.assertFalse(output.exists())

    def test_rejects_invalid_metadata_before_writing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            output = root / "fixture"
            write_source(source)
            metadata = valid_metadata()
            invalid = PackageMetadata(
                **{**metadata.__dict__, "polar_commit": "abc1234"}
            )

            with self.assertRaisesRegex(PackageError, "manifest_contract"):
                package_fixture(source, output, SCHEMA_PATH, invalid)

            self.assertFalse(output.exists())

    def test_loads_redaction_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "redactions.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "path": "request.json:$.headers.authorization",
                            "rule": "replace_with_redacted_marker",
                            "count": 1,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            records = load_redactions(path)

        self.assertEqual(records[0]["count"], 1)


if __name__ == "__main__":
    unittest.main()
