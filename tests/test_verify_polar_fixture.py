import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_polar_fixture import FAIL, PASS, verify_fixture


SCHEMA_PATH = Path(__file__).parent / "fixtures/polar/source-manifest.schema.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FixtureBuilder:
    def __init__(self, root: Path, fixture_type: str = "calculator_success"):
        self.root = root
        self.manifest = {
            "schema_version": "polar-fixture-manifest/v1",
            "fixture_id": f"polar-{fixture_type}-001",
            "fixture_type": fixture_type,
            "synthetic_fault": fixture_type == "calculator_fault",
            "created_at_utc": "2026-08-09T12:00:00Z",
            "source": {
                "polar_commit": "a" * 40,
                "model_id": "Qwen/Qwen3-4B-Instruct-2507",
                "model_revision": "b" * 40,
                "tokenizer_revision": "b" * 40,
                "runtime_image_identity": "sha256:" + "c" * 64,
                "harness": "polar-calculator-built-in",
                "policy_version": None,
            },
            "files": [],
            "redactions": [],
            "known_missing_fields": ["old_logprobs", "policy_version"],
            "notes": [],
        }
        self.write_json("request.json", {"messages": [{"role": "user", "content": "17 * 23"}]}, "request")
        self.write_json("response.json", {"answer": "391"}, "response")
        if fixture_type == "calculator_fault":
            summary = {
                "status": "ERROR",
                "error": "runtime initialization failed: injected failure",
            }
        else:
            summary = {
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
            }
        self.write_json("summary.json", summary, "summary")
        if fixture_type.startswith("coding_"):
            self._add_coding_evidence(fixture_type)

    def write_text(self, relative_path: str, payload: str, role: str, media_type: str) -> None:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        self.manifest["files"].append(
            {
                "path": relative_path,
                "role": role,
                "media_type": media_type,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    def _add_coding_evidence(self, fixture_type: str) -> None:
        patch = "diff --git a/example.py b/example.py\n"
        self.write_text("patch.diff", patch, "patch", "text/x-diff")
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
        self.write_json(
            "verifier-evidence.json",
            {
                "schema_version": "coding-verifier-evidence/v1",
                "outcome_class": outcome,
                "task_id": "swebench-qwen-code-example",
                "instance_id": "example__repo-1",
                "base_commit": "d" * 40,
                "runtime_image_identity": "sha256:" + "c" * 64,
                "synthetic_fault": False,
                "test": {"command": "pytest -q", "timeout_seconds": 1200},
                "result": result,
                "provenance": {
                    "source_file": "summary.json",
                    "json_path": "$.trajectory.traces[-1].evaluator",
                },
                "patch_sha256": sha256_file(self.root / "patch.diff"),
            },
            "verifier_evidence",
        )
        if fixture_type in {"coding_success", "coding_valid_failure"}:
            self.write_json(
                "replay.json",
                {
                    "schema_version": "coding-clean-replay/v1",
                    "clean_workspace": True,
                    "patch_applied": True,
                    "verifier_completed": True,
                    "resolved": fixture_type == "coding_success",
                },
                "replay_evidence",
            )

    def write_json(self, relative_path: str, payload, role: str) -> None:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        existing = next(
            (entry for entry in self.manifest["files"] if entry["path"] == relative_path),
            None,
        )
        entry = {
            "path": relative_path,
            "role": role,
            "media_type": "application/json",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if existing is None:
            self.manifest["files"].append(entry)
        else:
            existing.update(entry)

    def refresh_file(self, relative_path: str) -> None:
        path = self.root / relative_path
        entry = next(item for item in self.manifest["files"] if item["path"] == relative_path)
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = sha256_file(path)

    def save_manifest(self) -> None:
        (self.root / "source-manifest.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def result_for(results, name: str):
    return next(result for result in results if result.name == name)


class VerifyPolarFixtureTest(unittest.TestCase):
    def verify(self, builder: FixtureBuilder, **kwargs):
        builder.save_manifest()
        return verify_fixture(builder.root, SCHEMA_PATH, **kwargs)

    def test_accepts_valid_success_fixture(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results = self.verify(FixtureBuilder(Path(temp_dir)))

        self.assertTrue(results)
        self.assertTrue(all(result.status == PASS for result in results))

    def test_accepts_valid_synthetic_fault_fixture(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results = self.verify(
                FixtureBuilder(Path(temp_dir), fixture_type="calculator_fault")
            )

        self.assertEqual(result_for(results, "manifest_contract").status, PASS)

    def test_accepts_completed_calculator_path_with_valid_task_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = FixtureBuilder(Path(temp_dir), fixture_type="calculator_success")
            builder.write_json(
                "summary.json",
                {
                    "status": "COMPLETED",
                    "trajectory": {
                        "metadata": {
                            "evaluation": {
                                "outcome_reward": 0.0,
                                "report": {
                                    "resolved": False,
                                    "empty_generation": True,
                                    "error_eval": False,
                                    "test_timeout": False,
                                },
                            }
                        }
                    },
                },
                "summary",
            )
            results = self.verify(builder)

        semantic = result_for(results, "calculator_evidence")
        self.assertEqual(semantic.status, PASS)

    def test_rejects_calculator_success_with_evaluator_timeout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = FixtureBuilder(Path(temp_dir), fixture_type="calculator_success")
            builder.write_json(
                "summary.json",
                {
                    "status": "COMPLETED",
                    "trajectory": {
                        "metadata": {
                            "evaluation": {
                                "outcome_reward": None,
                                "report": {
                                    "resolved": False,
                                    "empty_generation": False,
                                    "error_eval": False,
                                    "test_timeout": True,
                                },
                            }
                        }
                    },
                },
                "summary",
            )
            results = self.verify(builder)

        self.assertEqual(result_for(results, "calculator_evidence").status, FAIL)

    def test_accepts_all_three_coding_outcomes(self):
        for fixture_type in (
            "coding_success",
            "coding_valid_failure",
            "coding_invalid_infra",
        ):
            with self.subTest(fixture_type=fixture_type), tempfile.TemporaryDirectory() as temp_dir:
                results = self.verify(FixtureBuilder(Path(temp_dir), fixture_type))
                self.assertTrue(all(result.status == PASS for result in results))

    def test_rejects_coding_outcome_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = FixtureBuilder(Path(temp_dir), "coding_success")
            evidence_path = builder.root / "verifier-evidence.json"
            evidence = json.loads(evidence_path.read_text())
            evidence["outcome_class"] = "VALID_FAILURE"
            builder.write_json("verifier-evidence.json", evidence, "verifier_evidence")
            results = self.verify(builder)

        self.assertEqual(result_for(results, "coding_evidence").status, FAIL)

    def test_rejects_infrastructure_failure_mapped_to_reward_zero(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = FixtureBuilder(Path(temp_dir), "coding_invalid_infra")
            evidence_path = builder.root / "verifier-evidence.json"
            evidence = json.loads(evidence_path.read_text())
            evidence["result"]["reward"] = 0.0
            builder.write_json("verifier-evidence.json", evidence, "verifier_evidence")
            results = self.verify(builder)

        evidence_result = result_for(results, "coding_evidence")
        self.assertEqual(evidence_result.status, FAIL)
        self.assertIn("reward null", evidence_result.message)

    def test_rejects_patch_checksum_or_replay_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = FixtureBuilder(Path(temp_dir), "coding_success")
            (builder.root / "patch.diff").write_text("changed", encoding="utf-8")
            builder.refresh_file("patch.diff")
            replay_path = builder.root / "replay.json"
            replay = json.loads(replay_path.read_text())
            replay["resolved"] = False
            builder.write_json("replay.json", replay, "replay_evidence")
            results = self.verify(builder)

        evidence_result = result_for(results, "coding_evidence")
        self.assertEqual(evidence_result.status, FAIL)
        self.assertIn("patch_sha256", evidence_result.message)

    def test_rejects_missing_required_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = FixtureBuilder(Path(temp_dir))
            (builder.root / "summary.json").unlink()
            results = self.verify(builder)

        self.assertEqual(result_for(results, "required_json_files").status, FAIL)
        self.assertEqual(result_for(results, "file_integrity").status, FAIL)

    def test_rejects_invalid_required_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = FixtureBuilder(Path(temp_dir))
            (builder.root / "response.json").write_text("{broken", encoding="utf-8")
            builder.refresh_file("response.json")
            results = self.verify(builder)

        self.assertEqual(result_for(results, "required_json_files").status, FAIL)

    def test_rejects_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = FixtureBuilder(Path(temp_dir))
            builder.save_manifest()
            (builder.root / "response.json").write_text('{"answer":"changed"}', encoding="utf-8")
            results = verify_fixture(builder.root, SCHEMA_PATH)

        integrity = result_for(results, "file_integrity")
        self.assertEqual(integrity.status, FAIL)
        self.assertIn("checksum mismatch", integrity.message)

    def test_rejects_unsafe_manifest_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = FixtureBuilder(Path(temp_dir))
            builder.manifest["files"].append(
                {
                    "path": "../outside.log",
                    "role": "raw_log",
                    "media_type": "text/plain",
                    "bytes": 0,
                    "sha256": "d" * 64,
                }
            )
            results = self.verify(builder)

        self.assertEqual(result_for(results, "manifest_contract").status, FAIL)

    def test_rejects_duplicate_manifest_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = FixtureBuilder(Path(temp_dir))
            builder.manifest["files"].append(dict(builder.manifest["files"][0]))
            results = self.verify(builder)

        contract = result_for(results, "manifest_contract")
        self.assertEqual(contract.status, FAIL)
        self.assertIn("duplicate", contract.message)

    def test_rejects_fixture_type_fault_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = FixtureBuilder(Path(temp_dir), fixture_type="calculator_fault")
            builder.manifest["synthetic_fault"] = False
            results = self.verify(builder)

        self.assertEqual(result_for(results, "manifest_contract").status, FAIL)

    def test_rejects_short_polar_commit_and_placeholder_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = FixtureBuilder(Path(temp_dir))
            builder.manifest["source"]["polar_commit"] = "abc1234"
            builder.manifest["source"]["model_revision"] = "TBD"
            results = self.verify(builder)

        contract = result_for(results, "manifest_contract")
        self.assertEqual(contract.status, FAIL)
        self.assertIn("polar_commit", contract.message)

    def test_rejects_unlisted_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = FixtureBuilder(Path(temp_dir))
            (builder.root / "forgotten.log").write_text("not listed", encoding="utf-8")
            results = self.verify(builder)

        self.assertEqual(result_for(results, "unlisted_files").status, FAIL)

    def test_rejects_file_over_repository_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = FixtureBuilder(Path(temp_dir))
            results = self.verify(builder, max_file_bytes=10)

        self.assertEqual(result_for(results, "repository_size").status, FAIL)

    def test_rejects_obvious_unredacted_secret(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = FixtureBuilder(Path(temp_dir))
            builder.write_json(
                "request.json",
                {"headers": {"authorization": "Bearer example-secret"}},
                "request",
            )
            results = self.verify(builder)

        sensitive = result_for(results, "obvious_sensitive_fields")
        self.assertEqual(sensitive.status, FAIL)
        self.assertIn("authorization", sensitive.message)

    def test_allows_explicitly_redacted_sensitive_value(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = FixtureBuilder(Path(temp_dir))
            builder.write_json(
                "request.json",
                {"headers": {"authorization": "<redacted>"}},
                "request",
            )
            results = self.verify(builder)

        self.assertEqual(result_for(results, "obvious_sensitive_fields").status, PASS)

    def test_rejects_secret_in_optional_coding_evidence_or_patch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            builder = FixtureBuilder(Path(temp_dir), "coding_invalid_infra")
            builder.write_json(
                "fault-injection.json",
                {"headers": {"authorization": "Bearer secret"}},
                "fault_injection",
            )
            results = self.verify(builder)

        self.assertEqual(result_for(results, "obvious_sensitive_fields").status, FAIL)

        with tempfile.TemporaryDirectory() as temp_dir:
            builder = FixtureBuilder(Path(temp_dir), "coding_success")
            (builder.root / "patch.diff").write_text(
                "-----BEGIN PRIVATE KEY-----\nsecret\n", encoding="utf-8"
            )
            builder.refresh_file("patch.diff")
            evidence_path = builder.root / "verifier-evidence.json"
            evidence = json.loads(evidence_path.read_text())
            evidence["patch_sha256"] = sha256_file(builder.root / "patch.diff")
            builder.write_json("verifier-evidence.json", evidence, "verifier_evidence")
            results = self.verify(builder)

        self.assertEqual(result_for(results, "obvious_sensitive_fields").status, FAIL)


if __name__ == "__main__":
    unittest.main()
