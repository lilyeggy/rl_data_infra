#!/usr/bin/env python3
"""Verify the integrity and provenance metadata of a Polar golden fixture.

This verifier is deliberately read-only.  It checks that a committed fixture
matches its manifest; it does not judge the model answer or prove that training
fields such as token IDs came from the model backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

MANIFEST_FILE = "source-manifest.json"
MANIFEST_SCHEMA_VERSION = "polar-fixture-manifest/v1"
DEFAULT_MAX_FILE_BYTES = 2 * 1024 * 1024
BASE_REQUIRED_JSON_FILES = {
    "request.json": "request",
    "response.json": "response",
    "summary.json": "summary",
}
CODING_REQUIRED_FILES = {
    "verifier-evidence.json": "verifier_evidence",
    "patch.diff": "patch",
}
CODING_REPLAY_FILES = {"replay.json": "replay_evidence"}
FIXTURE_TYPES = {
    "calculator_success",
    "calculator_fault",
    "coding_success",
    "coding_valid_failure",
    "coding_invalid_infra",
}
FILE_ROLES = {
    "request",
    "response",
    "summary",
    "verifier_evidence",
    "patch",
    "replay_evidence",
    "fault_injection",
    "task_metadata",
    "raw_log",
    "normalized_log",
    "external_reference",
}
ROOT_FIELDS = {
    "schema_version",
    "fixture_id",
    "fixture_type",
    "synthetic_fault",
    "created_at_utc",
    "source",
    "files",
    "redactions",
    "known_missing_fields",
    "notes",
}
SOURCE_FIELDS = {
    "polar_commit",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "runtime_image_identity",
    "harness",
    "policy_version",
}
REQUIRED_SOURCE_FIELDS = SOURCE_FIELDS - {"policy_version"}
FILE_FIELDS = {"path", "role", "media_type", "bytes", "sha256"}
REDACTION_FIELDS = {"path", "rule", "count"}
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "private_key",
    "proxy_authorization",
    "refresh_token",
    "secret",
    "set_cookie",
}
REDACTED_MARKERS = {"", "<redacted>", "[redacted]", "redacted", "***", "null"}
PLACEHOLDER_VALUES = {"tbd", "todo", "unknown", "placeholder", "changeme"}
FULL_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PRIVATE_KEY_PATTERN = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a captured Polar golden fixture."
    )
    parser.add_argument(
        "fixture_dir",
        type=Path,
        help="Fixture directory containing source-manifest.json.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("tests/fixtures/polar/source-manifest.schema.json"),
        help="Path to the source manifest JSON schema.",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
        help="Maximum size of any file committed inside the fixture.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_placeholder(value: str) -> bool:
    stripped = value.strip()
    lowered = stripped.lower()
    return lowered in PLACEHOLDER_VALUES or (
        stripped.startswith("<") and stripped.endswith(">")
    )


def _is_utc_datetime(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0


def _is_safe_relative_path(value: Any) -> bool:
    if not _is_nonempty_string(value):
        return False
    assert isinstance(value, str)
    if value.startswith(("/", "./")) or "\\" in value or "\x00" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def _format_errors(errors: Iterable[str], limit: int = 6) -> str:
    error_list = list(errors)
    shown = error_list[:limit]
    suffix = f"; ... and {len(error_list) - limit} more" if len(error_list) > limit else ""
    return "; ".join(shown) + suffix


def _validate_manifest_contract(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    missing_root = sorted(ROOT_FIELDS - {"notes"} - set(manifest))
    unknown_root = sorted(set(manifest) - ROOT_FIELDS)
    if missing_root:
        errors.append(f"missing root fields: {', '.join(missing_root)}")
    if unknown_root:
        errors.append(f"unknown root fields: {', '.join(unknown_root)}")

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append(f"schema_version must be {MANIFEST_SCHEMA_VERSION}")

    fixture_id = manifest.get("fixture_id")
    if not _is_nonempty_string(fixture_id) or (
        isinstance(fixture_id, str) and _is_placeholder(fixture_id)
    ):
        errors.append("fixture_id must be a non-placeholder string")

    fixture_type = manifest.get("fixture_type")
    if fixture_type not in FIXTURE_TYPES:
        errors.append("fixture_type is not a supported Polar fixture type")

    synthetic_fault = manifest.get("synthetic_fault")
    if not isinstance(synthetic_fault, bool):
        errors.append("synthetic_fault must be boolean")
    elif fixture_type == "calculator_success" and synthetic_fault:
        errors.append("calculator_success must set synthetic_fault=false")
    elif fixture_type == "calculator_fault" and not synthetic_fault:
        errors.append("calculator_fault must set synthetic_fault=true")
    elif fixture_type in {"coding_success", "coding_valid_failure"} and synthetic_fault:
        errors.append(f"{fixture_type} must set synthetic_fault=false")

    if not _is_utc_datetime(manifest.get("created_at_utc")):
        errors.append("created_at_utc must be a UTC ISO-8601 timestamp ending in Z")

    source = manifest.get("source")
    if not isinstance(source, dict):
        errors.append("source must be an object")
    else:
        missing_source = sorted(REQUIRED_SOURCE_FIELDS - set(source))
        unknown_source = sorted(set(source) - SOURCE_FIELDS)
        if missing_source:
            errors.append(f"missing source fields: {', '.join(missing_source)}")
        if unknown_source:
            errors.append(f"unknown source fields: {', '.join(unknown_source)}")
        if not FULL_SHA_PATTERN.fullmatch(str(source.get("polar_commit", ""))):
            errors.append("source.polar_commit must be a full 40-character lowercase SHA")
        for field in REQUIRED_SOURCE_FIELDS - {"polar_commit"}:
            value = source.get(field)
            if not _is_nonempty_string(value) or (
                isinstance(value, str) and _is_placeholder(value)
            ):
                errors.append(f"source.{field} must be a non-placeholder string")
        policy_version = source.get("policy_version")
        if policy_version is not None and not _is_nonempty_string(policy_version):
            errors.append("source.policy_version must be a string or null")

    files = manifest.get("files")
    if not isinstance(files, list) or len(files) < 3:
        errors.append("files must contain at least three entries")
    else:
        seen_paths: set[str] = set()
        for index, entry in enumerate(files):
            prefix = f"files[{index}]"
            if not isinstance(entry, dict):
                errors.append(f"{prefix} must be an object")
                continue
            missing_fields = FILE_FIELDS - set(entry)
            unknown_fields = set(entry) - FILE_FIELDS
            if missing_fields:
                errors.append(f"{prefix} missing: {', '.join(sorted(missing_fields))}")
            if unknown_fields:
                errors.append(f"{prefix} unknown: {', '.join(sorted(unknown_fields))}")
            path_value = entry.get("path")
            if not _is_safe_relative_path(path_value):
                errors.append(f"{prefix}.path is not a safe relative path")
            elif path_value in seen_paths:
                errors.append(f"duplicate manifest path: {path_value}")
            else:
                seen_paths.add(path_value)
            if entry.get("role") not in FILE_ROLES:
                errors.append(f"{prefix}.role is invalid")
            if not _is_nonempty_string(entry.get("media_type")):
                errors.append(f"{prefix}.media_type must be a non-empty string")
            byte_count = entry.get("bytes")
            if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
                errors.append(f"{prefix}.bytes must be a non-negative integer")
            if not SHA256_PATTERN.fullmatch(str(entry.get("sha256", ""))):
                errors.append(f"{prefix}.sha256 must be 64 lowercase hex characters")

    redactions = manifest.get("redactions")
    if not isinstance(redactions, list):
        errors.append("redactions must be an array")
    else:
        for index, entry in enumerate(redactions):
            if not isinstance(entry, dict) or set(entry) != REDACTION_FIELDS:
                errors.append(f"redactions[{index}] must contain path, rule, and count")
                continue
            count = entry.get("count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                errors.append(f"redactions[{index}].count must be a non-negative integer")

    missing_fields = manifest.get("known_missing_fields")
    if not isinstance(missing_fields, list) or not all(
        _is_nonempty_string(item) for item in missing_fields
    ):
        errors.append("known_missing_fields must be an array of non-empty strings")
    elif len(missing_fields) != len(set(missing_fields)):
        errors.append("known_missing_fields must not contain duplicates")

    notes = manifest.get("notes", [])
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        errors.append("notes must be an array of strings")

    return errors


def _sensitive_locations(payload: Any, location: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_location = f"{location}.{key}"
            normalized_key = str(key).lower().replace("-", "_")
            if normalized_key in SENSITIVE_KEYS:
                rendered = "null" if value is None else str(value).strip().lower()
                if rendered not in REDACTED_MARKERS:
                    findings.append(child_location)
            findings.extend(_sensitive_locations(value, child_location))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            findings.extend(_sensitive_locations(value, f"{location}[{index}]"))
    elif isinstance(payload, str) and PRIVATE_KEY_PATTERN.search(payload):
        findings.append(location)
    return findings


def _required_files_for_type(fixture_type: Any) -> dict[str, str]:
    required = dict(BASE_REQUIRED_JSON_FILES)
    if fixture_type in {
        "coding_success",
        "coding_valid_failure",
        "coding_invalid_infra",
    }:
        required.update(CODING_REQUIRED_FILES)
    if fixture_type in {"coding_success", "coding_valid_failure"}:
        required.update(CODING_REPLAY_FILES)
    return required


def _validate_coding_evidence(
    fixture_dir: Path,
    manifest: dict[str, Any],
    payloads: dict[str, Any],
) -> list[str]:
    fixture_type = manifest.get("fixture_type")
    if fixture_type not in {
        "coding_success",
        "coding_valid_failure",
        "coding_invalid_infra",
    }:
        return []

    errors: list[str] = []
    evidence = payloads.get("verifier-evidence.json")
    if not isinstance(evidence, dict):
        return ["verifier-evidence.json must contain an object"]

    required = {
        "schema_version",
        "outcome_class",
        "task_id",
        "instance_id",
        "base_commit",
        "runtime_image_identity",
        "synthetic_fault",
        "test",
        "result",
        "provenance",
        "patch_sha256",
    }
    missing = sorted(required - set(evidence))
    if missing:
        errors.append("verifier evidence missing: " + ", ".join(missing))
        return errors
    if evidence.get("schema_version") != "coding-verifier-evidence/v1":
        errors.append("verifier evidence schema_version must be coding-verifier-evidence/v1")

    expected_outcome = {
        "coding_success": "VALID_SUCCESS",
        "coding_valid_failure": "VALID_FAILURE",
        "coding_invalid_infra": "INVALID_INFRASTRUCTURE",
    }[fixture_type]
    if evidence.get("outcome_class") != expected_outcome:
        errors.append(f"outcome_class must be {expected_outcome}")
    for field in ("task_id", "instance_id", "runtime_image_identity"):
        if not _is_nonempty_string(evidence.get(field)):
            errors.append(f"verifier evidence {field} must be non-empty")
    if not FULL_SHA_PATTERN.fullmatch(str(evidence.get("base_commit", ""))):
        errors.append("verifier evidence base_commit must be a full 40-character SHA")
    if evidence.get("synthetic_fault") is not manifest.get("synthetic_fault"):
        errors.append("verifier evidence synthetic_fault must match manifest")

    test = evidence.get("test")
    if not isinstance(test, dict):
        errors.append("verifier evidence test must be an object")
    else:
        if not _is_nonempty_string(test.get("command")):
            errors.append("verifier evidence test.command must be non-empty")
        timeout = test.get("timeout_seconds")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            errors.append("verifier evidence test.timeout_seconds must be positive")

    result = evidence.get("result")
    if not isinstance(result, dict):
        errors.append("verifier evidence result must be an object")
    else:
        for field in ("verifier_completed", "timed_out", "crashed"):
            if not isinstance(result.get(field), bool):
                errors.append(f"verifier evidence result.{field} must be boolean")
        exit_code = result.get("exit_code")
        if exit_code is not None and (
            isinstance(exit_code, bool) or not isinstance(exit_code, int)
        ):
            errors.append("verifier evidence result.exit_code must be integer or null")
        resolved = result.get("resolved")
        if resolved is not None and not isinstance(resolved, bool):
            errors.append("verifier evidence result.resolved must be boolean or null")
        reward = result.get("reward")
        if reward is not None and (
            isinstance(reward, bool) or not isinstance(reward, (int, float))
        ):
            errors.append("verifier evidence result.reward must be numeric or null")
        elif isinstance(reward, (int, float)) and not math.isfinite(float(reward)):
            errors.append("verifier evidence result.reward must be finite")

        if fixture_type == "coding_success":
            if not result.get("verifier_completed") or result.get("timed_out") or result.get("crashed"):
                errors.append("VALID_SUCCESS requires a normally completed verifier")
            if resolved is not True or reward != 1:
                errors.append("VALID_SUCCESS requires resolved=true and reward=1")
        elif fixture_type == "coding_valid_failure":
            if not result.get("verifier_completed") or result.get("timed_out") or result.get("crashed"):
                errors.append("VALID_FAILURE requires a normally completed verifier")
            if resolved is not False or reward != 0:
                errors.append("VALID_FAILURE requires resolved=false and reward=0")
        else:
            error_type = result.get("error_type")
            has_infra_signal = bool(result.get("timed_out")) or bool(result.get("crashed")) or _is_nonempty_string(error_type)
            if not has_infra_signal:
                errors.append("INVALID_INFRASTRUCTURE requires timeout, crash, or error_type")
            if resolved is not None or reward is not None:
                errors.append("INVALID_INFRASTRUCTURE must keep resolved and reward null")

    provenance = evidence.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("verifier evidence provenance must be an object")
    else:
        for field in ("source_file", "json_path"):
            if not _is_nonempty_string(provenance.get(field)):
                errors.append(f"verifier evidence provenance.{field} must be non-empty")

    patch_path = fixture_dir / "patch.diff"
    expected_patch_sha = evidence.get("patch_sha256")
    if not SHA256_PATTERN.fullmatch(str(expected_patch_sha or "")):
        errors.append("verifier evidence patch_sha256 must be 64 lowercase hex characters")
    elif patch_path.is_file() and sha256_file(patch_path) != expected_patch_sha:
        errors.append("verifier evidence patch_sha256 does not match patch.diff")

    if fixture_type in {"coding_success", "coding_valid_failure"}:
        replay = payloads.get("replay.json")
        if not isinstance(replay, dict):
            errors.append("replay.json must contain an object")
        else:
            if replay.get("schema_version") != "coding-clean-replay/v1":
                errors.append("replay schema_version must be coding-clean-replay/v1")
            for field in ("clean_workspace", "patch_applied", "verifier_completed", "resolved"):
                if not isinstance(replay.get(field), bool):
                    errors.append(f"replay.{field} must be boolean")
            if replay.get("clean_workspace") is not True:
                errors.append("replay.clean_workspace must be true")
            if replay.get("resolved") is not (fixture_type == "coding_success"):
                errors.append("replay.resolved does not match fixture outcome")

    return errors


def _validate_calculator_evidence(
    manifest: dict[str, Any], payloads: dict[str, Any]
) -> list[str]:
    fixture_type = manifest.get("fixture_type")
    if fixture_type not in {"calculator_success", "calculator_fault"}:
        return []
    summary = payloads.get("summary.json")
    if not isinstance(summary, dict):
        return ["summary.json must contain an object"]

    errors: list[str] = []
    status = str(summary.get("status", "")).strip().lower()
    trajectory = summary.get("trajectory")
    metadata = trajectory.get("metadata", {}) if isinstance(trajectory, dict) else {}
    evaluation = metadata.get("evaluation", {}) if isinstance(metadata, dict) else {}
    report = evaluation.get("report", {}) if isinstance(evaluation, dict) else {}
    reward = (
        evaluation.get("outcome_reward")
        if isinstance(evaluation, dict) and "outcome_reward" in evaluation
        else summary.get("reward")
    )

    if fixture_type == "calculator_success":
        if status != "completed":
            errors.append("calculator_success requires summary.status=COMPLETED")
        if not isinstance(evaluation, dict) or not evaluation:
            errors.append("calculator_success requires evaluator evidence")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            errors.append("calculator_success requires a numeric evaluator outcome_reward")
        if not isinstance(report, dict) or not report:
            errors.append("calculator_success requires evaluation.report")
        elif (
            report.get("error_eval") is True
            or report.get("test_timeout") is True
            or report.get("failed_apply_patch") is True
        ):
            errors.append("calculator_success requires a normally completed evaluator path")
    else:
        if status not in {"error", "timeout", "failed"}:
            errors.append("calculator_fault requires terminal infrastructure error status")
        if reward is not None:
            errors.append("calculator_fault must not map infrastructure failure to reward")
        if not _is_nonempty_string(summary.get("error")):
            errors.append("calculator_fault requires an explicit error message")
    return errors


def verify_fixture(
    fixture_dir: Path,
    schema_path: Path,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> list[CheckResult]:
    """Return deterministic checks for one fixture directory without modifying it."""
    if max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be positive")

    results: list[CheckResult] = []
    if not fixture_dir.exists():
        return [CheckResult("fixture_directory", FAIL, f"does not exist: {fixture_dir}")]
    if not fixture_dir.is_dir():
        return [CheckResult("fixture_directory", FAIL, f"not a directory: {fixture_dir}")]
    results.append(CheckResult("fixture_directory", PASS, str(fixture_dir)))

    if not schema_path.is_file():
        raise ValueError(f"schema file does not exist: {schema_path}")
    schema = load_json(schema_path)
    if not isinstance(schema, dict):
        raise ValueError("schema root must be a JSON object")
    schema_version = schema.get("properties", {}).get("schema_version", {}).get("const")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError("schema does not describe the expected manifest version")
    results.append(CheckResult("schema_file", PASS, str(schema_path)))

    manifest_path = fixture_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        results.append(CheckResult("manifest_json", FAIL, f"missing {MANIFEST_FILE}"))
        results.append(
            CheckResult(
                "required_json_files",
                FAIL,
                f"manifest unavailable; expected {', '.join(BASE_REQUIRED_JSON_FILES)}",
            )
        )
        return results

    try:
        manifest = load_json(manifest_path)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        results.append(CheckResult("manifest_json", FAIL, f"invalid JSON: {exc}"))
        return results
    if not isinstance(manifest, dict):
        results.append(CheckResult("manifest_json", FAIL, "root must be a JSON object"))
        return results
    results.append(CheckResult("manifest_json", PASS, MANIFEST_FILE))

    contract_errors = _validate_manifest_contract(manifest)
    results.append(
        CheckResult(
            "manifest_contract",
            FAIL if contract_errors else PASS,
            _format_errors(contract_errors) if contract_errors else MANIFEST_SCHEMA_VERSION,
        )
    )

    entries = [entry for entry in manifest.get("files", []) if isinstance(entry, dict)]
    entry_by_path = {
        entry["path"]: entry
        for entry in entries
        if _is_safe_relative_path(entry.get("path"))
    }

    required_files = _required_files_for_type(manifest.get("fixture_type"))
    required_errors: list[str] = []
    required_payloads: dict[str, Any] = {}
    for filename, expected_role in required_files.items():
        entry = entry_by_path.get(filename)
        if entry is None:
            required_errors.append(f"{filename} missing from manifest")
        elif entry.get("role") != expected_role:
            required_errors.append(f"{filename} must use role={expected_role}")

        file_path = fixture_dir / filename
        if not file_path.is_file():
            required_errors.append(f"{filename} missing from fixture")
            continue
        if filename.endswith(".json"):
            try:
                required_payloads[filename] = load_json(file_path)
            except (json.JSONDecodeError, UnicodeDecodeError):
                required_errors.append(f"{filename} is not valid UTF-8 JSON")

    results.append(
        CheckResult(
            "required_json_files",
            FAIL if required_errors else PASS,
            _format_errors(required_errors)
            if required_errors
            else ", ".join(required_files),
        )
    )

    coding_evidence_errors = _validate_coding_evidence(
        fixture_dir, manifest, required_payloads
    )
    results.append(
        CheckResult(
            "coding_evidence",
            FAIL if coding_evidence_errors else PASS,
            _format_errors(coding_evidence_errors)
            if coding_evidence_errors
            else (
                "not a coding fixture"
                if not str(manifest.get("fixture_type", "")).startswith("coding_")
                else "outcome and replay evidence are consistent"
            ),
        )
    )

    calculator_evidence_errors = _validate_calculator_evidence(
        manifest, required_payloads
    )
    results.append(
        CheckResult(
            "calculator_evidence",
            FAIL if calculator_evidence_errors else PASS,
            _format_errors(calculator_evidence_errors)
            if calculator_evidence_errors
            else (
                "not a calculator fixture"
                if not str(manifest.get("fixture_type", "")).startswith("calculator_")
                else "outcome semantics match fixture type"
            ),
        )
    )

    path_errors: list[str] = []
    integrity_errors: list[str] = []
    fixture_root = fixture_dir.resolve()
    for relative_path, entry in entry_by_path.items():
        file_path = fixture_dir / relative_path
        if file_path.is_symlink():
            path_errors.append(f"symlink is not allowed: {relative_path}")
            continue
        try:
            resolved_path = file_path.resolve()
            resolved_path.relative_to(fixture_root)
        except (OSError, ValueError):
            path_errors.append(f"path escapes fixture directory: {relative_path}")
            continue
        if not file_path.is_file():
            integrity_errors.append(f"missing file: {relative_path}")
            continue
        actual_bytes = file_path.stat().st_size
        if entry.get("bytes") != actual_bytes:
            integrity_errors.append(
                f"size mismatch for {relative_path}: expected {entry.get('bytes')}, got {actual_bytes}"
            )
        expected_sha = entry.get("sha256")
        if SHA256_PATTERN.fullmatch(str(expected_sha or "")):
            actual_sha = sha256_file(file_path)
            if actual_sha != expected_sha:
                integrity_errors.append(f"checksum mismatch: {relative_path}")

    results.append(
        CheckResult(
            "manifest_paths",
            FAIL if path_errors else PASS,
            _format_errors(path_errors) if path_errors else "all paths stay inside fixture",
        )
    )
    results.append(
        CheckResult(
            "file_integrity",
            FAIL if integrity_errors else PASS,
            _format_errors(integrity_errors)
            if integrity_errors
            else f"verified {len(entry_by_path)} files",
        )
    )

    allowed_unlisted = {MANIFEST_FILE, "README.md"}
    actual_files = {
        path.relative_to(fixture_dir).as_posix()
        for path in fixture_dir.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    unlisted = sorted(actual_files - set(entry_by_path) - allowed_unlisted)
    results.append(
        CheckResult(
            "unlisted_files",
            FAIL if unlisted else PASS,
            f"not declared in manifest: {', '.join(unlisted)}" if unlisted else "none",
        )
    )

    oversized = sorted(
        f"{path.relative_to(fixture_dir).as_posix()} ({path.stat().st_size} bytes)"
        for path in fixture_dir.rglob("*")
        if path.is_file() and not path.is_symlink() and path.stat().st_size > max_file_bytes
    )
    results.append(
        CheckResult(
            "repository_size",
            FAIL if oversized else PASS,
            _format_errors(oversized)
            if oversized
            else f"all files <= {max_file_bytes} bytes",
        )
    )

    json_payload_errors: list[str] = []
    all_json_payloads = dict(required_payloads)
    for relative_path, entry in entry_by_path.items():
        if relative_path in all_json_payloads:
            continue
        if entry.get("media_type") != "application/json" and not relative_path.endswith(".json"):
            continue
        file_path = fixture_dir / relative_path
        if not file_path.is_file() or file_path.is_symlink():
            continue
        try:
            all_json_payloads[relative_path] = load_json(file_path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            json_payload_errors.append(f"{relative_path} is not valid UTF-8 JSON")
    results.append(
        CheckResult(
            "manifest_json_payloads",
            FAIL if json_payload_errors else PASS,
            _format_errors(json_payload_errors)
            if json_payload_errors
            else f"verified {len(all_json_payloads)} JSON payloads",
        )
    )

    sensitive: list[str] = []
    for filename, payload in all_json_payloads.items():
        sensitive.extend(f"{filename}:{location}" for location in _sensitive_locations(payload))
    for relative_path, entry in entry_by_path.items():
        if not str(entry.get("media_type", "")).startswith("text/"):
            continue
        file_path = fixture_dir / relative_path
        if not file_path.is_file() or file_path.is_symlink():
            continue
        try:
            text_payload = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if PRIVATE_KEY_PATTERN.search(text_payload):
            sensitive.append(f"{relative_path}:private_key_material")
    results.append(
        CheckResult(
            "obvious_sensitive_fields",
            FAIL if sensitive else PASS,
            _format_errors(sensitive) if sensitive else "none detected in manifest payloads",
        )
    )

    return results


def main() -> int:
    args = parse_args()
    try:
        results = verify_fixture(args.fixture_dir, args.schema, args.max_file_bytes)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        print(f"FAIL verifier_error: {exc}")
        return 2

    for result in results:
        print(f"{result.status} {result.name}: {result.message}")
    return 1 if any(result.status == FAIL for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
