#!/usr/bin/env python3
"""Package reviewed Polar artifacts into a verifiable golden fixture.

The input directory must already contain reviewed/redacted staging artifacts.
This tool copies bytes, records provenance, and computes checksums.  It never
derives or fills token IDs, logprobs, reward, or policy version.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from scripts.verify_polar_fixture import (
        DEFAULT_MAX_FILE_BYTES,
        FAIL,
        sha256_file,
        verify_fixture,
    )
except ModuleNotFoundError:  # Support `python scripts/package_polar_fixture.py`.
    from verify_polar_fixture import (  # type: ignore[no-redef]
        DEFAULT_MAX_FILE_BYTES,
        FAIL,
        sha256_file,
        verify_fixture,
    )


BASE_REQUIRED_FILES = {
    "request.json": "request",
    "response.json": "response",
    "summary.json": "summary",
}
CODING_REQUIRED_FILES = {
    "verifier-evidence.json": "verifier_evidence",
    "patch.diff": "patch",
}
CODING_REPLAY_FILES = {"replay.json": "replay_evidence"}
OPTIONAL_EVIDENCE_FILES = {
    "fault-injection.json": "fault_injection",
    "task-metadata.json": "task_metadata",
}
OPTIONAL_ROOT_LOGS = {
    "gateway.log",
    "rollout-server.log",
    "sglang.log",
    "runtime.log",
    "evaluator.log",
}
ALLOWED_EXISTING_OUTPUT_FILES = {"README.md"}


class PackageError(ValueError):
    """Expected rejection caused by unsafe or incomplete package input."""


@dataclass(frozen=True)
class PackageMetadata:
    fixture_id: str
    fixture_type: str
    polar_commit: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    runtime_image_identity: str
    harness: str
    synthetic_fault: bool | None = None
    policy_version: str | None = None
    created_at_utc: str | None = None
    known_missing_fields: tuple[str, ...] = ()
    redactions: tuple[dict[str, Any], ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceFile:
    relative_path: str
    role: str
    media_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package reviewed Polar artifacts as a golden fixture."
    )
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--fixture-type",
        required=True,
        choices=(
            "calculator_success",
            "calculator_fault",
            "coding_success",
            "coding_valid_failure",
            "coding_invalid_infra",
        ),
    )
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--polar-commit", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--runtime-image-identity", required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument(
        "--synthetic-fault",
        action="store_true",
        default=None,
        help="Mark a coding_invalid_infra fixture as intentionally injected.",
    )
    parser.add_argument("--policy-version")
    parser.add_argument("--created-at-utc")
    parser.add_argument(
        "--missing-field",
        action="append",
        default=[],
        help="Known source field that is absent. May be repeated.",
    )
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="Manifest note. May be repeated.",
    )
    parser.add_argument(
        "--redactions-json",
        type=Path,
        help="Optional JSON file containing the manifest redactions array.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("tests/fixtures/polar/source-manifest.schema.json"),
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def load_redactions(path: Path | None) -> tuple[dict[str, Any], ...]:
    if path is None:
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"cannot read redactions JSON: {exc}") from exc
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise PackageError("redactions JSON must contain an array of objects")
    return tuple(payload)


def _safe_relative_path(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise PackageError(f"source path escapes staging directory: {path}") from exc
    pure = PurePosixPath(relative)
    if not relative or pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise PackageError(f"unsafe source path: {relative}")
    return relative


def _media_type(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    if path.suffix in {".log", ".txt"}:
        return "text/plain"
    if path.suffix in {".diff", ".patch"}:
        return "text/x-diff"
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _required_files_for_type(fixture_type: str) -> dict[str, str]:
    required = dict(BASE_REQUIRED_FILES)
    if fixture_type.startswith("coding_"):
        required.update(CODING_REQUIRED_FILES)
    if fixture_type in {"coding_success", "coding_valid_failure"}:
        required.update(CODING_REPLAY_FILES)
    return required


def discover_source_files(source_dir: Path, fixture_type: str) -> list[SourceFile]:
    """Return an allowlisted, deterministic inventory of staging files."""
    if not source_dir.exists():
        raise PackageError(f"source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise PackageError(f"source path is not a directory: {source_dir}")
    if source_dir.is_symlink():
        raise PackageError("source directory must not be a symlink")

    discovered: list[SourceFile] = []
    allowed_paths: set[str] = set()
    for filename, role in _required_files_for_type(fixture_type).items():
        path = source_dir / filename
        if not path.is_file() or path.is_symlink():
            raise PackageError(f"missing required regular file: {filename}")
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PackageError(f"invalid UTF-8 JSON in {filename}: {exc}") from exc
        else:
            try:
                path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise PackageError(f"invalid UTF-8 text in {filename}: {exc}") from exc
        discovered.append(SourceFile(filename, role, _media_type(path)))
        allowed_paths.add(filename)

    for filename, role in OPTIONAL_EVIDENCE_FILES.items():
        path = source_dir / filename
        if not path.exists():
            continue
        if not path.is_file() or path.is_symlink():
            raise PackageError(f"optional evidence is not a regular file: {filename}")
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PackageError(f"invalid UTF-8 JSON in {filename}: {exc}") from exc
        discovered.append(SourceFile(filename, role, "application/json"))
        allowed_paths.add(filename)

    for filename in sorted(OPTIONAL_ROOT_LOGS):
        path = source_dir / filename
        if path.exists():
            if not path.is_file() or path.is_symlink():
                raise PackageError(f"optional log is not a regular file: {filename}")
            discovered.append(SourceFile(filename, "raw_log", _media_type(path)))
            allowed_paths.add(filename)

    normalized_root = source_dir / "normalized-logs"
    if normalized_root.exists():
        if not normalized_root.is_dir() or normalized_root.is_symlink():
            raise PackageError("normalized-logs must be a real directory")
        for path in sorted(normalized_root.rglob("*")):
            if path.is_symlink():
                raise PackageError(f"symlink is not allowed in staging: {path}")
            if path.is_file():
                relative = _safe_relative_path(path, source_dir)
                discovered.append(SourceFile(relative, "normalized_log", _media_type(path)))
                allowed_paths.add(relative)

    actual_files: set[str] = set()
    for path in source_dir.rglob("*"):
        if path.is_symlink():
            raise PackageError(f"symlink is not allowed in staging: {path}")
        if path.is_file():
            actual_files.add(_safe_relative_path(path, source_dir))
    unexpected = sorted(actual_files - allowed_paths)
    if unexpected:
        raise PackageError(
            "unexpected staging files; explicitly classify or remove them: "
            + ", ".join(unexpected)
        )

    return sorted(discovered, key=lambda item: item.relative_path)


def ensure_safe_output(output_dir: Path, source_dir: Path, max_file_bytes: int) -> None:
    source_resolved = source_dir.resolve()
    output_resolved = output_dir.resolve()
    if output_resolved == source_resolved:
        raise PackageError("source and output directories must be different")
    try:
        output_resolved.relative_to(source_resolved)
    except ValueError:
        pass
    else:
        raise PackageError("output directory must not be inside the source directory")

    if output_dir.exists() and not output_dir.is_dir():
        raise PackageError(f"output path is not a directory: {output_dir}")
    if output_dir.is_symlink():
        raise PackageError("output directory must not be a symlink")
    if not output_dir.exists():
        return

    existing_files = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    conflicts = sorted(existing_files - ALLOWED_EXISTING_OUTPUT_FILES)
    if conflicts:
        raise PackageError(
            "refusing to overwrite existing fixture files: " + ", ".join(conflicts)
        )
    readme = output_dir / "README.md"
    if readme.is_symlink():
        raise PackageError("output README.md must not be a symlink")
    if readme.is_file() and readme.stat().st_size > max_file_bytes:
        raise PackageError("existing README.md exceeds the fixture file size limit")


def build_manifest(
    fixture_root: Path,
    source_files: list[SourceFile],
    metadata: PackageMetadata,
) -> dict[str, Any]:
    file_entries: list[dict[str, Any]] = []
    for source_file in source_files:
        path = fixture_root / source_file.relative_path
        file_entries.append(
            {
                "path": source_file.relative_path,
                "role": source_file.role,
                "media_type": source_file.media_type,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return {
        "schema_version": "polar-fixture-manifest/v1",
        "fixture_id": metadata.fixture_id,
        "fixture_type": metadata.fixture_type,
        "synthetic_fault": (
            metadata.synthetic_fault
            if metadata.synthetic_fault is not None
            else metadata.fixture_type == "calculator_fault"
        ),
        "created_at_utc": metadata.created_at_utc or utc_now(),
        "source": {
            "polar_commit": metadata.polar_commit,
            "model_id": metadata.model_id,
            "model_revision": metadata.model_revision,
            "tokenizer_revision": metadata.tokenizer_revision,
            "runtime_image_identity": metadata.runtime_image_identity,
            "harness": metadata.harness,
            "policy_version": metadata.policy_version,
        },
        "files": file_entries,
        "redactions": list(metadata.redactions),
        "known_missing_fields": sorted(set(metadata.known_missing_fields)),
        "notes": list(metadata.notes),
    }


def package_fixture(
    source_dir: Path,
    output_dir: Path,
    schema_path: Path,
    metadata: PackageMetadata,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> dict[str, Any]:
    """Create and verify a fixture, refusing overwrite and partial source input."""
    if max_file_bytes <= 0:
        raise PackageError("max_file_bytes must be positive")
    source_files = discover_source_files(source_dir, metadata.fixture_type)
    ensure_safe_output(output_dir, source_dir, max_file_bytes)

    oversized = [
        item.relative_path
        for item in source_files
        if (source_dir / item.relative_path).stat().st_size > max_file_bytes
    ]
    if oversized:
        raise PackageError(
            "staging files exceed repository limit; use an external reference: "
            + ", ".join(oversized)
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-package-", dir=output_dir.parent
    ) as temp_dir:
        package_root = Path(temp_dir) / "fixture"
        package_root.mkdir()
        for source_file in source_files:
            source_path = source_dir / source_file.relative_path
            destination = package_root / source_file.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)

        manifest = build_manifest(package_root, source_files, metadata)
        (package_root / "source-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        verification = verify_fixture(
            package_root,
            schema_path,
            max_file_bytes=max_file_bytes,
        )
        failures = [result for result in verification if result.status == FAIL]
        if failures:
            details = "; ".join(f"{item.name}: {item.message}" for item in failures)
            raise PackageError(f"generated fixture failed verification: {details}")

        output_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(package_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(package_root)
            destination = output_dir / relative
            if destination.exists():
                raise PackageError(f"refusing to overwrite: {relative.as_posix()}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

    final_results = verify_fixture(output_dir, schema_path, max_file_bytes=max_file_bytes)
    final_failures = [result for result in final_results if result.status == FAIL]
    if final_failures:
        details = "; ".join(f"{item.name}: {item.message}" for item in final_failures)
        raise PackageError(f"written fixture failed final verification: {details}")
    return manifest


def main() -> int:
    args = parse_args()
    try:
        metadata = PackageMetadata(
            fixture_id=args.fixture_id,
            fixture_type=args.fixture_type,
            polar_commit=args.polar_commit,
            model_id=args.model_id,
            model_revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            runtime_image_identity=args.runtime_image_identity,
            harness=args.harness,
            synthetic_fault=args.synthetic_fault,
            policy_version=args.policy_version,
            created_at_utc=args.created_at_utc,
            known_missing_fields=tuple(args.missing_field),
            redactions=load_redactions(args.redactions_json),
            notes=tuple(args.note),
        )
        manifest = package_fixture(
            args.source_dir,
            args.output_dir,
            args.schema,
            metadata,
            max_file_bytes=args.max_file_bytes,
        )
    except (OSError, PackageError) as exc:
        print(f"FAIL package_error: {exc}")
        return 1

    print(f"PASS packaged_fixture: {args.output_dir}")
    print(f"fixture_id={manifest['fixture_id']}")
    print(f"files={len(manifest['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
