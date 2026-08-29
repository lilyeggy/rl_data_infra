"""Public deterministic JSONL interchange adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.contracts._json import canonical_json_bytes, sha256_bytes
from src.contracts.capabilities import (
    Capability,
    capabilities_for_record,
    common_capabilities,
)
from src.contracts.rollout_record import RolloutRecord
from src.errors import AdapterIssue, ErrorCode, ContractValidationError
from src.sources.base import AdapterResult


JSONL_SCHEMA_VERSION = "rollout-jsonl/v1"


def dumps_jsonl(records: Iterable[RolloutRecord]) -> str:
    """Serialize one canonical record per stable, newline-terminated envelope."""

    lines = []
    for record in records:
        envelope = {
            "schema_version": JSONL_SCHEMA_VERSION,
            "record": record.to_dict(),
        }
        lines.append(canonical_json_bytes(envelope).decode("utf-8"))
    return "" if not lines else "\n".join(lines) + "\n"


class JsonlSourceAdapter:
    """Convert the public JSONL envelope into canonical rollout records.

    ``capabilities()`` describes what the format can carry. ``AdapterResult``
    reports only capabilities present on every successfully parsed record.
    """

    name = "jsonl"
    version = "v1"

    def __init__(self, required_capabilities: Iterable[Capability] = ()) -> None:
        self.required_capabilities = frozenset(required_capabilities)

    def capabilities(self) -> frozenset[Capability]:
        return frozenset(Capability)

    def convert(self, payload: object) -> AdapterResult:
        if isinstance(payload, bytes):
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                return AdapterResult(
                    errors=(
                        AdapterIssue(
                            code=ErrorCode.ADAPTER_ERROR,
                            message=f"JSONL payload is not UTF-8: {exc}",
                        ),
                    )
                )
        elif isinstance(payload, str):
            text = payload
        else:
            return AdapterResult(
                errors=(
                    AdapterIssue(
                        code=ErrorCode.ADAPTER_ERROR,
                        message="JsonlSourceAdapter expects str or UTF-8 bytes",
                    ),
                )
            )
        return self._convert_text(text, source_ref=None)

    def convert_file(self, path: str | Path) -> AdapterResult:
        source_path = Path(path)
        try:
            payload = source_path.read_bytes()
            text = payload.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return AdapterResult(
                errors=(
                    AdapterIssue(
                        code=ErrorCode.ADAPTER_ERROR,
                        message=f"cannot read JSONL source {source_path}: {exc}",
                    ),
                )
            )
        return self._convert_text(text, source_ref=str(source_path))

    def _convert_text(self, text: str, source_ref: str | None) -> AdapterResult:
        records: list[RolloutRecord] = []
        warnings: list[AdapterIssue] = []
        errors: list[AdapterIssue] = []

        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            if not raw_line.strip():
                continue
            source_record_id = f"line:{line_number}"
            try:
                envelope = json.loads(raw_line)
                record = self._parse_envelope(envelope)
                source_payload_ref = (
                    f"{source_ref}#{source_record_id}" if source_ref is not None else None
                )
                record = record.with_source_envelope(
                    source_type=self.name,
                    source_record_id=source_record_id,
                    source_payload_ref=source_payload_ref,
                    source_payload_sha256=sha256_bytes(raw_line.encode("utf-8")),
                )
            except json.JSONDecodeError as exc:
                errors.append(
                    AdapterIssue(
                        code=ErrorCode.ADAPTER_ERROR,
                        message=f"line {line_number}: {exc}",
                        source_record_id=source_record_id,
                    )
                )
                continue
            except ContractValidationError as exc:
                errors.append(
                    AdapterIssue(
                        code=ErrorCode.CONTRACT_INVALID,
                        message=f"line {line_number}: {exc}",
                        source_record_id=source_record_id,
                    )
                )
                continue
            except (TypeError, ValueError) as exc:
                errors.append(
                    AdapterIssue(
                        code=ErrorCode.ADAPTER_ERROR,
                        message=f"line {line_number}: {exc}",
                        source_record_id=source_record_id,
                    )
                )
                continue

            missing = self.required_capabilities - capabilities_for_record(record)
            if missing:
                errors.append(
                    AdapterIssue(
                        code=ErrorCode.CAPABILITY_MISSING,
                        message="required capabilities are absent: "
                        + ", ".join(sorted(item.value for item in missing)),
                        source_record_id=source_record_id,
                        field="capabilities",
                        details={"missing": sorted(item.value for item in missing)},
                    )
                )
            records.append(record)

        if not records and not errors and text.strip() == "":
            warnings.append(
                AdapterIssue(
                    code=ErrorCode.SOURCE_WARNING,
                    message="JSONL payload contains no records",
                )
            )
        return AdapterResult(
            records=tuple(records),
            capabilities=common_capabilities(records),
            warnings=tuple(warnings),
            errors=tuple(errors),
        )

    @staticmethod
    def _parse_envelope(value: Any) -> RolloutRecord:
        if not isinstance(value, Mapping):
            raise ValueError("JSONL line must be an object")
        unknown = sorted(set(value) - {"schema_version", "record"})
        if unknown:
            raise ValueError(
                "unknown JSONL envelope fields: " + ", ".join(unknown)
            )
        if value.get("schema_version") != JSONL_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {JSONL_SCHEMA_VERSION!r}"
            )
        if "record" not in value:
            raise ValueError("JSONL envelope is missing record")
        return RolloutRecord.from_dict(value["record"])
