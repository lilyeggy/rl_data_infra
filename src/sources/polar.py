"""Polar fixture adapter implemented without importing the Polar package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.contracts import (
    Capability,
    ComponentStatus,
    RolloutRecord,
    RolloutStatus,
    VerifierStatus,
    capabilities_for_record,
    common_capabilities,
)
from src.errors import AdapterIssue, ContractValidationError, ErrorCode
from src.sources.base import AdapterResult


class PolarSourceAdapter:
    """Convert a reviewed project-side Polar fixture into canonical records.

    This adapter reads the persisted JSON contract only. It intentionally has
    no import-time or runtime dependency on the upstream ``polar`` package.
    """

    name = "polar"
    version = "v1"

    def __init__(self, required_capabilities: Iterable[Capability] = ()) -> None:
        self.required_capabilities = frozenset(required_capabilities)

    def capabilities(self) -> frozenset[Capability]:
        return frozenset(Capability)

    def convert(self, payload: object) -> AdapterResult:
        if not isinstance(payload, (str, Path)):
            return self._adapter_error("PolarSourceAdapter expects a fixture directory path")
        fixture_dir = Path(payload)
        try:
            manifest, summary, summary_sha = self._load_fixture(fixture_dir)
            records, warnings = self._records_from_summary(
                fixture_dir, manifest, summary, summary_sha
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return self._adapter_error(f"cannot read Polar fixture {fixture_dir}: {exc}")
        except ContractValidationError as exc:
            return AdapterResult(
                errors=(
                    AdapterIssue(code=ErrorCode.CONTRACT_INVALID, message=str(exc)),
                )
            )

        errors: list[AdapterIssue] = []
        for record in records:
            missing = self.required_capabilities - capabilities_for_record(record)
            if missing:
                errors.append(
                    AdapterIssue(
                        code=ErrorCode.CAPABILITY_MISSING,
                        message="required capabilities are absent: "
                        + ", ".join(sorted(item.value for item in missing)),
                        source_record_id=record.source_record_id,
                        field="capabilities",
                        details={"missing": sorted(item.value for item in missing)},
                    )
                )
        return AdapterResult(
            records=tuple(records),
            capabilities=common_capabilities(records),
            warnings=tuple(warnings),
            errors=tuple(errors),
        )

    @staticmethod
    def _adapter_error(message: str) -> AdapterResult:
        return AdapterResult(
            errors=(AdapterIssue(code=ErrorCode.ADAPTER_ERROR, message=message),)
        )

    @staticmethod
    def _load_fixture(
        fixture_dir: Path,
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        if not fixture_dir.is_dir() or fixture_dir.is_symlink():
            raise ContractValidationError(f"not a safe fixture directory: {fixture_dir}")
        manifest_path = fixture_dir / "source-manifest.json"
        summary_path = fixture_dir / "summary.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(summary, dict):
            raise ContractValidationError("manifest and summary must contain JSON objects")
        if manifest.get("schema_version") != "polar-fixture-manifest/v1":
            raise ContractValidationError("unsupported Polar fixture manifest version")

        entries = manifest.get("files")
        if not isinstance(entries, list):
            raise ContractValidationError("manifest.files must be an array")
        summary_entry = next(
            (
                entry
                for entry in entries
                if isinstance(entry, dict) and entry.get("path") == "summary.json"
            ),
            None,
        )
        if summary_entry is None or summary_entry.get("role") != "summary":
            raise ContractValidationError("manifest must declare summary.json with role=summary")
        summary_bytes = summary_path.read_bytes()
        summary_sha = hashlib.sha256(summary_bytes).hexdigest()
        if summary_entry.get("bytes") != len(summary_bytes):
            raise ContractValidationError("summary.json size does not match manifest")
        if summary_entry.get("sha256") != summary_sha:
            raise ContractValidationError("summary.json checksum does not match manifest")
        return manifest, summary, summary_sha

    def _records_from_summary(
        self,
        fixture_dir: Path,
        manifest: dict[str, Any],
        summary: dict[str, Any],
        summary_sha: str,
    ) -> tuple[list[RolloutRecord], list[AdapterIssue]]:
        task_id = self._required_text(summary.get("task_id"), "summary.task_id")
        session_id = self._required_text(summary.get("session_id"), "summary.session_id")
        source = manifest.get("source")
        if not isinstance(source, dict):
            raise ContractValidationError("manifest.source must be an object")
        fixture_type = self._required_text(manifest.get("fixture_type"), "fixture_type")
        trajectory = summary.get("trajectory")
        if not isinstance(trajectory, dict):
            raise ContractValidationError("summary.trajectory must be an object")
        traces = trajectory.get("traces", [])
        if not isinstance(traces, list):
            raise ContractValidationError("summary.trajectory.traces must be an array")

        records: list[RolloutRecord] = []
        warnings: list[AdapterIssue] = []
        if traces:
            for trace_index, trace in enumerate(traces):
                if not isinstance(trace, dict):
                    raise ContractValidationError(
                        f"summary.trajectory.traces[{trace_index}] must be an object"
                    )
                records.append(
                    self._trace_record(
                        fixture_dir=fixture_dir,
                        manifest=manifest,
                        source=source,
                        summary=summary,
                        trajectory=trajectory,
                        trace=trace,
                        trace_index=trace_index,
                        trace_count=len(traces),
                        task_id=task_id,
                        session_id=session_id,
                        fixture_type=fixture_type,
                        summary_sha=summary_sha,
                    )
                )
        else:
            records.append(
                self._empty_failure_record(
                    fixture_dir=fixture_dir,
                    manifest=manifest,
                    source=source,
                    summary=summary,
                    task_id=task_id,
                    session_id=session_id,
                    fixture_type=fixture_type,
                    summary_sha=summary_sha,
                )
            )

        rewards = [record.reward for record in records]
        if fixture_type == "coding_success" and (
            not rewards or any(reward != 1 for reward in rewards)
        ):
            warnings.append(
                AdapterIssue(
                    code=ErrorCode.SOURCE_WARNING,
                    message=(
                        f"fixture_type={fixture_type} conflicts with observed reward(s) "
                        f"{rewards}; source evidence is preserved without relabeling"
                    ),
                    source_record_id=session_id,
                    field="fixture_type",
                )
            )
        elif fixture_type == "calculator_success" and any(
            reward != 1 for reward in rewards
        ):
            warnings.append(
                AdapterIssue(
                    code=ErrorCode.SOURCE_WARNING,
                    message=(
                        "calculator_success records an execution-complete path, but the "
                        f"observed task reward(s) are {rewards}; do not describe it as a "
                        "task-level success"
                    ),
                    source_record_id=session_id,
                    field="fixture_type",
                )
            )
        return records, warnings

    def _trace_record(
        self,
        *,
        fixture_dir: Path,
        manifest: dict[str, Any],
        source: dict[str, Any],
        summary: dict[str, Any],
        trajectory: dict[str, Any],
        trace: dict[str, Any],
        trace_index: int,
        trace_count: int,
        task_id: str,
        session_id: str,
        fixture_type: str,
        summary_sha: str,
    ) -> RolloutRecord:
        """这部分很重要，是如何把 polar 的 trace 转换为rollout record"""
        prompt_ids = self._int_list(trace.get("prompt_ids"), "prompt_ids")
        response_ids = self._int_list(trace.get("response_ids"), "response_ids")
        response_mask = self._int_list(trace.get("loss_mask"), "loss_mask")
        if len(response_mask) != len(response_ids):
            raise ContractValidationError("Polar loss_mask must match response_ids length")
        response_logprobs = self._float_list_or_none(
            trace.get("response_logprobs"), "response_logprobs"
        )
        if response_logprobs is not None and len(response_logprobs) != len(response_ids):
            raise ContractValidationError(
                "Polar response_logprobs must match response_ids length"
            )
        token_ids = tuple(prompt_ids + response_ids)
        loss_mask = tuple([0] * len(prompt_ids) + response_mask)
        evaluation = self._evaluation(trajectory)
        verifier_status = self._verifier_status(evaluation)
        structured_tool_events = self._structured_tool_events(trace)
        source_record_id = session_id if trace_count == 1 else f"{session_id}:trace:{trace_index}"
        trajectory_id = source_record_id
        status = self._rollout_status(summary.get("status"))

        return RolloutRecord(
            trajectory_id=trajectory_id,
            task_id=task_id,
            group_id=source.get("group_id"),
            policy_version=source.get("policy_version"),
            source_type=self.name,
            source_record_id=source_record_id,
            token_ids=token_ids,
            prompt_token_count=len(prompt_ids),
            loss_mask=loss_mask,
            # Polar's pinned prefix_merging builder defines response_logprobs as
            # sampled-policy logprobs aligned with response_ids/loss_mask.
            old_logprobs=response_logprobs,
            reward=trace.get("reward"),
            rollout_status=status,
            runtime_status=ComponentStatus.UNKNOWN,
            harness_status=ComponentStatus.UNKNOWN,
            model_backend_status=ComponentStatus.UNKNOWN,
            verifier_status=verifier_status,
            verifier_evidence_ref=(
                f"{fixture_dir / 'summary.json'}#/trajectory/metadata/evaluation"
                if evaluation
                else None
            ),
            source_payload_ref=str(fixture_dir / "summary.json"),
            source_payload_sha256=summary_sha,
            model_id=source.get("model_id"),
            model_revision=source.get("model_revision"),
            tokenizer_revision=source.get("tokenizer_revision"),
            tool_events=structured_tool_events,
            opaque_metadata={
                "polar": {
                    "fixture_type": fixture_type,
                    "session_id": session_id,
                    "node_id": summary.get("node_id"),
                    "trace_index": trace_index,
                    "trace_count": trace_count,
                    "finish_reason": trace.get("finish_reason"),
                    "timing": summary.get("timing"),
                    "evaluation": evaluation or None,
                    "response_logprobs_provenance": (
                        f"$.trajectory.traces[{trace_index}].response_logprobs"
                        if response_logprobs is not None
                        else None
                    ),
                    "manifest_file_sha256": self._manifest_checksums(manifest),
                }
            },
        )

    def _empty_failure_record(
        self,
        *,
        fixture_dir: Path,
        manifest: dict[str, Any],
        source: dict[str, Any],
        summary: dict[str, Any],
        task_id: str,
        session_id: str,
        fixture_type: str,
        summary_sha: str,
    ) -> RolloutRecord:
        error = summary.get("error")
        runtime_status = (
            ComponentStatus.FAILED
            if isinstance(error, str) and "runtime" in error.lower()
            else ComponentStatus.UNKNOWN
        )
        return RolloutRecord(
            trajectory_id=session_id,
            task_id=task_id,
            group_id=source.get("group_id"),
            policy_version=source.get("policy_version"),
            source_type=self.name,
            source_record_id=session_id,
            rollout_status=self._rollout_status(summary.get("status")),
            runtime_status=runtime_status,
            harness_status=ComponentStatus.NOT_RUN,
            model_backend_status=ComponentStatus.NOT_RUN,
            verifier_status=VerifierStatus.NOT_RUN,
            source_payload_ref=str(fixture_dir / "summary.json"),
            source_payload_sha256=summary_sha,
            model_id=source.get("model_id"),
            model_revision=source.get("model_revision"),
            tokenizer_revision=source.get("tokenizer_revision"),
            opaque_metadata={
                "polar": {
                    "fixture_type": fixture_type,
                    "session_id": session_id,
                    "node_id": summary.get("node_id"),
                    "timing": summary.get("timing"),
                    "error": error,
                    "manifest_file_sha256": self._manifest_checksums(manifest),
                }
            },
        )

    @staticmethod
    def _manifest_checksums(manifest: dict[str, Any]) -> dict[str, str]:
        entries = manifest.get("files", [])
        return {
            str(entry["path"]): str(entry["sha256"])
            for entry in entries
            if isinstance(entry, dict) and "path" in entry and "sha256" in entry
        }

    @staticmethod
    def _evaluation(trajectory: dict[str, Any]) -> dict[str, Any]:
        metadata = trajectory.get("metadata")
        evaluation = metadata.get("evaluation") if isinstance(metadata, dict) else None
        return evaluation if isinstance(evaluation, dict) else {}

    @staticmethod
    def _structured_tool_events(trace: dict[str, Any]) -> tuple[Mapping[str, Any], ...] | None:
        events: list[Mapping[str, Any]] = []
        messages = trace.get("response_messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    events.extend(item for item in tool_calls if isinstance(item, dict))
        return tuple(events) if events else None

    @staticmethod
    def _verifier_status(evaluation: dict[str, Any]) -> VerifierStatus:
        if not evaluation:
            return VerifierStatus.NOT_RUN
        report = evaluation.get("report")
        if not isinstance(report, dict):
            return VerifierStatus.UNKNOWN
        if report.get("test_timeout") is True:
            return VerifierStatus.TIMEOUT
        if report.get("error_eval") is True or report.get("failed_apply_patch") is True:
            return VerifierStatus.ERROR
        if report.get("resolved") is True:
            return VerifierStatus.PASSED
        if report.get("resolved") is False:
            return VerifierStatus.FAILED
        return VerifierStatus.UNKNOWN

    @staticmethod
    def _rollout_status(value: Any) -> RolloutStatus:
        normalized = str(value or "").upper()
        return {
            "COMPLETED": RolloutStatus.COMPLETED,
            "ERROR": RolloutStatus.FAILED,
            "TIMEOUT": RolloutStatus.TIMEOUT,
        }.get(normalized, RolloutStatus.UNKNOWN)

    @staticmethod
    def _required_text(value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ContractValidationError(f"{field_name} must be a non-empty string")
        return value

    @staticmethod
    def _int_list(value: Any, field_name: str) -> list[int]:
        if not isinstance(value, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in value
        ):
            raise ContractValidationError(f"Polar {field_name} must be an integer array")
        return list(value)

    @staticmethod
    def _float_list_or_none(value: Any, field_name: str) -> list[float] | None:
        if value is None:
            return None
        if not isinstance(value, list) or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
        ):
            raise ContractValidationError(f"Polar {field_name} must be a numeric array or null")
        return [float(item) for item in value]
