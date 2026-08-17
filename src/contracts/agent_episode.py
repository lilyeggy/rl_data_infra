"""Canonical assembled view of one Agent task execution."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from src.contracts._json import sha256_json, thaw_json, validate_sha256
from src.contracts._validation import (
    enum_member,
    frozen_object,
    optional_text,
    required_text,
    strict_fields,
    utc_instant,
)
from src.contracts.artifacts import ArtifactRef
from src.contracts.manifests import (
    EnvironmentManifest,
    EvaluatorManifest,
    HarnessManifest,
    ModelManifest,
)
from src.contracts.trace_event import EventType, TraceEvent
from src.errors import ContractValidationError


SCHEMA_VERSION = "agent-episode/v1"


class CaptureCapability(str, Enum):
    MODEL_IO = "MODEL_IO"
    MODEL_TOKEN_USAGE = "MODEL_TOKEN_USAGE"
    MODEL_TOKEN_IDS = "MODEL_TOKEN_IDS"
    MODEL_LOGPROBS = "MODEL_LOGPROBS"
    TOOL_IO = "TOOL_IO"
    SANDBOX_LIFECYCLE = "SANDBOX_LIFECYCLE"
    SANDBOX_COMMAND_IO = "SANDBOX_COMMAND_IO"
    FILE_ARTIFACTS = "FILE_ARTIFACTS"
    VERIFIER_EVIDENCE = "VERIFIER_EVIDENCE"
    HARNESS_DECISIONS = "HARNESS_DECISIONS"
    CONTEXT_SELECTION = "CONTEXT_SELECTION"
    CONTEXT_COMPACTION = "CONTEXT_COMPACTION"
    RETRY_DECISIONS = "RETRY_DECISIONS"
    TERMINATION_DECISIONS = "TERMINATION_DECISIONS"


class TaskStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    UNKNOWN = "UNKNOWN"


class ExecutionValidity(str, Enum):
    VALID = "VALID"
    INFRA_INVALID = "INFRA_INVALID"
    UNKNOWN = "UNKNOWN"


class EpisodeVerifierStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    NOT_RUN = "NOT_RUN"
    UNKNOWN = "UNKNOWN"


class IntegrityState(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    CORRUPT = "CORRUPT"


@dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeOutcome:
    task_status: TaskStatus
    execution_validity: ExecutionValidity
    verifier_status: EpisodeVerifierStatus
    score: float | None = None
    evidence_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, enum_type in (
            ("task_status", TaskStatus),
            ("execution_validity", ExecutionValidity),
            ("verifier_status", EpisodeVerifierStatus),
        ):
            if not isinstance(getattr(self, name), enum_type):
                raise ContractValidationError(f"{name} must be a {enum_type.__name__}")
        if self.score is not None:
            if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
                raise ContractValidationError("score must be numeric")
            if not math.isfinite(float(self.score)):
                raise ContractValidationError("score must be finite")
            object.__setattr__(self, "score", float(self.score))
        refs = tuple(self.evidence_event_ids)
        for index, value in enumerate(refs):
            required_text(value, f"evidence_event_ids[{index}]")
        if len(refs) != len(set(refs)):
            raise ContractValidationError("evidence_event_ids must be unique")
        object.__setattr__(self, "evidence_event_ids", refs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_status": self.task_status.value,
            "execution_validity": self.execution_validity.value,
            "verifier_status": self.verifier_status.value,
            "score": self.score,
            "evidence_event_ids": list(self.evidence_event_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodeOutcome":
        fields = {
            "task_status",
            "execution_validity",
            "verifier_status",
            "score",
            "evidence_event_ids",
        }
        payload = strict_fields(value, fields, "EpisodeOutcome")
        return cls(
            task_status=enum_member(payload["task_status"], TaskStatus, "task_status"),
            execution_validity=enum_member(
                payload["execution_validity"], ExecutionValidity, "execution_validity"
            ),
            verifier_status=enum_member(
                payload["verifier_status"], EpisodeVerifierStatus, "verifier_status"
            ),
            score=payload["score"],
            evidence_event_ids=tuple(payload["evidence_event_ids"]),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeTermination:
    reason: str
    observable: bool
    source_event_id: str | None = None

    def __post_init__(self) -> None:
        required_text(self.reason, "termination.reason")
        if not isinstance(self.observable, bool):
            raise ContractValidationError("termination.observable must be boolean")
        optional_text(self.source_event_id, "termination.source_event_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "observable": self.observable,
            "source_event_id": self.source_event_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodeTermination":
        fields = {"reason", "observable", "source_event_id"}
        return cls(**strict_fields(value, fields, "EpisodeTermination"))


@dataclass(frozen=True, slots=True, kw_only=True)
class IntegrityReport:
    state: IntegrityState
    duplicate_event_count: int
    sequence_gaps: tuple[int, ...]
    orphan_span_ids: tuple[str, ...]
    missing_expected_event_types: tuple[EventType, ...]
    assembler_version: str
    input_checksum: str
    output_checksum: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.state, IntegrityState):
            raise ContractValidationError("integrity.state must be an IntegrityState")
        if (
            isinstance(self.duplicate_event_count, bool)
            or not isinstance(self.duplicate_event_count, int)
            or self.duplicate_event_count < 0
        ):
            raise ContractValidationError(
                "integrity.duplicate_event_count must be non-negative"
            )
        gaps = tuple(self.sequence_gaps)
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in gaps):
            raise ContractValidationError("integrity.sequence_gaps must be non-negative integers")
        if tuple(sorted(set(gaps))) != gaps:
            raise ContractValidationError("integrity.sequence_gaps must be sorted and unique")
        object.__setattr__(self, "sequence_gaps", gaps)
        orphans = tuple(self.orphan_span_ids)
        for index, value in enumerate(orphans):
            required_text(value, f"integrity.orphan_span_ids[{index}]")
        object.__setattr__(self, "orphan_span_ids", orphans)
        missing = tuple(self.missing_expected_event_types)
        if any(not isinstance(item, EventType) for item in missing):
            raise ContractValidationError(
                "integrity.missing_expected_event_types must contain EventType values"
            )
        object.__setattr__(self, "missing_expected_event_types", missing)
        required_text(self.assembler_version, "integrity.assembler_version")
        validate_sha256(self.input_checksum, "integrity.input_checksum")
        validate_sha256(self.output_checksum, "integrity.output_checksum")
        warnings = tuple(self.warnings)
        for index, value in enumerate(warnings):
            required_text(value, f"integrity.warnings[{index}]")
        object.__setattr__(self, "warnings", warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "duplicate_event_count": self.duplicate_event_count,
            "sequence_gaps": list(self.sequence_gaps),
            "orphan_span_ids": list(self.orphan_span_ids),
            "missing_expected_event_types": [item.value for item in self.missing_expected_event_types],
            "assembler_version": self.assembler_version,
            "input_checksum": self.input_checksum,
            "output_checksum": self.output_checksum,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IntegrityReport":
        fields = {
            "state",
            "duplicate_event_count",
            "sequence_gaps",
            "orphan_span_ids",
            "missing_expected_event_types",
            "assembler_version",
            "input_checksum",
            "output_checksum",
            "warnings",
        }
        payload = strict_fields(value, fields, "IntegrityReport")
        return cls(
            state=enum_member(payload["state"], IntegrityState, "integrity.state"),
            duplicate_event_count=payload["duplicate_event_count"],
            sequence_gaps=tuple(payload["sequence_gaps"]),
            orphan_span_ids=tuple(payload["orphan_span_ids"]),
            missing_expected_event_types=tuple(
                enum_member(item, EventType, "missing_expected_event_types")
                for item in payload["missing_expected_event_types"]
            ),
            assembler_version=payload["assembler_version"],
            input_checksum=payload["input_checksum"],
            output_checksum=payload["output_checksum"],
            warnings=tuple(payload["warnings"]),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentEpisode:
    episode_id: str
    run_id: str
    task_id: str
    attempt: int
    harness_manifest: HarnessManifest
    model_manifest: ModelManifest
    environment_manifest: EnvironmentManifest
    evaluator_manifest: EvaluatorManifest
    experiment_manifest_ref: str
    capabilities: frozenset[CaptureCapability]
    events: tuple[TraceEvent, ...]
    artifact_refs: tuple[ArtifactRef, ...]
    started_at: str
    ended_at: str
    outcome: EpisodeOutcome
    termination: EpisodeTermination
    integrity: IntegrityReport
    source_lineage: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("episode_id", "run_id", "task_id", "experiment_manifest_ref"):
            required_text(getattr(self, name), name)
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 1:
            raise ContractValidationError("attempt must be a positive integer")
        if self.schema_version != SCHEMA_VERSION:
            raise ContractValidationError(f"schema_version must be {SCHEMA_VERSION!r}")
        for name, manifest_type in (
            ("harness_manifest", HarnessManifest),
            ("model_manifest", ModelManifest),
            ("environment_manifest", EnvironmentManifest),
            ("evaluator_manifest", EvaluatorManifest),
        ):
            if not isinstance(getattr(self, name), manifest_type):
                raise ContractValidationError(f"{name} must be a {manifest_type.__name__}")

        capabilities = frozenset(self.capabilities)
        if any(not isinstance(item, CaptureCapability) for item in capabilities):
            raise ContractValidationError("capabilities must contain CaptureCapability values")
        object.__setattr__(self, "capabilities", capabilities)

        events = tuple(self.events)
        if any(not isinstance(event, TraceEvent) for event in events):
            raise ContractValidationError("events must contain TraceEvent values")
        expected_order = tuple(
            sorted(events, key=lambda item: (item.sequence, item.timestamp, item.event_id))
        )
        if events != expected_order:
            raise ContractValidationError("events must use deterministic assembler order")
        for event in events:
            if event.episode_id != self.episode_id or event.run_id != self.run_id:
                raise ContractValidationError("event identity does not match AgentEpisode")
        if len({event.event_id for event in events}) != len(events):
            raise ContractValidationError("events contain duplicate event_id values")
        object.__setattr__(self, "events", events)

        artifacts = tuple(self.artifact_refs)
        if any(not isinstance(item, ArtifactRef) for item in artifacts):
            raise ContractValidationError("artifact_refs must contain ArtifactRef values")
        if len({item.artifact_id for item in artifacts}) != len(artifacts):
            raise ContractValidationError("artifact_refs contain duplicate artifact_id values")
        object.__setattr__(self, "artifact_refs", artifacts)

        started = utc_instant(self.started_at, "started_at")
        ended = utc_instant(self.ended_at, "ended_at")
        if ended < started:
            raise ContractValidationError("ended_at cannot be earlier than started_at")
        if not isinstance(self.outcome, EpisodeOutcome):
            raise ContractValidationError("outcome must be an EpisodeOutcome")
        if not isinstance(self.termination, EpisodeTermination):
            raise ContractValidationError("termination must be an EpisodeTermination")
        if not isinstance(self.integrity, IntegrityReport):
            raise ContractValidationError("integrity must be an IntegrityReport")
        event_ids = {event.event_id for event in events}
        dangling = set(self.outcome.evidence_event_ids) - event_ids
        if dangling:
            raise ContractValidationError(f"outcome evidence references unknown events: {sorted(dangling)}")
        if self.termination.source_event_id not in event_ids | {None}:
            raise ContractValidationError("termination source_event_id is unknown")
        object.__setattr__(
            self, "source_lineage", frozen_object(self.source_lineage, "source_lineage")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "harness_manifest": self.harness_manifest.to_dict(),
            "model_manifest": self.model_manifest.to_dict(),
            "environment_manifest": self.environment_manifest.to_dict(),
            "evaluator_manifest": self.evaluator_manifest.to_dict(),
            "experiment_manifest_ref": self.experiment_manifest_ref,
            "capabilities": sorted(item.value for item in self.capabilities),
            "events": [event.to_dict() for event in self.events],
            "artifact_refs": [artifact.to_dict() for artifact in self.artifact_refs],
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "outcome": self.outcome.to_dict(),
            "termination": self.termination.to_dict(),
            "integrity": self.integrity.to_dict(),
            "source_lineage": thaw_json(self.source_lineage),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AgentEpisode":
        fields = {
            "schema_version",
            "episode_id",
            "run_id",
            "task_id",
            "attempt",
            "harness_manifest",
            "model_manifest",
            "environment_manifest",
            "evaluator_manifest",
            "experiment_manifest_ref",
            "capabilities",
            "events",
            "artifact_refs",
            "started_at",
            "ended_at",
            "outcome",
            "termination",
            "integrity",
            "source_lineage",
        }
        payload = strict_fields(value, fields, "AgentEpisode")
        return cls(
            schema_version=payload["schema_version"],
            episode_id=payload["episode_id"],
            run_id=payload["run_id"],
            task_id=payload["task_id"],
            attempt=payload["attempt"],
            harness_manifest=HarnessManifest.from_dict(payload["harness_manifest"]),
            model_manifest=ModelManifest.from_dict(payload["model_manifest"]),
            environment_manifest=EnvironmentManifest.from_dict(
                payload["environment_manifest"]
            ),
            evaluator_manifest=EvaluatorManifest.from_dict(payload["evaluator_manifest"]),
            experiment_manifest_ref=payload["experiment_manifest_ref"],
            capabilities=frozenset(
                enum_member(item, CaptureCapability, "capabilities")
                for item in payload["capabilities"]
            ),
            events=tuple(TraceEvent.from_dict(item) for item in payload["events"]),
            artifact_refs=tuple(
                ArtifactRef.from_dict(item) for item in payload["artifact_refs"]
            ),
            started_at=payload["started_at"],
            ended_at=payload["ended_at"],
            outcome=EpisodeOutcome.from_dict(payload["outcome"]),
            termination=EpisodeTermination.from_dict(payload["termination"]),
            integrity=IntegrityReport.from_dict(payload["integrity"]),
            source_lineage=payload["source_lineage"],
        )

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())
