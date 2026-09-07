"""Fail-closed phase contract for one-GPU rollout/train/evaluate cycles."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping

from src.contracts._json import sha256_json, validate_sha256
from src.contracts._validation import required_text, strict_fields
from src.errors import ContractValidationError

SINGLE_GPU_CYCLE_VERSION = "single-gpu-cycle/v1"


class CyclePhase(str, Enum):
    CREATED = "CREATED"
    ROLLOUT_RUNNING = "ROLLOUT_RUNNING"
    ROLLOUT_FROZEN = "ROLLOUT_FROZEN"
    DATASET_CERTIFIED = "DATASET_CERTIFIED"
    TRAINING_RUNNING = "TRAINING_RUNNING"
    CHECKPOINT_READY = "CHECKPOINT_READY"
    EVALUATION_RUNNING = "EVALUATION_RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class GpuOwner(str, Enum):
    NONE = "NONE"
    SERVING = "SERVING"
    TRAINING = "TRAINING"


_NEXT_PHASE = {
    CyclePhase.CREATED: CyclePhase.ROLLOUT_RUNNING,
    CyclePhase.ROLLOUT_RUNNING: CyclePhase.ROLLOUT_FROZEN,
    CyclePhase.ROLLOUT_FROZEN: CyclePhase.DATASET_CERTIFIED,
    CyclePhase.DATASET_CERTIFIED: CyclePhase.TRAINING_RUNNING,
    CyclePhase.TRAINING_RUNNING: CyclePhase.CHECKPOINT_READY,
    CyclePhase.CHECKPOINT_READY: CyclePhase.EVALUATION_RUNNING,
    CyclePhase.EVALUATION_RUNNING: CyclePhase.COMPLETE,
}

_GPU_OWNER = {
    CyclePhase.CREATED: GpuOwner.NONE,
    CyclePhase.ROLLOUT_RUNNING: GpuOwner.SERVING,
    CyclePhase.ROLLOUT_FROZEN: GpuOwner.NONE,
    CyclePhase.DATASET_CERTIFIED: GpuOwner.NONE,
    CyclePhase.TRAINING_RUNNING: GpuOwner.TRAINING,
    CyclePhase.CHECKPOINT_READY: GpuOwner.NONE,
    CyclePhase.EVALUATION_RUNNING: GpuOwner.SERVING,
    CyclePhase.COMPLETE: GpuOwner.NONE,
    CyclePhase.FAILED: GpuOwner.NONE,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class SingleGpuCycleState:
    """Immutable state and evidence handoff for one policy update cycle."""

    cycle_id: str
    phase: CyclePhase
    baseline_policy_fingerprint: str
    rollout_bundle_manifest_checksum: str | None = None
    dataset_manifest_checksum: str | None = None
    candidate_policy_fingerprint: str | None = None
    checkpoint_checksum: str | None = None
    evaluation_report_checksum: str | None = None
    failure_report_checksum: str | None = None
    schema_version: str = SINGLE_GPU_CYCLE_VERSION

    def __post_init__(self) -> None:
        required_text(self.cycle_id, "cycle_id")
        if not isinstance(self.phase, CyclePhase):
            raise ContractValidationError("phase must be a CyclePhase")
        for name in (
            "baseline_policy_fingerprint",
            "rollout_bundle_manifest_checksum",
            "dataset_manifest_checksum",
            "candidate_policy_fingerprint",
            "checkpoint_checksum",
            "evaluation_report_checksum",
            "failure_report_checksum",
        ):
            validate_sha256(getattr(self, name), name)
        if self.schema_version != SINGLE_GPU_CYCLE_VERSION:
            raise ContractValidationError(
                f"schema_version must be {SINGLE_GPU_CYCLE_VERSION!r}"
            )
        self._validate_phase_evidence()

    @classmethod
    def create(cls, *, cycle_id: str, baseline_policy_fingerprint: str) -> "SingleGpuCycleState":
        return cls(
            cycle_id=cycle_id,
            phase=CyclePhase.CREATED,
            baseline_policy_fingerprint=baseline_policy_fingerprint,
        )

    @property
    def gpu_owner(self) -> GpuOwner:
        return _GPU_OWNER[self.phase]

    @property
    def next_phase(self) -> CyclePhase | None:
        return _NEXT_PHASE.get(self.phase)

    def advance(
        self,
        next_phase: CyclePhase,
        *,
        evidence_checksum: str | None = None,
        candidate_policy_fingerprint: str | None = None,
    ) -> "SingleGpuCycleState":
        """Advance exactly one phase, attaching the required immutable evidence."""

        if not isinstance(next_phase, CyclePhase):
            raise TypeError("next_phase must be a CyclePhase")
        if self.phase in (CyclePhase.COMPLETE, CyclePhase.FAILED):
            raise ContractValidationError("terminal cycle cannot advance")
        if next_phase is CyclePhase.FAILED:
            validate_sha256(evidence_checksum, "failure evidence_checksum")
            if evidence_checksum is None:
                raise ContractValidationError("FAILED transition requires failure evidence")
            return replace(
                self,
                phase=next_phase,
                failure_report_checksum=evidence_checksum,
            )
        expected = _NEXT_PHASE[self.phase]
        if next_phase is not expected:
            raise ContractValidationError(
                f"illegal single-GPU phase transition {self.phase.value} -> {next_phase.value}"
            )

        updates: dict[str, Any] = {"phase": next_phase}
        evidence_field = {
            CyclePhase.ROLLOUT_FROZEN: "rollout_bundle_manifest_checksum",
            CyclePhase.DATASET_CERTIFIED: "dataset_manifest_checksum",
            CyclePhase.CHECKPOINT_READY: "checkpoint_checksum",
            CyclePhase.COMPLETE: "evaluation_report_checksum",
        }.get(next_phase)
        if evidence_field is not None:
            validate_sha256(evidence_checksum, "evidence_checksum")
            if evidence_checksum is None:
                raise ContractValidationError(
                    f"{next_phase.value} transition requires evidence_checksum"
                )
            updates[evidence_field] = evidence_checksum
        elif evidence_checksum is not None:
            raise ContractValidationError(
                f"{next_phase.value} transition does not accept evidence_checksum"
            )

        if next_phase is CyclePhase.CHECKPOINT_READY:
            validate_sha256(candidate_policy_fingerprint, "candidate_policy_fingerprint")
            if candidate_policy_fingerprint is None:
                raise ContractValidationError(
                    "CHECKPOINT_READY requires candidate_policy_fingerprint"
                )
            if candidate_policy_fingerprint == self.baseline_policy_fingerprint:
                raise ContractValidationError(
                    "candidate policy fingerprint must differ from baseline"
                )
            updates["candidate_policy_fingerprint"] = candidate_policy_fingerprint
        elif candidate_policy_fingerprint is not None:
            raise ContractValidationError(
                f"{next_phase.value} transition does not accept candidate policy"
            )
        return replace(self, **updates)

    def _validate_phase_evidence(self) -> None:
        rank = list(CyclePhase).index(self.phase)
        required_after = {
            "rollout_bundle_manifest_checksum": CyclePhase.ROLLOUT_FROZEN,
            "dataset_manifest_checksum": CyclePhase.DATASET_CERTIFIED,
            "candidate_policy_fingerprint": CyclePhase.CHECKPOINT_READY,
            "checkpoint_checksum": CyclePhase.CHECKPOINT_READY,
            "evaluation_report_checksum": CyclePhase.COMPLETE,
        }
        if self.phase is CyclePhase.FAILED:
            if self.failure_report_checksum is None:
                raise ContractValidationError("FAILED state requires failure report")
            return
        if self.failure_report_checksum is not None:
            raise ContractValidationError("non-FAILED state cannot carry failure report")
        phases = list(CyclePhase)
        for field_name, required_phase in required_after.items():
            value = getattr(self, field_name)
            required = rank >= phases.index(required_phase)
            if required and value is None:
                raise ContractValidationError(
                    f"phase {self.phase.value} requires {field_name}"
                )
            if not required and value is not None:
                raise ContractValidationError(
                    f"phase {self.phase.value} cannot carry {field_name}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cycle_id": self.cycle_id,
            "phase": self.phase.value,
            "gpu_owner": self.gpu_owner.value,
            "baseline_policy_fingerprint": self.baseline_policy_fingerprint,
            "rollout_bundle_manifest_checksum": self.rollout_bundle_manifest_checksum,
            "dataset_manifest_checksum": self.dataset_manifest_checksum,
            "candidate_policy_fingerprint": self.candidate_policy_fingerprint,
            "checkpoint_checksum": self.checkpoint_checksum,
            "evaluation_report_checksum": self.evaluation_report_checksum,
            "failure_report_checksum": self.failure_report_checksum,
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SingleGpuCycleState":
        payload = strict_fields(
            value,
            {
                "schema_version",
                "cycle_id",
                "phase",
                "gpu_owner",
                "baseline_policy_fingerprint",
                "rollout_bundle_manifest_checksum",
                "dataset_manifest_checksum",
                "candidate_policy_fingerprint",
                "checkpoint_checksum",
                "evaluation_report_checksum",
                "failure_report_checksum",
            },
            "SingleGpuCycleState",
        )
        phase = CyclePhase(payload["phase"])
        if payload["gpu_owner"] != _GPU_OWNER[phase].value:
            raise ContractValidationError("serialized gpu_owner disagrees with phase")
        return cls(
            schema_version=payload["schema_version"],
            cycle_id=payload["cycle_id"],
            phase=phase,
            baseline_policy_fingerprint=payload["baseline_policy_fingerprint"],
            rollout_bundle_manifest_checksum=payload[
                "rollout_bundle_manifest_checksum"
            ],
            dataset_manifest_checksum=payload["dataset_manifest_checksum"],
            candidate_policy_fingerprint=payload["candidate_policy_fingerprint"],
            checkpoint_checksum=payload["checkpoint_checksum"],
            evaluation_report_checksum=payload["evaluation_report_checksum"],
            failure_report_checksum=payload["failure_report_checksum"],
        )
