"""Deterministic quality summaries over producer and certification evidence."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from src.certification import ConsumerVerdict, EligibilityDecision
from src.contracts._json import sha256_json
from src.producers import ProducerArtifact, ProducerCapability

DATA_QUALITY_REPORT_VERSION = "data-quality-report/v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class DataQualityReport:
    artifact_count: int
    decision_count: int
    producer_status_counts: tuple[tuple[str, int], ...]
    capability_counts: tuple[tuple[str, int], ...]
    decision_verdict_counts: tuple[tuple[str, int], ...]
    rejection_reason_counts: tuple[tuple[str, int], ...]
    duplicate_artifact_count: int
    orphan_decision_count: int
    schema_version: str = DATA_QUALITY_REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_count": self.artifact_count,
            "decision_count": self.decision_count,
            "producer_status_counts": dict(self.producer_status_counts),
            "capability_counts": dict(self.capability_counts),
            "decision_verdict_counts": dict(self.decision_verdict_counts),
            "rejection_reason_counts": dict(self.rejection_reason_counts),
            "duplicate_artifact_count": self.duplicate_artifact_count,
            "orphan_decision_count": self.orphan_decision_count,
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


def build_data_quality_report(
    artifacts: Iterable[ProducerArtifact],
    decisions: Iterable[EligibilityDecision],
) -> DataQualityReport:
    artifact_values = tuple(artifacts)
    decision_values = tuple(decisions)
    if any(not isinstance(item, ProducerArtifact) for item in artifact_values):
        raise TypeError("artifacts must contain ProducerArtifact values")
    if any(not isinstance(item, EligibilityDecision) for item in decision_values):
        raise TypeError("decisions must contain EligibilityDecision values")

    artifact_checksums = [item.checksum for item in artifact_values]
    artifact_episode_ids = {item.identity.episode_id for item in artifact_values}
    status_counts = Counter(item.status.value for item in artifact_values)
    capability_counts = Counter(
        capability.value
        for artifact in artifact_values
        for capability in artifact.capabilities
    )
    verdict_counts = Counter(item.verdict.value for item in decision_values)
    reason_counts = Counter(
        reason
        for decision in decision_values
        if decision.verdict is not ConsumerVerdict.ELIGIBLE
        for reason in decision.reasons
    )
    return DataQualityReport(
        artifact_count=len(artifact_values),
        decision_count=len(decision_values),
        producer_status_counts=_sorted_counts(status_counts),
        capability_counts=tuple(
            (capability.value, capability_counts.get(capability.value, 0))
            for capability in ProducerCapability
        ),
        decision_verdict_counts=_sorted_counts(verdict_counts),
        rejection_reason_counts=_sorted_counts(reason_counts),
        duplicate_artifact_count=len(artifact_checksums) - len(set(artifact_checksums)),
        orphan_decision_count=sum(
            decision.episode_id not in artifact_episode_ids for decision in decision_values
        ),
    )


def _sorted_counts(counter: Counter[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(counter.items()))
