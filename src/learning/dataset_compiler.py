"""Compile immutable manifests from consumer-certified learning artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from src.certification import ConsumerProfile, ConsumerVerdict, EligibilityDecision
from src.contracts.dataset import (
    DatasetManifest,
    DatasetMember,
    DatasetPurpose,
    DatasetRole,
    DatasetSplit,
)
from src.contracts.execution_identity import ExecutionIdentity
from src.errors import ContractValidationError


@dataclass(frozen=True, slots=True, kw_only=True)
class CertifiedArtifact:
    identity: ExecutionIdentity
    decision: EligibilityDecision
    artifact_checksum: str
    split: DatasetSplit
    role: DatasetRole


_PURPOSE_PROFILE = {
    DatasetPurpose.HARNESS_REGRESSION: ConsumerProfile.HARNESS_ANALYSIS,
    DatasetPurpose.EVALUATION: ConsumerProfile.EVALUATION,
    DatasetPurpose.SFT: ConsumerProfile.SFT,
    DatasetPurpose.PREFERENCE: ConsumerProfile.PREFERENCE,
    DatasetPurpose.OFF_POLICY_RL: ConsumerProfile.OFF_POLICY_RL,
    DatasetPurpose.ON_POLICY_RL: ConsumerProfile.ON_POLICY_RL,
}


def compile_dataset(
    *,
    dataset_id: str,
    revision: str,
    purpose: DatasetPurpose,
    selection_policy_version: str,
    artifacts: tuple[CertifiedArtifact, ...],
) -> DatasetManifest:
    """Compile a manifest; reject any mismatched or non-eligible artifact."""

    expected_profile = _PURPOSE_PROFILE[purpose]
    members: list[DatasetMember] = []
    for artifact in artifacts:
        decision = artifact.decision
        if decision.verdict is not ConsumerVerdict.ELIGIBLE:
            raise ContractValidationError(
                f"episode {decision.episode_id!r} is not ELIGIBLE"
            )
        if decision.profile is not expected_profile:
            raise ContractValidationError(
                f"certification profile {decision.profile.value} cannot build "
                f"a {purpose.value} dataset"
            )
        if artifact.identity.episode_id != decision.episode_id:
            raise ContractValidationError("identity and certification episode_id mismatch")
        if purpose in (DatasetPurpose.OFF_POLICY_RL, DatasetPurpose.ON_POLICY_RL):
            if (
                decision.execution_bundle_checksum is None
                or decision.policy_artifact_checksum is None
            ):
                raise ContractValidationError(
                    "RL dataset member requires bundle-bound policy certification"
                )
            if artifact.artifact_checksum != decision.policy_artifact_checksum:
                raise ContractValidationError(
                    "RL artifact checksum does not match certified policy artifact"
                )
        members.append(
            DatasetMember(
                identity=artifact.identity,
                episode_checksum=decision.episode_checksum,
                certification_checksum=decision.checksum,
                artifact_checksum=artifact.artifact_checksum,
                split=artifact.split,
                role=artifact.role,
                execution_bundle_checksum=decision.execution_bundle_checksum,
                policy_artifact_checksum=decision.policy_artifact_checksum,
            )
        )
    return DatasetManifest(
        dataset_id=dataset_id,
        revision=revision,
        purpose=purpose,
        selection_policy_version=selection_policy_version,
        members=tuple(members),
    )
