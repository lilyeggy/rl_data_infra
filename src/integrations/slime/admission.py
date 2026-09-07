"""Revalidate certified manifests before handing trajectories to Slime."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.certification import ConsumerProfile, ConsumerVerdict, EligibilityDecision
from src.contracts._json import sha256_json
from src.contracts.dataset import DatasetManifest, DatasetPurpose, DatasetRole
from src.contracts.execution_bundle import ExecutionBundle
from src.errors import ContractValidationError
from src.producers.base import (
    ProducerArtifact,
    ProducerCapability,
    ProducerExecutionStatus,
)

SLIME_ADMISSION_VERSION = "slime-admission/v1"

_REQUIRED_CAPABILITIES = frozenset(
    {
        ProducerCapability.TOKEN_IDS,
        ProducerCapability.ACTION_MASK,
        ProducerCapability.BEHAVIOR_LOGPROBS,
        ProducerCapability.POLICY_VERSION,
        ProducerCapability.VERIFIER_EVIDENCE,
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class AdmittedPolicyTrace:
    trajectory_id: str
    trace_index: int
    group_id: str
    policy_fingerprint: str
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    response_logprobs: tuple[float, ...]
    reward: float
    member_id: str
    execution_bundle_checksum: str
    policy_artifact_checksum: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "trace_index": self.trace_index,
            "group_id": self.group_id,
            "policy_fingerprint": self.policy_fingerprint,
            "prompt_ids": list(self.prompt_ids),
            "response_ids": list(self.response_ids),
            "loss_mask": list(self.loss_mask),
            "response_logprobs": list(self.response_logprobs),
            "reward": self.reward,
            "member_id": self.member_id,
            "execution_bundle_checksum": self.execution_bundle_checksum,
            "policy_artifact_checksum": self.policy_artifact_checksum,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SlimeAdmissionBatch:
    dataset_manifest_checksum: str
    policy_fingerprint: str
    traces: tuple[AdmittedPolicyTrace, ...]
    trajectory_count: int
    group_sizes: Mapping[str, int]
    schema_version: str = SLIME_ADMISSION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_manifest_checksum": self.dataset_manifest_checksum,
            "policy_fingerprint": self.policy_fingerprint,
            "traces": [trace.to_dict() for trace in self.traces],
            "trajectory_count": self.trajectory_count,
            "group_sizes": dict(sorted(self.group_sizes.items())),
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


def admit_on_policy_manifest(
    manifest: DatasetManifest,
    *,
    decisions_by_checksum: Mapping[str, EligibilityDecision],
    bundles_by_checksum: Mapping[str, ExecutionBundle],
    artifacts_by_checksum: Mapping[str, ProducerArtifact],
    minimum_group_size: int = 2,
) -> SlimeAdmissionBatch:
    """Create a token-faithful admission batch without importing Slime.

    Group size counts unique trajectories/sessions, not the number of traces a
    trajectory builder emitted. The upstream bridge remains responsible for
    converting this admitted evidence into its pinned Slime ``Sample`` type.
    """

    if not isinstance(manifest, DatasetManifest):
        raise TypeError("manifest must be a DatasetManifest")
    if manifest.purpose is not DatasetPurpose.ON_POLICY_RL:
        raise ContractValidationError("Slime GRPO admission requires ON_POLICY_RL manifest")
    if (
        isinstance(minimum_group_size, bool)
        or not isinstance(minimum_group_size, int)
        or minimum_group_size < 2
    ):
        raise ContractValidationError("minimum_group_size must be an integer >= 2")

    admitted: list[AdmittedPolicyTrace] = []
    trajectories_by_group: dict[str, set[str]] = {}
    policy_fingerprints: set[str] = set()
    for member in manifest.members:
        if member.role is not DatasetRole.TRAJECTORY:
            raise ContractValidationError("on-policy member role must be TRAJECTORY")
        decision = decisions_by_checksum.get(member.certification_checksum)
        if decision is None or decision.checksum != member.certification_checksum:
            raise ContractValidationError("certification decision is missing or changed")
        if (
            decision.profile is not ConsumerProfile.ON_POLICY_RL
            or decision.verdict is not ConsumerVerdict.ELIGIBLE
        ):
            raise ContractValidationError("member lacks ELIGIBLE ON_POLICY_RL decision")
        if decision.episode_id != member.identity.episode_id:
            raise ContractValidationError("decision/member episode identity mismatch")
        if decision.episode_checksum != member.episode_checksum:
            raise ContractValidationError("decision/member episode checksum mismatch")

        bundle_checksum = member.execution_bundle_checksum
        artifact_checksum = member.policy_artifact_checksum
        if bundle_checksum is None or artifact_checksum is None:
            raise ContractValidationError("RL member lacks bundle or policy artifact checksum")
        if decision.execution_bundle_checksum != bundle_checksum:
            raise ContractValidationError("decision/member bundle checksum mismatch")
        if decision.policy_artifact_checksum != artifact_checksum:
            raise ContractValidationError("decision/member policy checksum mismatch")

        bundle = bundles_by_checksum.get(bundle_checksum)
        artifact = artifacts_by_checksum.get(artifact_checksum)
        if bundle is None or bundle.checksum != bundle_checksum:
            raise ContractValidationError("ExecutionBundle is missing or changed")
        if artifact is None or artifact.checksum != artifact_checksum:
            raise ContractValidationError("policy artifact is missing or changed")
        if bundle.identity != member.identity or artifact.identity != member.identity:
            raise ContractValidationError("bundle/artifact/member identity mismatch")
        if bundle.episode_checksum != member.episode_checksum:
            raise ContractValidationError("bundle/member episode checksum mismatch")
        if artifact.checksum not in bundle.policy_trace_checksums:
            raise ContractValidationError("policy artifact is not bound by bundle")
        if artifact.status is not ProducerExecutionStatus.COMPLETED:
            raise ContractValidationError("policy artifact execution is not completed")
        missing = _REQUIRED_CAPABILITIES - artifact.capabilities
        if missing:
            raise ContractValidationError(
                "policy artifact lacks: "
                + ", ".join(sorted(item.value for item in missing))
            )

        identity = member.identity
        if identity.group_id is None or identity.policy_fingerprint is None:
            raise ContractValidationError("on-policy identity requires group and policy")
        policy_fingerprints.add(identity.policy_fingerprint)
        trajectories_by_group.setdefault(identity.group_id, set()).add(identity.episode_id)
        admitted.extend(
            _admit_artifact_traces(
                artifact,
                member_id=member.member_id,
                bundle_checksum=bundle_checksum,
            )
        )

    if len(policy_fingerprints) != 1:
        raise ContractValidationError("admission batch mixes behavior policies")
    group_sizes = {
        group_id: len(trajectory_ids)
        for group_id, trajectory_ids in trajectories_by_group.items()
    }
    undersized = {
        group_id: size for group_id, size in group_sizes.items() if size < minimum_group_size
    }
    if undersized:
        raise ContractValidationError(f"undersized rollout groups: {undersized}")
    if not admitted:
        raise ContractValidationError("admission batch contains no trainable traces")
    return SlimeAdmissionBatch(
        dataset_manifest_checksum=manifest.checksum,
        policy_fingerprint=next(iter(policy_fingerprints)),
        traces=tuple(
            sorted(admitted, key=lambda item: (item.trajectory_id, item.trace_index))
        ),
        trajectory_count=sum(group_sizes.values()),
        group_sizes=group_sizes,
    )


def _admit_artifact_traces(
    artifact: ProducerArtifact,
    *,
    member_id: str,
    bundle_checksum: str,
) -> tuple[AdmittedPolicyTrace, ...]:
    trajectory = artifact.payload.get("trajectory")
    if not isinstance(trajectory, Mapping):
        raise ContractValidationError("policy artifact trajectory must be an object")
    raw_traces = trajectory.get("traces")
    if (
        not isinstance(raw_traces, Sequence)
        or isinstance(raw_traces, (str, bytes))
        or not raw_traces
    ):
        raise ContractValidationError("policy artifact trajectory has no traces")

    output = []
    for index, raw_trace in enumerate(raw_traces):
        if not isinstance(raw_trace, Mapping):
            raise ContractValidationError(f"trace[{index}] must be an object")
        prompt_ids = _int_array(raw_trace.get("prompt_ids", ()), f"trace[{index}].prompt_ids")
        response_ids = _int_array(
            raw_trace.get("response_ids"), f"trace[{index}].response_ids"
        )
        loss_mask = _int_array(raw_trace.get("loss_mask"), f"trace[{index}].loss_mask")
        if not response_ids or len(loss_mask) != len(response_ids):
            raise ContractValidationError(f"trace[{index}] response/mask alignment invalid")
        if any(value not in (0, 1) for value in loss_mask) or not any(loss_mask):
            raise ContractValidationError(f"trace[{index}] has no valid trainable mask")
        raw_logprobs = raw_trace.get("response_logprobs")
        if not isinstance(raw_logprobs, Sequence) or isinstance(raw_logprobs, (str, bytes)):
            raise ContractValidationError(f"trace[{index}] logprobs must be an array")
        logprobs = tuple(_finite_number(item, f"trace[{index}].logprobs") for item in raw_logprobs)
        if len(logprobs) != len(response_ids):
            raise ContractValidationError(f"trace[{index}] response/logprob alignment invalid")
        reward = _finite_number(raw_trace.get("reward"), f"trace[{index}].reward")
        identity = artifact.identity
        assert identity.group_id is not None
        assert identity.policy_fingerprint is not None
        output.append(
            AdmittedPolicyTrace(
                trajectory_id=identity.episode_id,
                trace_index=index,
                group_id=identity.group_id,
                policy_fingerprint=identity.policy_fingerprint,
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                loss_mask=loss_mask,
                response_logprobs=logprobs,
                reward=reward,
                member_id=member_id,
                execution_bundle_checksum=bundle_checksum,
                policy_artifact_checksum=artifact.checksum,
            )
        )
    return tuple(output)


def _int_array(value: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContractValidationError(f"{field_name} must be an array")
    output = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ContractValidationError(f"{field_name} must contain integers")
        output.append(item)
    return tuple(output)


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractValidationError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ContractValidationError(f"{field_name} must be finite")
    return number
