"""Revalidate certified manifests before handing sequences to verl."""

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
from src.integrations.verl.sequence import SEQUENCE_ASSEMBLER_VERSION, AssembledSequence
from src.producers.base import (
    ProducerArtifact,
    ProducerCapability,
    ProducerExecutionStatus,
)

VERL_ADMISSION_VERSION = "verl-admission/v1"

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
class AdmittedVerlSequence:
    """One episode admitted as a single verl training sequence."""

    episode_id: str
    group_id: str
    policy_fingerprint: str
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    response_logprobs: tuple[float, ...]
    reward: float
    num_model_calls: int
    num_tool_rounds: int
    member_id: str
    execution_bundle_checksum: str
    policy_artifact_checksum: str
    sequence_schema_version: str = SEQUENCE_ASSEMBLER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "group_id": self.group_id,
            "policy_fingerprint": self.policy_fingerprint,
            "prompt_ids": list(self.prompt_ids),
            "response_ids": list(self.response_ids),
            "response_mask": list(self.response_mask),
            "response_logprobs": list(self.response_logprobs),
            "reward": self.reward,
            "num_model_calls": self.num_model_calls,
            "num_tool_rounds": self.num_tool_rounds,
            "member_id": self.member_id,
            "execution_bundle_checksum": self.execution_bundle_checksum,
            "policy_artifact_checksum": self.policy_artifact_checksum,
            "sequence_schema_version": self.sequence_schema_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AdmittedVerlSequence:
        """Rebuild one admitted sequence from its persisted mapping.

        Used when a batch is certified from on-disk episode evidence instead of
        in-process objects. Every invariant is re-checked on the way in, so a
        tampered or truncated evidence file fails closed rather than silently
        reaching the trainer.
        """
        if not isinstance(value, Mapping):
            raise ContractValidationError("admitted sequence must be an object")
        expected = {
            "episode_id",
            "group_id",
            "policy_fingerprint",
            "prompt_ids",
            "response_ids",
            "response_mask",
            "response_logprobs",
            "reward",
            "num_model_calls",
            "num_tool_rounds",
            "member_id",
            "execution_bundle_checksum",
            "policy_artifact_checksum",
            "sequence_schema_version",
        }
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        if missing or unknown:
            raise ContractValidationError(
                f"admitted sequence fields mismatch: missing={missing} unknown={unknown}"
            )
        schema_version = value["sequence_schema_version"]
        if schema_version != SEQUENCE_ASSEMBLER_VERSION:
            raise ContractValidationError(
                f"admitted sequence schema must be {SEQUENCE_ASSEMBLER_VERSION!r}"
            )
        prompt_ids = _int_array(value["prompt_ids"], "prompt_ids")
        response_ids = _int_array(value["response_ids"], "response_ids")
        response_mask = _int_array(value["response_mask"], "response_mask")
        response_logprobs = _float_array(value["response_logprobs"], "response_logprobs")
        if not prompt_ids or not response_ids:
            raise ContractValidationError("admitted sequence arrays must be non-empty")
        if len(response_ids) != len(response_mask) or len(response_ids) != len(response_logprobs):
            raise ContractValidationError("admitted sequence arrays are not aligned")
        if any(bit not in (0, 1) for bit in response_mask):
            raise ContractValidationError("response_mask must contain only 0 and 1")
        if not any(response_mask):
            raise ContractValidationError("admitted sequence has no trainable token")
        reward = value["reward"]
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
        ):
            raise ContractValidationError("admitted sequence reward must be finite")
        num_model_calls = _non_negative_int(value["num_model_calls"], "num_model_calls")
        num_tool_rounds = _non_negative_int(value["num_tool_rounds"], "num_tool_rounds")
        return cls(
            episode_id=_required_text(value["episode_id"], "episode_id"),
            group_id=_required_text(value["group_id"], "group_id"),
            policy_fingerprint=_required_text(value["policy_fingerprint"], "policy_fingerprint"),
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            reward=float(reward),
            num_model_calls=num_model_calls,
            num_tool_rounds=num_tool_rounds,
            member_id=_required_text(value["member_id"], "member_id"),
            execution_bundle_checksum=_required_text(
                value["execution_bundle_checksum"], "execution_bundle_checksum"
            ),
            policy_artifact_checksum=_required_text(
                value["policy_artifact_checksum"], "policy_artifact_checksum"
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class VerlAdmissionBatch:
    dataset_manifest_checksum: str
    policy_fingerprint: str
    sequences: tuple[AdmittedVerlSequence, ...]
    trajectory_count: int
    group_sizes: Mapping[str, int]
    schema_version: str = VERL_ADMISSION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_manifest_checksum": self.dataset_manifest_checksum,
            "policy_fingerprint": self.policy_fingerprint,
            "sequences": [item.to_dict() for item in self.sequences],
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
    minimum_group_size: int = 4,
) -> VerlAdmissionBatch:
    """Create an episode-faithful admission batch without importing verl.

    Group size counts unique episodes, never model calls inside one episode.
    The upstream bridge remains responsible for converting admitted sequences
    into the pinned verl ``AgentLoopOutput`` type.
    """

    if not isinstance(manifest, DatasetManifest):
        raise TypeError("manifest must be a DatasetManifest")
    if manifest.purpose is not DatasetPurpose.ON_POLICY_RL:
        raise ContractValidationError("verl GRPO admission requires ON_POLICY_RL manifest")
    if (
        isinstance(minimum_group_size, bool)
        or not isinstance(minimum_group_size, int)
        or minimum_group_size < 2
    ):
        raise ContractValidationError("minimum_group_size must be an integer >= 2")

    admitted: list[AdmittedVerlSequence] = []
    episodes_by_group: dict[str, set[str]] = {}
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
                "policy artifact lacks: " + ", ".join(sorted(item.value for item in missing))
            )

        identity = member.identity
        if identity.group_id is None or identity.policy_fingerprint is None:
            raise ContractValidationError("on-policy identity requires group and policy")
        policy_fingerprints.add(identity.policy_fingerprint)
        episodes_by_group.setdefault(identity.group_id, set()).add(identity.episode_id)
        admitted.append(
            _admit_episode_sequence(
                artifact,
                member_id=member.member_id,
                bundle_checksum=bundle_checksum,
            )
        )

    if len(policy_fingerprints) != 1:
        raise ContractValidationError("admission batch mixes behavior policies")
    group_sizes = {
        group_id: len(episode_ids) for group_id, episode_ids in episodes_by_group.items()
    }
    undersized = {
        group_id: size for group_id, size in group_sizes.items() if size < minimum_group_size
    }
    if undersized:
        raise ContractValidationError(f"undersized rollout groups: {undersized}")
    if not admitted:
        raise ContractValidationError("admission batch contains no trainable sequences")
    _require_reward_variance(admitted)
    return VerlAdmissionBatch(
        dataset_manifest_checksum=manifest.checksum,
        policy_fingerprint=next(iter(policy_fingerprints)),
        sequences=tuple(sorted(admitted, key=lambda item: item.episode_id)),
        trajectory_count=sum(group_sizes.values()),
        group_sizes=group_sizes,
    )


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractValidationError(f"{field_name} must be a non-empty string")
    return value


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractValidationError(f"{field_name} must be a non-negative integer")
    return value


def _int_array(value: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContractValidationError(f"{field_name} must be an array")
    output: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ContractValidationError(f"{field_name} must contain integers")
        output.append(item)
    return tuple(output)


def _float_array(value: Any, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContractValidationError(f"{field_name} must be an array")
    output: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ContractValidationError(f"{field_name} must contain numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ContractValidationError(f"{field_name} must contain finite numbers")
        output.append(number)
    return tuple(output)


def _require_reward_variance(sequences: Sequence[AdmittedVerlSequence]) -> None:
    rewards = {item.reward for item in sequences}
    if len(rewards) < 2:
        raise ContractValidationError(
            "admission batch has no intra-group reward variance; "
            "refusing to fabricate a learning signal"
        )


def _admit_episode_sequence(
    artifact: ProducerArtifact,
    *,
    member_id: str,
    bundle_checksum: str,
) -> AdmittedVerlSequence:
    trajectory = artifact.payload.get("trajectory")
    if not isinstance(trajectory, Mapping):
        raise ContractValidationError("policy artifact trajectory must be an object")
    sequence_raw = trajectory.get("verl_sequence")
    if not isinstance(sequence_raw, Mapping):
        raise ContractValidationError(
            "policy artifact lacks an episode-level verl_sequence; "
            "per-call traces are not trainable sequences"
        )
    try:
        sequence = AssembledSequence(
            episode_id=str(sequence_raw["episode_id"]),
            prompt_ids=tuple(sequence_raw["prompt_ids"]),
            response_ids=tuple(sequence_raw["response_ids"]),
            response_mask=tuple(sequence_raw["response_mask"]),
            response_logprobs=tuple(sequence_raw["response_logprobs"]),
            num_model_calls=int(sequence_raw["num_model_calls"]),
            num_tool_rounds=int(sequence_raw["num_tool_rounds"]),
        )
    except (KeyError, TypeError, ValueError, ContractValidationError) as exc:
        raise ContractValidationError(f"verl_sequence is malformed: {exc}") from exc
    if sequence.schema_version != SEQUENCE_ASSEMBLER_VERSION:
        raise ContractValidationError("verl_sequence schema version mismatch")
    if sequence.episode_id != artifact.identity.episode_id:
        raise ContractValidationError("verl_sequence/episode identity mismatch")
    if sequence.num_model_calls < 1:
        raise ContractValidationError("verl_sequence has no model calls")
    if sequence.num_tool_rounds < 1:
        raise ContractValidationError(
            "verl_sequence has no tool round; a qualifying trajectory must "
            "contain at least one real tool call, its result, and a follow-up "
            "model request"
        )
    if not (len(sequence.response_ids) == len(sequence.response_mask) == len(sequence.response_logprobs)):
        raise ContractValidationError("verl_sequence response/mask/logprob alignment invalid")
    if any(value not in (0, 1) for value in sequence.response_mask) or not any(
        sequence.response_mask
    ):
        raise ContractValidationError("verl_sequence has no valid trainable mask")
    for value in sequence.response_logprobs:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractValidationError("verl_sequence logprobs must be numeric")
        if not math.isfinite(float(value)):
            raise ContractValidationError("verl_sequence logprobs must be finite")
    reward_raw = trajectory.get("reward")
    if isinstance(reward_raw, bool) or not isinstance(reward_raw, (int, float)):
        raise ContractValidationError("verl_sequence reward must be numeric")
    reward = float(reward_raw)
    if not math.isfinite(reward):
        raise ContractValidationError("verl_sequence reward must be finite")
    identity = artifact.identity
    assert identity.group_id is not None
    assert identity.policy_fingerprint is not None
    return AdmittedVerlSequence(
        episode_id=sequence.episode_id,
        group_id=identity.group_id,
        policy_fingerprint=identity.policy_fingerprint,
        prompt_ids=sequence.prompt_ids,
        response_ids=sequence.response_ids,
        response_mask=sequence.response_mask,
        response_logprobs=tuple(float(value) for value in sequence.response_logprobs),
        reward=reward,
        num_model_calls=sequence.num_model_calls,
        num_tool_rounds=sequence.num_tool_rounds,
        member_id=member_id,
        execution_bundle_checksum=bundle_checksum,
        policy_artifact_checksum=artifact.checksum,
    )
