"""Batch certification for one verl GRPO update.

The manager never computes advantages, never syncs weights, and never trains.
It binds every admitted sequence to the current round policy, verifies the
whole batch before training, and emits an auditable lineage record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.contracts._json import sha256_json
from src.errors import ContractValidationError, DegenerateBatchError
from src.integrations.verl.admission import VERL_ADMISSION_VERSION, AdmittedVerlSequence
from src.training.policy_fingerprint import PolicyFingerprint

CERTIFIED_LOOP_MANAGER_VERSION = "verl-certified-loop-manager/v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class CertifiedBatch:
    """A training batch the manager approved for exactly one verl update."""

    batch_id: str
    policy_generation: str
    policy_fingerprint_checksum: str
    sequence_checksums: tuple[str, ...]
    group_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    schema_version: str = CERTIFIED_LOOP_MANAGER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "batch_id": self.batch_id,
            "policy_generation": self.policy_generation,
            "policy_fingerprint_checksum": self.policy_fingerprint_checksum,
            "sequence_checksums": list(self.sequence_checksums),
            "group_ids": list(self.group_ids),
            "task_ids": list(self.task_ids),
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True, kw_only=True)
class CertifiedAgentLoopManager:
    """Inject the round policy and certify complete batches for verl."""

    policy: PolicyFingerprint
    minimum_group_size: int = 4
    schema_version: str = CERTIFIED_LOOP_MANAGER_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.policy, PolicyFingerprint):
            raise ContractValidationError("policy must be a PolicyFingerprint")
        if not self.policy.is_complete():
            raise ContractValidationError(
                "manager policy is incomplete: "
                + ", ".join(self.policy.missing_fields())
            )
        if (
            isinstance(self.minimum_group_size, bool)
            or not isinstance(self.minimum_group_size, int)
            or self.minimum_group_size < 2
        ):
            raise ContractValidationError("minimum_group_size must be an integer >= 2")
        if self.schema_version != CERTIFIED_LOOP_MANAGER_VERSION:
            raise ContractValidationError(
                f"schema_version must be {CERTIFIED_LOOP_MANAGER_VERSION!r}"
            )

    def certify_batch(
        self,
        sequences: Sequence[AdmittedVerlSequence],
        *,
        batch_id: str,
        task_ids_by_episode: Mapping[str, str],
        admission_schema_version: str = VERL_ADMISSION_VERSION,
    ) -> CertifiedBatch:
        """Approve one batch for a single update; reject partial batches."""
        if not batch_id or not isinstance(batch_id, str):
            raise ContractValidationError("batch_id must be a non-empty string")
        if admission_schema_version != VERL_ADMISSION_VERSION:
            raise ContractValidationError("admission schema version mismatch")
        items = tuple(sequences)
        if not items:
            raise ContractValidationError("certify_batch received an empty batch")
        if len({item.policy_fingerprint for item in items}) != 1:
            raise ContractValidationError("batch mixes behavior policies")
        if next(iter({item.policy_fingerprint for item in items})) != self.policy.checksum():
            raise ContractValidationError("batch policy does not match round policy")
        episode_ids = [item.episode_id for item in items]
        if len(set(episode_ids)) != len(episode_ids):
            raise ContractValidationError("batch reuses an episode; one episode per sample")
        groups: dict[str, list[str]] = {}
        for item in items:
            groups.setdefault(item.group_id, []).append(item.episode_id)
        undersized = {
            group_id: len(episode_ids)
            for group_id, episode_ids in groups.items()
            if len(episode_ids) < self.minimum_group_size
        }
        if undersized:
            raise ContractValidationError(f"undersized rollout groups: {undersized}")
        task_ids: list[str] = []
        for item in items:
            task_id = task_ids_by_episode.get(item.episode_id)
            if not task_id or not isinstance(task_id, str):
                raise ContractValidationError(
                    f"episode {item.episode_id!r} lacks a task binding"
                )
            task_ids.append(task_id)
        for group_id, members in groups.items():
            if len({task_ids_by_episode[e] for e in members}) != 1:
                raise ContractValidationError(f"group {group_id!r} mixes tasks")
            if len({item.reward for item in items if item.group_id == group_id}) < 2:
                raise DegenerateBatchError(
                    f"group {group_id!r} has no intra-group reward variance"
                )
        return CertifiedBatch(
            batch_id=batch_id,
            policy_generation=str(self.policy.policy_generation),
            policy_fingerprint_checksum=self.policy.checksum(),
            sequence_checksums=tuple(sha256_json(item.to_dict()) for item in items),
            group_ids=tuple(sorted(groups)),
            task_ids=tuple(task_ids),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PiAgentLoopConfig:
    """Configuration entry point for the verl-backed Pi rollout path."""

    run_id: str
    policy: PolicyFingerprint
    max_model_requests_per_episode: int = 8
    episode_timeout_seconds: float = 480.0
    max_prompt_tokens: int = 4096
    max_response_tokens: int = 4096
    max_tokens_per_generation: int = 1024
    max_concurrent_episodes: int = 2
    sampling_temperature: float = 1.0
    sampling_top_p: float = 1.0
    random_seed: int = 0
    tools: tuple[str, ...] = ("read", "bash", "write", "edit", "ls")
    schema_version: str = CERTIFIED_LOOP_MANAGER_VERSION
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id or not isinstance(self.run_id, str):
            raise ContractValidationError("run_id must be a non-empty string")
        if not isinstance(self.policy, PolicyFingerprint):
            raise ContractValidationError("policy must be a PolicyFingerprint")
        for name in (
            "max_model_requests_per_episode",
            "max_prompt_tokens",
            "max_response_tokens",
            "max_tokens_per_generation",
            "max_concurrent_episodes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ContractValidationError(f"{name} must be a positive integer")
        if self.max_prompt_tokens + self.max_response_tokens > 8192:
            raise ContractValidationError("total training context exceeds 8192 tokens")
        if not (isinstance(self.episode_timeout_seconds, (int, float))) or isinstance(
            self.episode_timeout_seconds, bool
        ):
            raise ContractValidationError("episode_timeout_seconds must be numeric")
        if float(self.episode_timeout_seconds) <= 0:
            raise ContractValidationError("episode_timeout_seconds must be positive")
        if sorted(self.tools) != sorted(("read", "bash", "write", "edit", "ls")):
            raise ContractValidationError("tool set is fixed to read/bash/write/edit/ls")
