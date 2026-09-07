"""Content-addressed dataset manifests compiled only from certified artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.contracts._json import freeze_json, sha256_json, thaw_json, validate_sha256
from src.contracts._validation import required_text, strict_fields
from src.contracts.execution_identity import ExecutionIdentity
from src.errors import ContractValidationError

DATASET_MANIFEST_VERSION = "dataset-manifest/v1"


class DatasetPurpose(str, Enum):
    HARNESS_REGRESSION = "HARNESS_REGRESSION"
    EVALUATION = "EVALUATION"
    SFT = "SFT"
    PREFERENCE = "PREFERENCE"
    OFF_POLICY_RL = "OFF_POLICY_RL"
    ON_POLICY_RL = "ON_POLICY_RL"


class DatasetSplit(str, Enum):
    TRAIN = "TRAIN"
    DEVELOPMENT = "DEVELOPMENT"
    TEST = "TEST"
    REGRESSION = "REGRESSION"


class DatasetRole(str, Enum):
    EXAMPLE = "EXAMPLE"
    CHOSEN = "CHOSEN"
    REJECTED = "REJECTED"
    TRAJECTORY = "TRAJECTORY"


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetMember:
    identity: ExecutionIdentity
    episode_checksum: str
    certification_checksum: str
    artifact_checksum: str
    split: DatasetSplit
    role: DatasetRole
    execution_bundle_checksum: str | None = None
    policy_artifact_checksum: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ExecutionIdentity):
            raise ContractValidationError("identity must be an ExecutionIdentity")
        for name in (
            "episode_checksum",
            "certification_checksum",
            "artifact_checksum",
            "execution_bundle_checksum",
            "policy_artifact_checksum",
        ):
            validate_sha256(getattr(self, name), name)
        if not isinstance(self.split, DatasetSplit):
            raise ContractValidationError("split must be a DatasetSplit")
        if not isinstance(self.role, DatasetRole):
            raise ContractValidationError("role must be a DatasetRole")

    @property
    def member_id(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "episode_checksum": self.episode_checksum,
            "certification_checksum": self.certification_checksum,
            "artifact_checksum": self.artifact_checksum,
            "split": self.split.value,
            "role": self.role.value,
            "execution_bundle_checksum": self.execution_bundle_checksum,
            "policy_artifact_checksum": self.policy_artifact_checksum,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetMember":
        payload = strict_fields(
            value,
            {
                "identity",
                "episode_checksum",
                "certification_checksum",
                "artifact_checksum",
                "split",
                "role",
                "execution_bundle_checksum",
                "policy_artifact_checksum",
            },
            "DatasetMember",
        )
        try:
            split = DatasetSplit(payload["split"])
            role = DatasetRole(payload["role"])
        except (TypeError, ValueError) as exc:
            raise ContractValidationError("invalid DatasetMember enum value") from exc
        return cls(
            identity=ExecutionIdentity.from_dict(payload["identity"]),
            episode_checksum=payload["episode_checksum"],
            certification_checksum=payload["certification_checksum"],
            artifact_checksum=payload["artifact_checksum"],
            split=split,
            role=role,
            execution_bundle_checksum=payload["execution_bundle_checksum"],
            policy_artifact_checksum=payload["policy_artifact_checksum"],
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetManifest:
    dataset_id: str
    revision: str
    purpose: DatasetPurpose
    selection_policy_version: str
    members: tuple[DatasetMember, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = DATASET_MANIFEST_VERSION

    def __post_init__(self) -> None:
        for name in ("dataset_id", "revision", "selection_policy_version"):
            required_text(getattr(self, name), name)
        if not isinstance(self.purpose, DatasetPurpose):
            raise ContractValidationError("purpose must be a DatasetPurpose")
        members = tuple(self.members)
        if not members:
            raise ContractValidationError("dataset manifest requires at least one member")
        if any(not isinstance(member, DatasetMember) for member in members):
            raise ContractValidationError("members must contain DatasetMember values")
        member_ids = tuple(member.member_id for member in members)
        if len(member_ids) != len(set(member_ids)):
            raise ContractValidationError("duplicate dataset member")

        # A task, including all of its attempts/groups, belongs to exactly one
        # split. This prevents benchmark and retry leakage across train/test.
        task_splits: dict[str, DatasetSplit] = {}
        for member in members:
            previous = task_splits.setdefault(member.identity.task_id, member.split)
            if previous is not member.split:
                raise ContractValidationError(
                    f"task {member.identity.task_id!r} leaks across dataset splits"
                )

        allowed_roles = {
            DatasetPurpose.SFT: {DatasetRole.EXAMPLE},
            DatasetPurpose.PREFERENCE: {DatasetRole.CHOSEN, DatasetRole.REJECTED},
            DatasetPurpose.OFF_POLICY_RL: {DatasetRole.TRAJECTORY},
            DatasetPurpose.ON_POLICY_RL: {DatasetRole.TRAJECTORY},
            DatasetPurpose.EVALUATION: {DatasetRole.EXAMPLE},
            DatasetPurpose.HARNESS_REGRESSION: {DatasetRole.EXAMPLE},
        }[self.purpose]
        if any(member.role not in allowed_roles for member in members):
            raise ContractValidationError(
                f"member role is incompatible with dataset purpose {self.purpose.value}"
            )
        if self.purpose is DatasetPurpose.PREFERENCE:
            pair_groups: dict[str, list[DatasetMember]] = {}
            for member in members:
                if member.identity.group_id is None:
                    raise ContractValidationError(
                        "preference members require a group_id identifying the pair"
                    )
                pair_groups.setdefault(member.identity.group_id, []).append(member)
            expected_pair_roles = {DatasetRole.CHOSEN, DatasetRole.REJECTED}
            for group_id, pair in pair_groups.items():
                roles = {member.role for member in pair}
                if len(pair) != 2 or roles != expected_pair_roles:
                    raise ContractValidationError(
                        f"preference group {group_id!r} must contain exactly one "
                        "CHOSEN and one REJECTED member"
                    )
                if len({member.identity.task_id for member in pair}) != 1:
                    raise ContractValidationError(
                        f"preference group {group_id!r} crosses logical tasks"
                    )
        object.__setattr__(self, "members", tuple(sorted(members, key=lambda m: m.member_id)))
        frozen = freeze_json(self.metadata)
        if not isinstance(frozen, Mapping):
            raise ContractValidationError("metadata must be an object")
        object.__setattr__(self, "metadata", frozen)
        if self.schema_version != DATASET_MANIFEST_VERSION:
            raise ContractValidationError(
                f"schema_version must be {DATASET_MANIFEST_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "revision": self.revision,
            "purpose": self.purpose.value,
            "selection_policy_version": self.selection_policy_version,
            "members": [member.to_dict() for member in self.members],
            "metadata": thaw_json(self.metadata),
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetManifest":
        payload = strict_fields(
            value,
            {
                "schema_version",
                "dataset_id",
                "revision",
                "purpose",
                "selection_policy_version",
                "members",
                "metadata",
            },
            "DatasetManifest",
        )
        try:
            purpose = DatasetPurpose(payload["purpose"])
        except (TypeError, ValueError) as exc:
            raise ContractValidationError("invalid DatasetManifest purpose") from exc
        return cls(
            schema_version=payload["schema_version"],
            dataset_id=payload["dataset_id"],
            revision=payload["revision"],
            purpose=purpose,
            selection_policy_version=payload["selection_policy_version"],
            members=tuple(DatasetMember.from_dict(item) for item in payload["members"]),
            metadata=payload["metadata"],
        )
