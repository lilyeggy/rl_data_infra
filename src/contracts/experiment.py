"""Versioned contract for a controlled Harness comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from src.contracts._json import freeze_json, sha256_json, thaw_json
from src.contracts._validation import required_text, strict_fields
from src.errors import ContractValidationError


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentManifest:
    experiment_id: str
    revision: str
    task_dataset_revision: str
    control_run_id: str
    candidate_run_id: str
    target_policy_flag: str
    minimum_pairs: int = 3
    pairing_fields: tuple[str, ...] = ("task_id", "seed", "attempt")
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "experiment-manifest/v1"

    def __post_init__(self) -> None:
        for name in (
            "experiment_id",
            "revision",
            "task_dataset_revision",
            "control_run_id",
            "candidate_run_id",
            "target_policy_flag",
        ):
            required_text(getattr(self, name), name)
        if self.control_run_id == self.candidate_run_id:
            raise ContractValidationError("control_run_id and candidate_run_id must differ")
        if (
            isinstance(self.minimum_pairs, bool)
            or not isinstance(self.minimum_pairs, int)
            or self.minimum_pairs < 1
        ):
            raise ContractValidationError("minimum_pairs must be a positive integer")
        fields = tuple(self.pairing_fields)
        if fields != ("task_id", "seed", "attempt"):
            raise ContractValidationError(
                "v1 pairing_fields must be ('task_id', 'seed', 'attempt')"
            )
        object.__setattr__(self, "pairing_fields", fields)
        object.__setattr__(self, "metadata", freeze_json(self.metadata, "$.metadata"))
        if self.schema_version != "experiment-manifest/v1":
            raise ContractValidationError("unsupported ExperimentManifest schema_version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "revision": self.revision,
            "task_dataset_revision": self.task_dataset_revision,
            "control_run_id": self.control_run_id,
            "candidate_run_id": self.candidate_run_id,
            "target_policy_flag": self.target_policy_flag,
            "minimum_pairs": self.minimum_pairs,
            "pairing_fields": list(self.pairing_fields),
            "metadata": thaw_json(self.metadata),
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentManifest":
        fields = {
            "schema_version",
            "experiment_id",
            "revision",
            "task_dataset_revision",
            "control_run_id",
            "candidate_run_id",
            "target_policy_flag",
            "minimum_pairs",
            "pairing_fields",
            "metadata",
        }
        payload = dict(strict_fields(value, fields, "ExperimentManifest"))
        payload["pairing_fields"] = tuple(payload["pairing_fields"])
        return cls(**payload)
