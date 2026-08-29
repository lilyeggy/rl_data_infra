"""Versioned identities used to prove what changed between Agent runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from src.contracts._json import sha256_json, thaw_json, validate_sha256
from src.contracts._validation import frozen_object, optional_text, required_text
from src.errors import ContractValidationError


def _validate_digest(value: str, field_name: str) -> None:
    required_text(value, field_name)
    validate_sha256(value, field_name)


@dataclass(frozen=True, slots=True, kw_only=True)
class HarnessManifest:
    name: str
    version: str
    revision: str
    config_digest: str
    policy_flags: Mapping[str, Any] = field(default_factory=dict)
    hook_version: str | None = None
    schema_version: str = "harness-manifest/v1"

    def __post_init__(self) -> None:
        for name in ("name", "version", "revision"):
            required_text(getattr(self, name), name)
        _validate_digest(self.config_digest, "config_digest")
        optional_text(self.hook_version, "hook_version")
        object.__setattr__(
            self, "policy_flags", frozen_object(self.policy_flags, "policy_flags")
        )
        if self.schema_version != "harness-manifest/v1":
            raise ContractValidationError("unsupported HarnessManifest schema_version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "revision": self.revision,
            "config_digest": self.config_digest,
            "policy_flags": thaw_json(self.policy_flags),
            "hook_version": self.hook_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HarnessManifest":
        try:
            return cls(**dict(value))
        except TypeError as exc:
            raise ContractValidationError(f"invalid HarnessManifest fields: {exc}") from exc


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelManifest:
    provider: str
    model_id: str
    revision: str
    sampling_config: Mapping[str, Any] = field(default_factory=dict)
    tokenizer_revision: str | None = None
    schema_version: str = "model-manifest/v1"

    def __post_init__(self) -> None:
        for name in ("provider", "model_id", "revision"):
            required_text(getattr(self, name), name)
        optional_text(self.tokenizer_revision, "tokenizer_revision")
        object.__setattr__(
            self,
            "sampling_config",
            frozen_object(self.sampling_config, "sampling_config"),
        )
        if self.schema_version != "model-manifest/v1":
            raise ContractValidationError("unsupported ModelManifest schema_version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "model_id": self.model_id,
            "revision": self.revision,
            "sampling_config": thaw_json(self.sampling_config),
            "tokenizer_revision": self.tokenizer_revision,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelManifest":
        try:
            return cls(**dict(value))
        except TypeError as exc:
            raise ContractValidationError(f"invalid ModelManifest fields: {exc}") from exc


@dataclass(frozen=True, slots=True, kw_only=True)
class EnvironmentManifest:
    runtime_type: str
    revision: str
    image: str | None = None
    resource_limits: Mapping[str, Any] = field(default_factory=dict)
    network_policy: str = "unspecified"
    task_snapshot: str | None = None
    schema_version: str = "environment-manifest/v1"

    def __post_init__(self) -> None:
        for name in ("runtime_type", "revision", "network_policy"):
            required_text(getattr(self, name), name)
        optional_text(self.image, "image")
        optional_text(self.task_snapshot, "task_snapshot")
        object.__setattr__(
            self,
            "resource_limits",
            frozen_object(self.resource_limits, "resource_limits"),
        )
        if self.schema_version != "environment-manifest/v1":
            raise ContractValidationError("unsupported EnvironmentManifest schema_version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "runtime_type": self.runtime_type,
            "revision": self.revision,
            "image": self.image,
            "resource_limits": thaw_json(self.resource_limits),
            "network_policy": self.network_policy,
            "task_snapshot": self.task_snapshot,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EnvironmentManifest":
        try:
            return cls(**dict(value))
        except TypeError as exc:
            raise ContractValidationError(f"invalid EnvironmentManifest fields: {exc}") from exc


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluatorManifest:
    name: str
    revision: str
    config_digest: str
    schema_version: str = "evaluator-manifest/v1"

    def __post_init__(self) -> None:
        required_text(self.name, "name")
        required_text(self.revision, "revision")
        _validate_digest(self.config_digest, "config_digest")
        if self.schema_version != "evaluator-manifest/v1":
            raise ContractValidationError("unsupported EvaluatorManifest schema_version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "revision": self.revision,
            "config_digest": self.config_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluatorManifest":
        try:
            return cls(**dict(value))
        except TypeError as exc:
            raise ContractValidationError(f"invalid EvaluatorManifest fields: {exc}") from exc


def manifest_digest(manifest: object) -> str:
    to_dict = getattr(manifest, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("manifest must implement to_dict()")
    return sha256_json(to_dict())
