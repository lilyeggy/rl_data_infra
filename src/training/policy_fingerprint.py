"""Strict on-policy identity fingerprint.

On-policy identity is **not** determined by ``model_id`` alone.  Two episodes
are only ON_POLICY with respect to a target policy when their fingerprints are
complete and exactly match across provider, model, base revision, adapter/LoRA
revision or checksum, tokenizer revision, chat-template checksum, tool-schema
checksum, sampling configuration, temperature, top_p and policy generation.

When any critical identity field is missing the fingerprint is *incomplete* and
must never be treated as matching an on-policy target.  The system therefore
fails closed: no complete fingerprint -> no ON_POLICY determination.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from src.contracts._json import sha256_json, thaw_json
from src.contracts._validation import optional_text, required_text, strict_fields
from src.contracts.manifests import ModelManifest
from src.errors import ContractValidationError

POLICY_FINGERPRINT_VERSION = "policy-fingerprint/v1"


# Fields that must be observable to prove a fingerprint complete.
_REQUIRED_FIELDS = (
    "provider",
    "model_id",
    "base_model_revision",
    "adapter_revision",
    "tokenizer_revision",
    "chat_template_checksum",
    "tool_schema_checksum",
    "temperature",
    "top_p",
    "policy_generation",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class PolicyFingerprint:
    provider: str
    model_id: str
    base_model_revision: str
    adapter_revision: str | None = None
    tokenizer_revision: str | None = None
    chat_template_checksum: str | None = None
    tool_schema_checksum: str | None = None
    sampling_config: Mapping[str, Any] = field(default_factory=dict)
    temperature: float | None = None
    top_p: float | None = None
    policy_generation: str | None = None
    schema_version: str = POLICY_FINGERPRINT_VERSION

    def __post_init__(self) -> None:
        for name in ("provider", "model_id", "base_model_revision"):
            required_text(getattr(self, name), name)
        for name in (
            "adapter_revision",
            "tokenizer_revision",
            "chat_template_checksum",
            "tool_schema_checksum",
            "policy_generation",
        ):
            optional_text(getattr(self, name), name)
        for name in ("temperature", "top_p"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ContractValidationError(f"{name} must be numeric")
                if not math.isfinite(float(value)):
                    raise ContractValidationError(f"{name} must be finite")
                object.__setattr__(self, name, float(value))
        object.__setattr__(
            self,
            "sampling_config",
            dict(thaw_json(self.sampling_config)),
        )
        if self.schema_version != POLICY_FINGERPRINT_VERSION:
            raise ContractValidationError(
                f"schema_version must be {POLICY_FINGERPRINT_VERSION!r}"
            )

    def is_complete(self) -> bool:
        """True only when every critical fingerprint field is observable."""
        return all(getattr(self, name) is not None for name in _REQUIRED_FIELDS)

    def missing_fields(self) -> tuple[str, ...]:
        return tuple(name for name in _REQUIRED_FIELDS if getattr(self, name) is None)

    def _identity_map(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _REQUIRED_FIELDS}

    def matches(self, other: PolicyFingerprint) -> bool:
        """Two fingerprints are ON_POLICY-equal only when both are complete and equal."""
        if not isinstance(other, PolicyFingerprint):
            return False
        if not self.is_complete() or not other.is_complete():
            return False
        mine = self._identity_map()
        theirs = other._identity_map()
        for key in _REQUIRED_FIELDS:
            a, b = mine[key], theirs[key]
            if key in ("temperature", "top_p"):
                # treat -0.0/0.0 and tiny float repr drift as equal
                if not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9):
                    return False
            elif a != b:
                return False
        del mine, theirs
        # canonical sampling-config equality
        return self.sampling_config == other.sampling_config

    def checksum(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "model_id": self.model_id,
            "base_model_revision": self.base_model_revision,
            "adapter_revision": self.adapter_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "chat_template_checksum": self.chat_template_checksum,
            "tool_schema_checksum": self.tool_schema_checksum,
            "sampling_config": self.sampling_config,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "policy_generation": self.policy_generation,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PolicyFingerprint:
        payload = strict_fields(
            value,
            {
                "schema_version",
                "provider",
                "model_id",
                "base_model_revision",
                "adapter_revision",
                "tokenizer_revision",
                "chat_template_checksum",
                "tool_schema_checksum",
                "sampling_config",
                "temperature",
                "top_p",
                "policy_generation",
            },
            "PolicyFingerprint",
        )
        return cls(**payload)

    @classmethod
    def from_model_manifest(
        cls,
        manifest: ModelManifest,
        *,
        adapter_revision: str | None = None,
        chat_template_checksum: str | None = None,
        tool_schema_checksum: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        policy_generation: str | None = None,
    ) -> PolicyFingerprint:
        """Derive a fingerprint from a ModelManifest plus observable identity.

        Fields that are not observable remain ``None`` and make the fingerprint
        incomplete (fail closed for on-policy use).
        """
        return cls(
            provider=manifest.provider,
            model_id=manifest.model_id,
            base_model_revision=manifest.revision,
            adapter_revision=adapter_revision,
            tokenizer_revision=manifest.tokenizer_revision,
            chat_template_checksum=chat_template_checksum,
            tool_schema_checksum=tool_schema_checksum,
            sampling_config=dict(
                thaw_json(manifest.sampling_config)
                if isinstance(manifest.sampling_config, Mapping)
                else dict(manifest.sampling_config)
            ),
            temperature=temperature,
            top_p=top_p,
            policy_generation=policy_generation,
        )
