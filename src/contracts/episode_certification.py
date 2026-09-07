"""Derived Episode certification for the training-data trust boundary.

Certification is a **derived** object: it never mutates raw ``TraceEvent``
values or the ``AgentEpisode`` it describes.  It records which semantic checks
passed and which pieces of evidence (verifier attestation, real model
observation/action, actual token/logprob/mask fields) were observable.

Fail-closed rules enforced here:
- a ``SUCCESS + VALID`` terminal outcome must be backed by real verifier
  PASSED evidence, otherwise the episode is INSUFFICIENT_EVIDENCE;
- a verifier FAILED attestation cannot coexist with a terminal SUCCESS;
- a verifier ERROR/TIMEOUT cannot support a trustworthy task SUCCESS nor be
  converted into a fabricated reward (including 0);
- outcome ``evidence_event_ids`` must reference real, semantically relevant
  verifier events;
- PARTIAL/CORRUPT episodes are not trainable-certified;
- a terminal line alone, without real model observation/action/verifier
  evidence, cannot be certified.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.contracts._json import sha256_json, thaw_json
from src.contracts._validation import required_text, strict_fields
from src.errors import ContractValidationError

EPISODE_CERTIFICATION_VERSION = "episode-certification/v1"


class CertificationStatus(str, Enum):
    CERTIFIED = "CERTIFIED"
    REJECTED = "REJECTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True, slots=True)
class SemanticCheck:
    """One named, stable, serializable semantic assertion about an episode."""

    name: str
    passed: bool
    detail: str

    def __post_init__(self) -> None:
        required_text(self.name, "SemanticCheck.name")
        if not isinstance(self.passed, bool):
            raise ContractValidationError("SemanticCheck.passed must be boolean")
        required_text(self.detail, "SemanticCheck.detail")
    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticCheck:
        return cls(**strict_fields(value, {"name", "passed", "detail"}, "SemanticCheck"))


@dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeCertification:
    """Derived certification verdict over a canonical AgentEpisode.

    ``capability_evidence`` maps a capability name to a stable evidence status:

    - ``EVIDENCE_PRESENT`` — a real observable field/event proves the capability;
    - ``MISSING`` — the capability is declared or required but not observable;
    - ``MISALIGNED`` — fields exist but violate length/position semantics.
    """

    episode_id: str
    episode_checksum: str
    certification_status: CertificationStatus
    verifier_attested: bool
    semantic_checks: tuple[SemanticCheck, ...]
    capability_evidence: Mapping[str, str]
    rejection_reasons: tuple[str, ...]
    validator_version: str
    schema_version: str = EPISODE_CERTIFICATION_VERSION

    def __post_init__(self) -> None:
        required_text(self.episode_id, "episode_id")
        from src.contracts._json import validate_sha256

        validate_sha256(self.episode_checksum, "episode_checksum")
        if not isinstance(self.certification_status, CertificationStatus):
            raise ContractValidationError(
                "certification_status must be a CertificationStatus"
            )
        if not isinstance(self.verifier_attested, bool):
            raise ContractValidationError("verifier_attested must be boolean")
        checks = tuple(self.semantic_checks)
        if any(not isinstance(item, SemanticCheck) for item in checks):
            raise ContractValidationError("semantic_checks must contain SemanticCheck values")
        object.__setattr__(self, "semantic_checks", checks)
        frozen = thaw_json(self.capability_evidence)
        if not isinstance(frozen, Mapping):
            raise ContractValidationError("capability_evidence must be an object")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in frozen.items()):
            raise ContractValidationError("capability_evidence must map str to str")
        object.__setattr__(self, "capability_evidence", dict(frozen))
        reasons = tuple(self.rejection_reasons)
        for index, value in enumerate(reasons):
            required_text(value, f"rejection_reasons[{index}]")
        object.__setattr__(self, "rejection_reasons", reasons)
        required_text(self.validator_version, "validator_version")
        if self.schema_version != EPISODE_CERTIFICATION_VERSION:
            raise ContractValidationError(
                f"schema_version must be {EPISODE_CERTIFICATION_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "episode_checksum": self.episode_checksum,
            "certification_status": self.certification_status.value,
            "verifier_attested": self.verifier_attested,
            "semantic_checks": [item.to_dict() for item in self.semantic_checks],
            "capability_evidence": self.capability_evidence,
            "rejection_reasons": list(self.rejection_reasons),
            "validator_version": self.validator_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EpisodeCertification:
        fields = {
            "schema_version",
            "episode_id",
            "episode_checksum",
            "certification_status",
            "verifier_attested",
            "semantic_checks",
            "capability_evidence",
            "rejection_reasons",
            "validator_version",
        }
        payload = strict_fields(value, fields, "EpisodeCertification")
        return cls(
            schema_version=payload["schema_version"],
            episode_id=payload["episode_id"],
            episode_checksum=payload["episode_checksum"],
            certification_status=CertificationStatus(payload["certification_status"]),
            verifier_attested=payload["verifier_attested"],
            semantic_checks=tuple(
                SemanticCheck.from_dict(item) for item in payload["semantic_checks"]
            ),
            capability_evidence=payload["capability_evidence"],
            rejection_reasons=tuple(payload["rejection_reasons"]),
            validator_version=payload["validator_version"],
        )

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())

    @property
    def is_certified(self) -> bool:
        return self.certification_status is CertificationStatus.CERTIFIED
