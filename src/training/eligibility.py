"""Unified, fail-closed training eligibility over a certified Episode.

Every training role consumes an ``EpisodeCertification``.  A data point is only
eligible for SFT / preference / off-policy RL / on-policy RL when it can be
proven by *real* verifier evidence, *real* model observation/action evidence
and, where required, *real* token/logprob/mask/reward fields plus a fully
matching policy fingerprint.

Nothing here fabricates a reward or guesses policy identity.  Missing evidence
results in REJECTED or INSUFFICIENT_EVIDENCE.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.contracts._json import sha256_json, thaw_json
from src.contracts.agent_episode import AgentEpisode, ExecutionValidity, TaskStatus
from src.contracts.episode_certification import (
    CertificationStatus,
    EpisodeCertification,
)

ELIGIBILITY_VERSION = "training-eligibility/v1"


class TrainingEligibilityStatus(str, Enum):
    SFT_ELIGIBLE = "SFT_ELIGIBLE"
    PREFERENCE_ELIGIBLE = "PREFERENCE_ELIGIBLE"
    OFF_POLICY_RL_ELIGIBLE = "OFF_POLICY_RL_ELIGIBLE"
    ON_POLICY_RL_ELIGIBLE = "ON_POLICY_RL_ELIGIBLE"
    REJECTED = "REJECTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingEligibility:
    episode_id: str
    episode_checksum: str
    certification_status: CertificationStatus
    status: TrainingEligibilityStatus
    sft_eligible: bool
    preference_eligible: bool
    off_policy_rl_eligible: bool
    on_policy_rl_eligible: bool
    behavior_policy: Mapping[str, Any] | None
    target_policy: Mapping[str, Any] | None
    reasons: tuple[str, ...]
    schema_version: str = ELIGIBILITY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "episode_checksum": self.episode_checksum,
            "certification_status": self.certification_status.value,
            "status": self.status.value,
            "sft_eligible": self.sft_eligible,
            "preference_eligible": self.preference_eligible,
            "off_policy_rl_eligible": self.off_policy_rl_eligible,
            "on_policy_rl_eligible": self.on_policy_rl_eligible,
            "behavior_policy": thaw_json(self.behavior_policy),
            "target_policy": thaw_json(self.target_policy),
            "reasons": list(self.reasons),
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


def _valid_reward(episode: AgentEpisode) -> bool:
    return not (
        episode.outcome.score is None
        or isinstance(episode.outcome.score, bool)
    )


def _observable_observation(episode: AgentEpisode) -> bool:
    from src.contracts.trace_event import EventType

    return any(event.event_type is EventType.MODEL_REQUEST for event in episode.events)


def _observable_action(episode: AgentEpisode) -> bool:
    from src.contracts.trace_event import EventType

    return any(
        event.event_type in (EventType.MODEL_RESPONSE, EventType.TOOL_CALL)
        for event in episode.events
    )


def _cap(evidence: Mapping[str, str], name: str, present_value: str = "EVIDENCE_PRESENT") -> bool:
    return evidence.get(name) == present_value


def evaluate_training_eligibility(
    episode: AgentEpisode,
    *,
    certification: EpisodeCertification | None = None,
    target_policy: Any | None = None,
    behavior_policy: Any | None = None,
) -> TrainingEligibility:
    """Fail-closed unified training eligibility for one canonical episode."""

    # lazy imports avoid a circular dependency with the contracts package
    from src.training.policy_fingerprint import PolicyFingerprint
    from src.validation.episode_semantics import certify_episode

    if not isinstance(episode, AgentEpisode):
        raise TypeError("evaluate_training_eligibility requires an AgentEpisode")

    cert = certification if certification is not None else certify_episode(episode)
    reasons: list[str] = []

    if cert.certification_status is CertificationStatus.REJECTED:
        return TrainingEligibility(
            episode_id=episode.episode_id,
            episode_checksum=episode.checksum,
            certification_status=cert.certification_status,
            status=TrainingEligibilityStatus.REJECTED,
            sft_eligible=False,
            preference_eligible=False,
            off_policy_rl_eligible=False,
            on_policy_rl_eligible=False,
            behavior_policy=None,
            target_policy=None,
            reasons=tuple(cert.rejection_reasons),
        )
    if cert.certification_status is CertificationStatus.INSUFFICIENT_EVIDENCE:
        return TrainingEligibility(
            episode_id=episode.episode_id,
            episode_checksum=episode.checksum,
            certification_status=cert.certification_status,
            status=TrainingEligibilityStatus.INSUFFICIENT_EVIDENCE,
            sft_eligible=False,
            preference_eligible=False,
            off_policy_rl_eligible=False,
            on_policy_rl_eligible=False,
            behavior_policy=None,
            target_policy=None,
            reasons=tuple(cert.rejection_reasons or ["insufficient evidence"]),
        )

    # -------- CERTIFIED: determine each role ------------------------------------
    evidence = cert.capability_evidence
    is_success = (
        episode.outcome.task_status is TaskStatus.SUCCESS
        and episode.outcome.execution_validity is ExecutionValidity.VALID
    )
    verifier_passed = cert.verifier_attested and (
        episode.outcome.verifier_status.value == "PASSED"
    )

    # SFT
    sft_eligible = (
        is_success
        and verifier_passed
        and _observable_observation(episode)
        and _observable_action(episode)
    )
    if is_success and not verifier_passed:
        reasons.append("SUCCESS episode lacks verifier PASSED attestation")
    if not _observable_observation(episode):
        reasons.append("no observable input observation for SFT")
    if not _observable_action(episode):
        reasons.append("no real assistant action/response for SFT")

    # Preference (either half of a valid preference pair)
    preference_eligible = cert.verifier_attested and (
        episode.outcome.execution_validity is ExecutionValidity.VALID
    )
    if not cert.verifier_attested:
        reasons.append("no verifier-attested valid outcome for preference")
    elif episode.outcome.execution_validity is not ExecutionValidity.VALID:
        reasons.append("execution not VALID for preference")

    # RL field prerequisites (shared by off/on policy)
    has_token_ids = _cap(evidence, "MODEL_TOKEN_IDS")
    has_logprobs = _cap(evidence, "MODEL_LOGPROBS")
    has_mask = _cap(evidence, "ACTION_MASK")
    arrays_aligned = _cap(evidence, "ARRAYS_ALIGNED")
    # A verifier-attested VALID failure with reward 0 is essential RL signal,
    # not an infrastructure fault. Only SFT requires task success.
    reward_ok = (
        _valid_reward(episode)
        and episode.outcome.execution_validity is ExecutionValidity.VALID
        and cert.verifier_attested
    )
    if not has_token_ids:
        reasons.append("RL rejected: no real token ID array evidence")
    if not has_logprobs:
        reasons.append("RL rejected: no real behavior logprob evidence")
    if not has_mask:
        reasons.append("RL rejected: no real action/loss mask evidence")
    if not arrays_aligned:
        reasons.append("RL rejected: token/logprob/mask arrays not aligned")
    if not reward_ok:
        reasons.append(
            "RL rejected: no verifier-attested finite reward for a VALID execution"
        )

    # Fingerprints
    behavior_fp = behavior_policy or PolicyFingerprint.from_model_manifest(
        episode.model_manifest
    )
    target_fp = target_policy
    behavior_complete = behavior_fp.is_complete()
    target_complete = target_fp.is_complete() if target_fp is not None else False
    policies_equal = (
        behavior_complete and target_complete and behavior_fp.matches(target_fp)
    )
    if not behavior_complete:
        reasons.append(
            "behavior policy fingerprint incomplete: "
            + ", ".join(behavior_fp.missing_fields())
        )
    if target_fp is not None and not target_complete:
        reasons.append("target policy fingerprint incomplete: " + ", ".join(target_fp.missing_fields()))

    rl_fields = has_token_ids and has_mask and arrays_aligned and reward_ok

    off_policy_rl_eligible = rl_fields and not policies_equal
    on_policy_rl_eligible = rl_fields and has_logprobs and policies_equal

    if on_policy_rl_eligible:
        status = TrainingEligibilityStatus.ON_POLICY_RL_ELIGIBLE
    elif off_policy_rl_eligible:
        status = TrainingEligibilityStatus.OFF_POLICY_RL_ELIGIBLE
    elif sft_eligible:
        status = TrainingEligibilityStatus.SFT_ELIGIBLE
    elif preference_eligible:
        status = TrainingEligibilityStatus.PREFERENCE_ELIGIBLE
    else:
        status = TrainingEligibilityStatus.REJECTED
        if not reasons:
            reasons.append("certified but no training role prerequisites satisfied")

    return TrainingEligibility(
        episode_id=episode.episode_id,
        episode_checksum=episode.checksum,
        certification_status=cert.certification_status,
        status=status,
        sft_eligible=sft_eligible,
        preference_eligible=preference_eligible,
        off_policy_rl_eligible=off_policy_rl_eligible,
        on_policy_rl_eligible=on_policy_rl_eligible,
        behavior_policy=behavior_fp.to_dict() if behavior_fp is not None else None,
        target_policy=target_fp.to_dict() if target_fp is not None else None,
        reasons=tuple(reasons),
    )
