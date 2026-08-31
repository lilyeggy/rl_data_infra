"""Unified, versioned eligibility decision for every data consumer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.contracts._json import sha256_json, validate_sha256
from src.contracts._validation import strict_fields
from src.contracts.agent_episode import AgentEpisode, ExecutionValidity, IntegrityState
from src.contracts.execution_bundle import ExecutionBundle
from src.errors import ContractValidationError
from src.producers.base import (
    ProducerArtifact,
    ProducerCapability,
    ProducerExecutionStatus,
)
from src.training.eligibility import evaluate_training_eligibility
from src.validation.episode_semantics import certify_episode

CERTIFICATION_ENGINE_VERSION = "consumer-certification/v2"


class ConsumerProfile(str, Enum):
    HARNESS_ANALYSIS = "HARNESS_ANALYSIS"
    EVALUATION = "EVALUATION"
    SFT = "SFT"
    PREFERENCE = "PREFERENCE"
    OFF_POLICY_RL = "OFF_POLICY_RL"
    ON_POLICY_RL = "ON_POLICY_RL"


class ConsumerVerdict(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    REJECTED = "REJECTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True, slots=True, kw_only=True)
class EligibilityDecision:
    episode_id: str
    episode_checksum: str
    profile: ConsumerProfile
    verdict: ConsumerVerdict
    reasons: tuple[str, ...]
    episode_certification_checksum: str
    training_eligibility_checksum: str | None
    execution_bundle_checksum: str | None = None
    policy_artifact_checksum: str | None = None
    engine_version: str = CERTIFICATION_ENGINE_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ConsumerProfile):
            raise ContractValidationError("profile must be a ConsumerProfile")
        if not isinstance(self.verdict, ConsumerVerdict):
            raise ContractValidationError("verdict must be a ConsumerVerdict")
        for name in (
            "episode_checksum",
            "episode_certification_checksum",
            "training_eligibility_checksum",
            "execution_bundle_checksum",
            "policy_artifact_checksum",
        ):
            validate_sha256(getattr(self, name), name)
        if self.engine_version != CERTIFICATION_ENGINE_VERSION:
            raise ContractValidationError(
                f"engine_version must be {CERTIFICATION_ENGINE_VERSION!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_version": self.engine_version,
            "episode_id": self.episode_id,
            "episode_checksum": self.episode_checksum,
            "profile": self.profile.value,
            "verdict": self.verdict.value,
            "reasons": list(self.reasons),
            "episode_certification_checksum": self.episode_certification_checksum,
            "training_eligibility_checksum": self.training_eligibility_checksum,
            "execution_bundle_checksum": self.execution_bundle_checksum,
            "policy_artifact_checksum": self.policy_artifact_checksum,
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EligibilityDecision":
        payload = strict_fields(
            value,
            {
                "engine_version",
                "episode_id",
                "episode_checksum",
                "profile",
                "verdict",
                "reasons",
                "episode_certification_checksum",
                "training_eligibility_checksum",
                "execution_bundle_checksum",
                "policy_artifact_checksum",
            },
            "EligibilityDecision",
        )
        try:
            profile = ConsumerProfile(payload["profile"])
            verdict = ConsumerVerdict(payload["verdict"])
        except (TypeError, ValueError) as exc:
            raise ContractValidationError("invalid EligibilityDecision enum value") from exc
        return cls(
            engine_version=payload["engine_version"],
            episode_id=payload["episode_id"],
            episode_checksum=payload["episode_checksum"],
            profile=profile,
            verdict=verdict,
            reasons=tuple(payload["reasons"]),
            episode_certification_checksum=payload["episode_certification_checksum"],
            training_eligibility_checksum=payload["training_eligibility_checksum"],
            execution_bundle_checksum=payload["execution_bundle_checksum"],
            policy_artifact_checksum=payload["policy_artifact_checksum"],
        )


def certify_for(
    episode: AgentEpisode,
    profile: ConsumerProfile,
    *,
    target_policy: Any | None = None,
    behavior_policy: Any | None = None,
    execution_bundle: ExecutionBundle | None = None,
    policy_artifact: ProducerArtifact | None = None,
    target_policy_fingerprint: str | None = None,
) -> EligibilityDecision:
    """Return the only supported consumer-eligibility decision for an episode."""

    if not isinstance(episode, AgentEpisode):
        raise TypeError("certify_for requires an AgentEpisode")
    if not isinstance(profile, ConsumerProfile):
        raise TypeError("profile must be a ConsumerProfile")

    cert = certify_episode(episode)
    training = None
    reasons: tuple[str, ...]

    if profile is ConsumerProfile.HARNESS_ANALYSIS:
        if episode.integrity.state is IntegrityState.CORRUPT:
            verdict = ConsumerVerdict.REJECTED
            reasons = ("corrupt episodes cannot support harness analysis",)
        elif not episode.events:
            verdict = ConsumerVerdict.INSUFFICIENT_EVIDENCE
            reasons = ("episode has no observable events",)
        else:
            verdict = ConsumerVerdict.ELIGIBLE
            reasons = ()
    elif profile is ConsumerProfile.EVALUATION:
        if not cert.is_certified:
            verdict = ConsumerVerdict.INSUFFICIENT_EVIDENCE
            reasons = cert.rejection_reasons or ("episode is not certified",)
        elif episode.outcome.execution_validity is not ExecutionValidity.VALID:
            verdict = ConsumerVerdict.REJECTED
            reasons = ("evaluation requires execution validity VALID",)
        elif not cert.verifier_attested:
            verdict = ConsumerVerdict.INSUFFICIENT_EVIDENCE
            reasons = ("evaluation requires verifier attestation",)
        else:
            verdict = ConsumerVerdict.ELIGIBLE
            reasons = ()
    elif profile in (ConsumerProfile.OFF_POLICY_RL, ConsumerProfile.ON_POLICY_RL):
        return _certify_bundle_for_rl(
            episode,
            profile,
            execution_bundle=execution_bundle,
            policy_artifact=policy_artifact,
            target_policy_fingerprint=target_policy_fingerprint,
        )
    else:
        training = evaluate_training_eligibility(
            episode,
            certification=cert,
            target_policy=target_policy,
            behavior_policy=behavior_policy,
        )
        flag = {
            ConsumerProfile.SFT: training.sft_eligible,
            ConsumerProfile.PREFERENCE: training.preference_eligible,
            ConsumerProfile.OFF_POLICY_RL: training.off_policy_rl_eligible,
            ConsumerProfile.ON_POLICY_RL: training.on_policy_rl_eligible,
        }[profile]
        if flag:
            verdict = ConsumerVerdict.ELIGIBLE
            reasons = ()
        elif cert.certification_status.value == "INSUFFICIENT_EVIDENCE":
            verdict = ConsumerVerdict.INSUFFICIENT_EVIDENCE
            reasons = training.reasons or ("insufficient training evidence",)
        else:
            verdict = ConsumerVerdict.REJECTED
            reasons = (
                _sft_rejection_reasons(episode, training.reasons)
                if profile is ConsumerProfile.SFT
                else training.reasons or (f"not eligible for {profile.value}",)
            )

    return EligibilityDecision(
        episode_id=episode.episode_id,
        episode_checksum=episode.checksum,
        profile=profile,
        verdict=verdict,
        reasons=reasons,
        episode_certification_checksum=cert.checksum,
        training_eligibility_checksum=training.checksum if training is not None else None,
    )


def _sft_rejection_reasons(
    episode: AgentEpisode, training_reasons: Sequence[str]
) -> tuple[str, ...]:
    reasons: list[str] = []
    if episode.outcome.task_status.value != "SUCCESS":
        reasons.append("SFT requires task status SUCCESS")
    if episode.outcome.execution_validity is not ExecutionValidity.VALID:
        reasons.append("SFT requires execution validity VALID")
    if episode.outcome.verifier_status.value != "PASSED":
        reasons.append("SFT requires verifier status PASSED")
    reasons.extend(
        reason
        for reason in training_reasons
        if reason.startswith("no observable") or "verifier PASSED" in reason
    )
    return tuple(dict.fromkeys(reasons)) or ("not eligible for SFT",)


def _certify_bundle_for_rl(
    episode: AgentEpisode,
    profile: ConsumerProfile,
    *,
    execution_bundle: ExecutionBundle | None,
    policy_artifact: ProducerArtifact | None,
    target_policy_fingerprint: str | None,
) -> EligibilityDecision:
    cert = certify_episode(episode)
    reasons: list[str] = []
    verdict = ConsumerVerdict.INSUFFICIENT_EVIDENCE

    if execution_bundle is None or policy_artifact is None:
        reasons.append("RL requires an ExecutionBundle and bound policy artifact")
    elif execution_bundle.episode_checksum != episode.checksum:
        verdict = ConsumerVerdict.REJECTED
        reasons.append("ExecutionBundle episode checksum mismatch")
    elif (
        execution_bundle.identity.run_id,
        execution_bundle.identity.task_id,
        execution_bundle.identity.episode_id,
        execution_bundle.identity.attempt_id,
    ) != (episode.run_id, episode.task_id, episode.episode_id, episode.attempt):
        verdict = ConsumerVerdict.REJECTED
        reasons.append("ExecutionBundle episode identity mismatch")
    elif policy_artifact.identity != execution_bundle.identity:
        verdict = ConsumerVerdict.REJECTED
        reasons.append("policy artifact identity mismatch")
    elif policy_artifact.checksum not in execution_bundle.policy_trace_checksums:
        verdict = ConsumerVerdict.REJECTED
        reasons.append("policy artifact is not bound by ExecutionBundle")
    elif policy_artifact.status is not ProducerExecutionStatus.COMPLETED:
        verdict = ConsumerVerdict.REJECTED
        reasons.append("RL requires a completed producer execution")
    elif not cert.is_certified:
        reasons.extend(cert.rejection_reasons or ("Episode is not certified",))
    elif episode.outcome.execution_validity is not ExecutionValidity.VALID:
        verdict = ConsumerVerdict.REJECTED
        reasons.append("RL requires execution validity VALID")
    elif not cert.verifier_attested:
        reasons.append("RL requires verifier attestation")
    elif execution_bundle.verifier_report_checksum is None:
        reasons.append("ExecutionBundle has no verifier report")
    else:
        required = {
            ProducerCapability.TOKEN_IDS,
            ProducerCapability.ACTION_MASK,
            ProducerCapability.POLICY_VERSION,
            ProducerCapability.VERIFIER_EVIDENCE,
        }
        if profile is ConsumerProfile.ON_POLICY_RL:
            required.add(ProducerCapability.BEHAVIOR_LOGPROBS)
        missing = required - policy_artifact.capabilities
        if missing:
            reasons.append(
                "policy evidence lacks capabilities: "
                + ", ".join(sorted(item.value for item in missing))
            )
        if not _artifact_has_finite_rewards(policy_artifact):
            reasons.append("policy artifact has no complete finite reward evidence")

        behavior_fingerprint = execution_bundle.identity.policy_fingerprint
        if behavior_fingerprint is None:
            reasons.append("behavior policy fingerprint is missing")
        validate_sha256(target_policy_fingerprint, "target_policy_fingerprint")
        if profile is ConsumerProfile.ON_POLICY_RL:
            if target_policy_fingerprint is None:
                reasons.append("on-policy RL requires target policy fingerprint")
            elif behavior_fingerprint != target_policy_fingerprint:
                verdict = ConsumerVerdict.REJECTED
                reasons.append("behavior and target policy fingerprints differ")
        elif (
            target_policy_fingerprint is not None
            and behavior_fingerprint == target_policy_fingerprint
        ):
            verdict = ConsumerVerdict.REJECTED
            reasons.append("off-policy profile received target-policy data")

        if not reasons:
            verdict = ConsumerVerdict.ELIGIBLE

    return EligibilityDecision(
        episode_id=episode.episode_id,
        episode_checksum=episode.checksum,
        profile=profile,
        verdict=verdict,
        reasons=tuple(reasons),
        episode_certification_checksum=cert.checksum,
        training_eligibility_checksum=None,
        execution_bundle_checksum=(
            execution_bundle.checksum if execution_bundle is not None else None
        ),
        policy_artifact_checksum=(
            policy_artifact.checksum if policy_artifact is not None else None
        ),
    )


def _artifact_has_finite_rewards(artifact: ProducerArtifact) -> bool:
    trajectory = artifact.payload.get("trajectory")
    if not isinstance(trajectory, Mapping):
        return False
    traces = trajectory.get("traces")
    if not isinstance(traces, Sequence) or isinstance(traces, (str, bytes)) or not traces:
        return False
    for trace in traces:
        if not isinstance(trace, Mapping):
            return False
        reward = trace.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            return False
    return True
