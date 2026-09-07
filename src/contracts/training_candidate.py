"""Compatibility projection over the unified training eligibility decision.

New dataset code should consume ``certification.EligibilityDecision`` and
``learning.compile_dataset``.  This view remains for historical scenario
outputs; it does not own a second eligibility policy.

The view reports, per episode, whether it is an SFT candidate and an on-policy
RL candidate by consuming the unified, fail-closed ``TrainingEligibility``.
Capability *declarations* are no longer treated as proof: SFT/on-policy RL now
require real verifier evidence, real observation/action evidence and, for RL,
actual token/logprob/mask fields plus a fully matching policy fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.contracts._json import sha256_json
from src.contracts.agent_episode import AgentEpisode, CaptureCapability

TRAINING_CANDIDATE_VERSION = "training-candidate-view/v1"


@dataclass(frozen=True, slots=True)
class TrainingCandidateView:
    episode_id: str
    episode_checksum: str
    behavior_provider: str
    behavior_model: str
    behavior_model_revision: str
    training_role: str
    policy_relation: str
    sft_candidate: bool
    on_policy_rl_candidate: bool
    missing_rl_capabilities: tuple[str, ...]
    exclusion_reasons: tuple[str, ...]
    source_event_ids: tuple[str, ...]
    view_version: str = TRAINING_CANDIDATE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_version": self.view_version,
            "episode_id": self.episode_id,
            "episode_checksum": self.episode_checksum,
            "behavior_provider": self.behavior_provider,
            "behavior_model": self.behavior_model,
            "behavior_model_revision": self.behavior_model_revision,
            "training_role": self.training_role,
            "policy_relation": self.policy_relation,
            "sft_candidate": self.sft_candidate,
            "on_policy_rl_candidate": self.on_policy_rl_candidate,
            "missing_rl_capabilities": list(self.missing_rl_capabilities),
            "exclusion_reasons": list(self.exclusion_reasons),
            "source_event_ids": list(self.source_event_ids),
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


def build_training_candidate_view(
    episode: AgentEpisode,
    *,
    target_policy_model: str | None = None,
    target_policy: Any | None = None,
    eligibility: Any | None = None,
) -> TrainingCandidateView:
    """Build the view by consuming unified eligibility (fail-closed).

    The legacy ``target_policy_model`` string alone can never prove a complete
    policy fingerprint, so it only affects labelling and never grants on-policy
    RL eligibility on its own.  Provide a full ``target_policy`` fingerprint and
    real behavior token/logprob/mask/reward evidence for on-policy eligibility.
    """

    # lazy imports avoid a circular dependency with the contracts package
    from src.training.eligibility import evaluate_training_eligibility
    from src.validation.episode_semantics import certify_episode

    if eligibility is None:
        target_fp = None
        if target_policy is not None:
            target_fp = target_policy
        eligibility = evaluate_training_eligibility(
            episode, target_policy=target_fp
        )

    reasons: list[str] = list(eligibility.reasons)

    sft_candidate = eligibility.sft_eligible
    on_policy = eligibility.on_policy_rl_eligible

    if target_policy is not None:
        same_identity = target_policy.model_id == episode.model_manifest.model_id
    else:
        same_identity = target_policy_model == episode.model_manifest.model_id
    target_provided = target_policy is not None or target_policy_model is not None
    policy_relation = (
        "ON_POLICY" if on_policy else ("OFF_POLICY" if target_provided else "UNSPECIFIED")
    )
    training_role = "TARGET_POLICY" if same_identity else "TEACHER"

    cert = certify_episode(episode)
    capability_evidence = cert.capability_evidence
    missing = tuple(
        sorted(
            cap
            for cap in (
                CaptureCapability.MODEL_TOKEN_IDS.value,
                CaptureCapability.MODEL_LOGPROBS.value,
                "ACTION_MASK",
            )
            if capability_evidence.get(cap) != "EVIDENCE_PRESENT"
        )
    )

    return TrainingCandidateView(
        episode_id=episode.episode_id,
        episode_checksum=episode.checksum,
        behavior_provider=episode.model_manifest.provider,
        behavior_model=episode.model_manifest.model_id,
        behavior_model_revision=episode.model_manifest.revision,
        training_role=training_role,
        policy_relation=policy_relation,
        sft_candidate=sft_candidate,
        on_policy_rl_candidate=on_policy,
        missing_rl_capabilities=missing,
        exclusion_reasons=tuple(reasons),
        source_event_ids=tuple(event.event_id for event in episode.events),
    )
