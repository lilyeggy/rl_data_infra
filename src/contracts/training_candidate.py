"""Non-mutating training eligibility view over a canonical Episode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.contracts._json import sha256_json
from src.contracts.agent_episode import (
    AgentEpisode,
    CaptureCapability,
    ExecutionValidity,
    IntegrityState,
    TaskStatus,
)


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
) -> TrainingCandidateView:
    reasons: list[str] = []
    if episode.integrity.state is not IntegrityState.COMPLETE:
        reasons.append("integrity is not COMPLETE")
    if episode.outcome.execution_validity is not ExecutionValidity.VALID:
        reasons.append("execution is not VALID")
    if episode.outcome.task_status is not TaskStatus.SUCCESS:
        reasons.append("task is not SUCCESS")
    sft_candidate = not reasons
    required = {CaptureCapability.MODEL_TOKEN_IDS, CaptureCapability.MODEL_LOGPROBS}
    missing = tuple(sorted(item.value for item in required - episode.capabilities))
    same_policy = target_policy_model is not None and target_policy_model == episode.model_manifest.model_id
    if not same_policy:
        reasons.append("behavior model differs from target policy or target policy is unspecified")
    if missing:
        reasons.append("exact behavior token IDs/logprobs are unavailable")
    on_policy = sft_candidate and same_policy and not missing
    return TrainingCandidateView(
        episode_id=episode.episode_id,
        episode_checksum=episode.checksum,
        behavior_provider=episode.model_manifest.provider,
        behavior_model=episode.model_manifest.model_id,
        behavior_model_revision=episode.model_manifest.revision,
        training_role="TEACHER" if not same_policy else "TARGET_POLICY",
        policy_relation="ON_POLICY" if same_policy else "OFF_POLICY",
        sft_candidate=sft_candidate,
        on_policy_rl_candidate=on_policy,
        missing_rl_capabilities=missing,
        exclusion_reasons=tuple(reasons),
        source_event_ids=tuple(event.event_id for event in episode.events),
    )
