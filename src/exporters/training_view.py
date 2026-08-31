"""Versioned offline projection from AgentEpisode to the training view.

This layer converts canonical episodes into `RolloutRecord` values for offline
SFT/distillation and into preference pairs for DPO/SimPO-style optimization.

Honest boundaries:
- token IDs, masks and logprobs are NOT synthesized; absent training fields stay
  ``None`` and are surfaced by capability gates.
- infrastructure-invalid episodes never receive a fabricated reward.
- these projections are offline teacher candidates, never on-policy rollouts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from src.contracts._json import sha256_json
from src.contracts.agent_episode import (
    AgentEpisode,
    EpisodeVerifierStatus,
    ExecutionValidity,
    IntegrityState,
    TaskStatus,
)
from src.contracts.rollout_record import (
    RolloutRecord,
    RolloutStatus,
    VerifierStatus,
)
from src.contracts.trace_event import EventType, TraceEvent
from src.training.policy_fingerprint import PolicyFingerprint
from src.validation.episode_semantics import (
    certified_verifier_status,
    certify_episode,
)

TRAINING_VIEW_PROJECTION_VERSION = "episode-rollout-projection/v1"


def _verifier_status(episode: AgentEpisode) -> VerifierStatus:
    return {
        "PASSED": VerifierStatus.PASSED,
        "FAILED": VerifierStatus.FAILED,
        "ERROR": VerifierStatus.ERROR,
        "TIMEOUT": VerifierStatus.TIMEOUT,
        "NOT_RUN": VerifierStatus.NOT_RUN,
    }.get(episode.outcome.verifier_status.value, VerifierStatus.UNKNOWN)


def _rollout_status(episode: AgentEpisode) -> RolloutStatus:
    if episode.outcome.execution_validity is ExecutionValidity.INFRA_INVALID:
        return RolloutStatus.INCOMPLETE
    if episode.outcome.task_status is TaskStatus.SUCCESS:
        return RolloutStatus.COMPLETED
    if episode.outcome.task_status is TaskStatus.FAILURE:
        return RolloutStatus.FAILED
    return RolloutStatus.UNKNOWN


def _reward(episode: AgentEpisode) -> float | None:
    if episode.outcome.execution_validity is not ExecutionValidity.VALID:
        return None
    return episode.outcome.score


def _tool_events(events: tuple[TraceEvent, ...]) -> tuple[Mapping[str, Any], ...]:
    result: list[Mapping[str, Any]] = []
    for event in events:
        if event.event_type not in {EventType.TOOL_CALL, EventType.TOOL_RESULT}:
            continue
        result.append(
            {
                "event_type": event.event_type.value,
                "event_id": event.event_id,
                "tool_name": event.attributes.get("tool_name"),
                "status": event.status.value,
                "attributes": event.attributes,
            }
        )
    return tuple(result)


def project_episode_to_rollout(episode: AgentEpisode) -> RolloutRecord:
    """Project one canonical episode into an offline training-view rollout.

    Missing training fields (token ids, masks, logprobs) remain ``None``; a
    consumer that requires them must use a capability gate and will fail
    explicitly instead of receiving fabricated values.
    """

    source_identity = f"episode:{episode.episode_id}:{episode.checksum}"
    record = RolloutRecord(
        trajectory_id=episode.episode_id,
        task_id=episode.task_id,
        source_type="agent-episode",
        source_record_id=source_identity,
        model_id=episode.model_manifest.model_id,
        model_revision=episode.model_manifest.revision,
        tokenizer_revision=episode.model_manifest.tokenizer_revision,
        reward=_reward(episode),
        rollout_status=_rollout_status(episode),
        termination_reason=episode.termination.reason,
        verifier_status=_verifier_status(episode),
        verifier_evidence_ref=(
            f"episode:{episode.episode_id}:verifier:{episode.checksum}"
        ),
        started_at=episode.started_at,
        ended_at=episode.ended_at,
        tool_events=_tool_events(episode.events),
        opaque_metadata={
            "episode_id": episode.episode_id,
            "episode_checksum": episode.checksum,
            "projection_version": TRAINING_VIEW_PROJECTION_VERSION,
            "behavior_policy": (
                f"{episode.model_manifest.provider}/{episode.model_manifest.model_id}"
            ),
            "harness": episode.harness_manifest.name,
            "harness_version": episode.harness_manifest.version,
            "execution_validity": episode.outcome.execution_validity.value,
            "integrity_state": episode.integrity.state.value,
        },
    )
    return record.with_source_envelope(
        source_type="agent-episode",
        source_record_id=episode.episode_id,
        source_payload_ref=f"episode:{episode.episode_id}",
        source_payload_sha256=episode.checksum,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class TrajectoryPreferencePair:
    """One same-task, same-identity preference pair.

    ``rejection_reasons`` is empty for a formed pair; otherwise it carries the
    stable reasons why the group did NOT form a pair.
    """

    pair_key: str
    chosen_episode_id: str
    rejected_episode_id: str
    chosen_rollout_checksum: str
    rejected_rollout_checksum: str
    chosen_verifier: str
    rejected_verifier: str
    reason: str
    rejection_reasons: tuple[str, ...] = ()
    schema_version: str = "trajectory-preference-pair/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pair_key": self.pair_key,
            "chosen_episode_id": self.chosen_episode_id,
            "rejected_episode_id": self.rejected_episode_id,
            "chosen_rollout_checksum": self.chosen_rollout_checksum,
            "rejected_rollout_checksum": self.rejected_rollout_checksum,
            "chosen_verifier": self.chosen_verifier,
            "rejected_verifier": self.rejected_verifier,
            "reason": self.reason,
            "rejection_reasons": list(self.rejection_reasons),
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True, kw_only=True)
class PreferencePairRejection:
    """Stable, serializable rejection of a candidate preference group."""

    pair_key: str
    identity_mismatch: tuple[str, ...]
    missing_chosen: bool
    missing_rejected: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_key": self.pair_key,
            "identity_mismatch": list(self.identity_mismatch),
            "missing_chosen": self.missing_chosen,
            "missing_rejected": self.missing_rejected,
            "reasons": list(self.reasons),
        }


def _pair_identity(episode: AgentEpisode) -> tuple[tuple[str, str | None], ...]:
    """Canonical, testable identity required for chosen/rejected pairing."""
    fp = PolicyFingerprint.from_model_manifest(episode.model_manifest)
    return (
        ("task_id", episode.task_id),
        ("task_revision", episode.environment_manifest.task_snapshot),
        ("attempt", str(episode.attempt)),
        ("model_provider", episode.model_manifest.provider),
        ("model_id", episode.model_manifest.model_id),
        ("model_revision", episode.model_manifest.revision),
        ("tokenizer_revision", episode.model_manifest.tokenizer_revision),
        ("environment_manifest", sha256_json(episode.environment_manifest.to_dict())),
        ("evaluator_manifest", sha256_json(episode.evaluator_manifest.to_dict())),
        ("tool_schema", episode.harness_manifest.config_digest),
        ("policy_fingerprint", fp.checksum()),
    )


def _certified_verifier_ok(episode: AgentEpisode) -> bool:
    """Integrity COMPLETE + certified + real verifier attestation."""
    if episode.integrity.state is not IntegrityState.COMPLETE:
        return False
    cert = certify_episode(episode)
    return cert.is_certified and cert.verifier_attested


def build_preference_pairs_with_reasons(
    episodes: Iterable[AgentEpisode],
    *,
    pair_key_of: Any = None,
) -> tuple[tuple[TrajectoryPreferencePair, ...], tuple[PreferencePairRejection, ...]]:
    """Build verified preference pairs and stable rejection reasons.

    A pair is produced only when every required identity dimension matches and
    the group contains a certified verifier-PASSED (chosen) and a certified
    verifier-FAILED valid failure (rejected).  Otherwise the group is rejected
    with stable, serializable reasons.
    """

    groups: dict[str, list[AgentEpisode]] = {}
    for episode in episodes:
        key = pair_key_of(episode) if pair_key_of is not None else episode.task_id
        groups.setdefault(str(key), []).append(episode)

    pairs: list[TrajectoryPreferencePair] = []
    rejections: list[PreferencePairRejection] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda item: item.episode_id)
        rejection_reasons: list[str] = []

        # identical identity across the whole group
        identities = {_pair_identity(item) for item in group}
        privilege = []
        if len(identities) != 1:
            for name in ("task_id", "task_revision", "model_id", "model_revision", "tokenizer_revision", "policy_fingerprint", "environment_manifest", "evaluator_manifest", "tool_schema"):
                values = {
                    dict(one).get(name)
                    for one in identities
                }
                if len(values) > 1:
                    privilege.append(name)
            identity_mismatch = tuple(sorted(privilege))
            rejection_reasons.append(f"identity mismatch on {', '.join(identity_mismatch)}")
        else:
            identity_mismatch = ()

        chosen: AgentEpisode | None = None
        rejected: AgentEpisode | None = None
        for item in group:
            if not _certified_verifier_ok(item):
                continue
            if (
                item.outcome.execution_validity is ExecutionValidity.VALID
                and item.outcome.task_status is TaskStatus.SUCCESS
                and certified_verifier_status(item) is EpisodeVerifierStatus.PASSED
                and chosen is None
            ):
                chosen = item
            elif (
                item.outcome.execution_validity is ExecutionValidity.VALID
                and item.outcome.task_status is TaskStatus.FAILURE
                and certified_verifier_status(item) is EpisodeVerifierStatus.FAILED
                and rejected is None
            ):
                rejected = item

        if chosen is None:
            rejection_reasons.append("no certified verifier-PASSED success episode")
        if rejected is None:
            rejection_reasons.append("no certified verifier-FAILED valid failure episode")

        if chosen is not None and rejected is not None and not identity_mismatch:
            pairs.append(
                TrajectoryPreferencePair(
                    pair_key=key,
                    chosen_episode_id=chosen.episode_id,
                    rejected_episode_id=rejected.episode_id,
                    chosen_rollout_checksum=project_episode_to_rollout(chosen).checksum,
                    rejected_rollout_checksum=project_episode_to_rollout(
                        rejected
                    ).checksum,
                    chosen_verifier=chosen.outcome.verifier_status.value,
                    rejected_verifier=rejected.outcome.verifier_status.value,
                    reason=(
                        "same task/revision/model/policy/env/evaluator/tool identity; "
                        "chosen verifier PASSED and rejected verifier FAILED, both "
                        "certified and integrity COMPLETE"
                    ),
                )
            )
            continue

        rejections.append(
            PreferencePairRejection(
                pair_key=key,
                identity_mismatch=identity_mismatch,
                missing_chosen=chosen is None,
                missing_rejected=rejected is None,
                reasons=tuple(dict.fromkeys(rejection_reasons)),
            )
        )
    return tuple(pairs), tuple(rejections)


def build_preference_pairs(
    episodes: Iterable[AgentEpisode],
    *,
    pair_key_of: Any = None,
) -> tuple[TrajectoryPreferencePair, ...]:
    """Backward-compatible wrapper: confirmed pairs only."""
    pairs, _ = build_preference_pairs_with_reasons(episodes, pair_key_of=pair_key_of)
    return pairs
