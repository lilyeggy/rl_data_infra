"""Per-agent training eligibility derived from explicit multi-agent evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.contracts._json import sha256_json
from src.contracts.agent_episode import (
    AgentEpisode,
    CaptureCapability,
    ExecutionValidity,
    IntegrityState,
    TaskStatus,
)
from src.contracts.trace_event import EventType
from src.multi_agent.graph import EvidenceStatus, MultiAgentExecutionGraph, RelationType


MULTI_AGENT_TRAINING_VERSION = "multi-agent-training-view/v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class MultiAgentTrainingView:
    episode_id: str
    episode_checksum: str
    agent_id: str
    role: str | None
    policy_id: str | None
    policy_revision: str | None
    policy_relation: str
    observation_status: EvidenceStatus
    action_status: EvidenceStatus
    reward_ownership_status: EvidenceStatus
    credit_assignment_status: EvidenceStatus
    sft_candidate: bool
    on_policy_rl_candidate: bool
    missing_capabilities: tuple[str, ...]
    exclusion_reasons: tuple[str, ...]
    source_event_ids: tuple[str, ...]
    view_version: str = MULTI_AGENT_TRAINING_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "view_version": self.view_version,
            "episode_id": self.episode_id,
            "episode_checksum": self.episode_checksum,
            "agent_id": self.agent_id,
            "role": self.role,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "policy_relation": self.policy_relation,
            "observation_status": self.observation_status.value,
            "action_status": self.action_status.value,
            "reward_ownership_status": self.reward_ownership_status.value,
            "credit_assignment_status": self.credit_assignment_status.value,
            "sft_candidate": self.sft_candidate,
            "on_policy_rl_candidate": self.on_policy_rl_candidate,
            "missing_capabilities": list(self.missing_capabilities),
            "exclusion_reasons": list(self.exclusion_reasons),
            "source_event_ids": list(self.source_event_ids),
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


def build_multi_agent_training_views(
    episode: AgentEpisode,
    graph: MultiAgentExecutionGraph,
    *,
    target_policy_by_agent: Mapping[str, str] | None = None,
) -> tuple[MultiAgentTrainingView, ...]:
    """Build conservative per-agent training views without inferring credit."""

    events = {event.event_id: event for event in episode.events}
    target_policies = target_policy_by_agent or {}
    reward_owners = {
        relation.source_agent_id
        for relation in graph.relations
        if relation.relation_type is RelationType.REWARD_OWNERSHIP
    }
    views: list[MultiAgentTrainingView] = []
    for node in graph.agents:
        owned_events = [events[event_id] for event_id in node.event_ids if event_id in events]
        has_model_request = any(
            event.event_type is EventType.MODEL_REQUEST for event in owned_events
        )
        has_action = any(
            event.event_type in {EventType.MODEL_RESPONSE, EventType.TOOL_CALL}
            for event in owned_events
        )
        observation_status = (
            EvidenceStatus.OBSERVED
            if has_model_request
            else EvidenceStatus.NOT_OBSERVABLE
        )
        action_status = EvidenceStatus.OBSERVED if has_action else EvidenceStatus.NOT_OBSERVABLE
        if graph.graph_status in {EvidenceStatus.INVALID, EvidenceStatus.NOT_OBSERVABLE}:
            observation_status = EvidenceStatus.INSUFFICIENT_EVIDENCE
            action_status = EvidenceStatus.INSUFFICIENT_EVIDENCE

        credit_relations = [
            relation
            for relation in graph.relations
            if relation.relation_type is RelationType.CREDIT_ASSIGNMENT
            and (
                relation.source_agent_id == node.agent_id
                or relation.target_agent_id == node.agent_id
            )
        ]
        if graph.field_status.get("credit_assignment") is EvidenceStatus.OBSERVED:
            credit_status = (
                EvidenceStatus.OBSERVED
                if credit_relations
                else EvidenceStatus.NOT_OBSERVABLE
            )
        else:
            credit_status = graph.field_status.get(
                "credit_assignment", EvidenceStatus.NOT_OBSERVABLE
            )

        if graph.field_status.get("reward_ownership") is EvidenceStatus.OBSERVED:
            reward_status = (
                EvidenceStatus.OBSERVED
                if node.agent_id in reward_owners
                else EvidenceStatus.NOT_OBSERVABLE
            )
        else:
            reward_status = graph.field_status.get(
                "reward_ownership", EvidenceStatus.NOT_OBSERVABLE
            )

        target_policy = target_policies.get(node.agent_id)
        if target_policy is None:
            policy_relation = "UNKNOWN"
        elif node.policy_id == target_policy:
            policy_relation = "ON_POLICY"
        else:
            policy_relation = "OFF_POLICY"

        missing: list[str] = []
        reasons: list[str] = []
        base_eligible = True
        if episode.integrity.state is not IntegrityState.COMPLETE:
            base_eligible = False
            reasons.append("episode integrity is not COMPLETE")
        if episode.outcome.execution_validity is not ExecutionValidity.VALID:
            base_eligible = False
            reasons.append("episode execution is not VALID")
        if episode.outcome.task_status is not TaskStatus.SUCCESS:
            base_eligible = False
            reasons.append("episode task is not SUCCESS")
        if graph.graph_status is not EvidenceStatus.OBSERVED:
            base_eligible = False
            reasons.append(f"multi-agent graph is {graph.graph_status.value}")
        if observation_status is not EvidenceStatus.OBSERVED:
            base_eligible = False
            reasons.append("agent observation ownership is not observable")
        if action_status is not EvidenceStatus.OBSERVED:
            base_eligible = False
            reasons.append("agent action ownership is not observable")

        sft_candidate = base_eligible
        required_rl = (
            CaptureCapability.MODEL_TOKEN_IDS,
            CaptureCapability.MODEL_LOGPROBS,
        )
        for capability in required_rl:
            if capability not in episode.capabilities:
                missing.append(capability.value)
        if missing:
            reasons.append("agent token IDs/logprobs are unavailable")
        if policy_relation != "ON_POLICY":
            reasons.append("agent policy does not match an explicitly supplied target policy")
        if reward_status is not EvidenceStatus.OBSERVED:
            reasons.append("agent-level reward ownership is not observable")
        if credit_status is not EvidenceStatus.OBSERVED:
            reasons.append("agent-level credit assignment is not observable")
        on_policy = (
            sft_candidate
            and policy_relation == "ON_POLICY"
            and reward_status is EvidenceStatus.OBSERVED
            and credit_status is EvidenceStatus.OBSERVED
            and not missing
        )
        if not on_policy:
            if not sft_candidate:
                reasons.append("agent is not an eligible complete SFT trajectory")
        views.append(
            MultiAgentTrainingView(
                episode_id=episode.episode_id,
                episode_checksum=episode.checksum,
                agent_id=node.agent_id,
                role=node.role,
                policy_id=node.policy_id,
                policy_revision=node.policy_revision,
                policy_relation=policy_relation,
                observation_status=observation_status,
                action_status=action_status,
                reward_ownership_status=reward_status,
                credit_assignment_status=credit_status,
                sft_candidate=sft_candidate,
                on_policy_rl_candidate=on_policy,
                missing_capabilities=tuple(sorted(set(missing))),
                exclusion_reasons=tuple(dict.fromkeys(reasons)),
                source_event_ids=node.event_ids,
            )
        )
    return tuple(sorted(views, key=lambda item: item.agent_id))
