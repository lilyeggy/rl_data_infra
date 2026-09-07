"""Derived multi-agent causal graph over a canonical AgentEpisode."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from src.contracts._json import freeze_json, sha256_json, thaw_json
from src.contracts.agent_episode import AgentEpisode
from src.contracts.trace_event import EventType, TraceEvent
from src.errors import ContractValidationError


MULTI_AGENT_GRAPH_VERSION = "multi-agent-execution-graph/v1"


class EvidenceStatus(str, Enum):
    OBSERVED = "OBSERVED"
    NOT_OBSERVABLE = "NOT_OBSERVABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INVALID = "INVALID"


class RelationType(str, Enum):
    PARENT_CHILD = "PARENT_CHILD"
    MESSAGE_ROUTE = "MESSAGE_ROUTE"
    TOOL_OWNERSHIP = "TOOL_OWNERSHIP"
    REWARD_OWNERSHIP = "REWARD_OWNERSHIP"
    CREDIT_ASSIGNMENT = "CREDIT_ASSIGNMENT"


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentNode:
    agent_id: str
    parent_agent_id: str | None
    role: str | None
    policy_id: str | None
    policy_revision: str | None
    event_ids: tuple[str, ...]
    branch_ids: tuple[str, ...] = ()
    parallel_group_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise ContractValidationError("agent_id must be a non-empty string")
        for name in ("parent_agent_id", "role", "policy_id", "policy_revision"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ContractValidationError(f"{name} must be non-empty when provided")
        for name, values in (
            ("event_ids", self.event_ids),
            ("branch_ids", self.branch_ids),
            ("parallel_group_ids", self.parallel_group_ids),
        ):
            normalized = tuple(values)
            if any(not isinstance(item, str) or not item.strip() for item in normalized):
                raise ContractValidationError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, tuple(sorted(set(normalized))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "parent_agent_id": self.parent_agent_id,
            "role": self.role,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "event_ids": list(self.event_ids),
            "branch_ids": list(self.branch_ids),
            "parallel_group_ids": list(self.parallel_group_ids),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentRelation:
    relation_type: RelationType
    source_agent_id: str
    target_agent_id: str | None
    evidence_event_ids: tuple[str, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.relation_type, RelationType):
            raise ContractValidationError("relation_type must be a RelationType")
        for name in ("source_agent_id", "target_agent_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ContractValidationError(f"{name} must be non-empty when provided")
        event_ids = tuple(self.evidence_event_ids)
        if any(not isinstance(item, str) or not item.strip() for item in event_ids):
            raise ContractValidationError("evidence_event_ids must contain non-empty strings")
        object.__setattr__(self, "evidence_event_ids", tuple(sorted(set(event_ids))))
        if not isinstance(self.attributes, Mapping):
            raise ContractValidationError("attributes must be an object")
        object.__setattr__(self, "attributes", freeze_json(self.attributes, "$.attributes"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_type": self.relation_type.value,
            "source_agent_id": self.source_agent_id,
            "target_agent_id": self.target_agent_id,
            "evidence_event_ids": list(self.evidence_event_ids),
            "attributes": thaw_json(self.attributes),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class MultiAgentExecutionGraph:
    episode_id: str
    episode_checksum: str
    graph_status: EvidenceStatus
    field_status: Mapping[str, EvidenceStatus]
    agents: tuple[AgentNode, ...]
    relations: tuple[AgentRelation, ...]
    unowned_event_ids: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    source_event_ids: tuple[str, ...]
    graph_version: str = MULTI_AGENT_GRAPH_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.graph_status, EvidenceStatus):
            raise ContractValidationError("graph_status must be an EvidenceStatus")
        if not isinstance(self.field_status, Mapping):
            raise ContractValidationError("field_status must be an object")
        statuses = {
            str(key): value if isinstance(value, EvidenceStatus) else EvidenceStatus(value)
            for key, value in self.field_status.items()
        }
        object.__setattr__(self, "field_status", MappingProxyType(statuses))
        agents = tuple(self.agents)
        if any(not isinstance(item, AgentNode) for item in agents):
            raise ContractValidationError("agents must contain AgentNode values")
        if len({item.agent_id for item in agents}) != len(agents):
            raise ContractValidationError("agents must have unique agent_id values")
        object.__setattr__(self, "agents", tuple(sorted(agents, key=lambda item: item.agent_id)))
        relations = tuple(self.relations)
        if any(not isinstance(item, AgentRelation) for item in relations):
            raise ContractValidationError("relations must contain AgentRelation values")
        object.__setattr__(self, "relations", relations)
        for name in ("unowned_event_ids", "missing_evidence", "source_event_ids"):
            values = tuple(getattr(self, name))
            if any(not isinstance(item, str) or not item.strip() for item in values):
                raise ContractValidationError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, tuple(sorted(set(values))))
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ContractValidationError("episode_id must be a non-empty string")
        if not isinstance(self.episode_checksum, str) or len(self.episode_checksum) != 64:
            raise ContractValidationError("episode_checksum must be a SHA256 checksum")

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_version": self.graph_version,
            "episode_id": self.episode_id,
            "episode_checksum": self.episode_checksum,
            "graph_status": self.graph_status.value,
            "field_status": {
                key: value.value for key, value in self.field_status.items()
            },
            "agents": [agent.to_dict() for agent in self.agents],
            "relations": [relation.to_dict() for relation in self.relations],
            "unowned_event_ids": list(self.unowned_event_ids),
            "missing_evidence": list(self.missing_evidence),
            "source_event_ids": list(self.source_event_ids),
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


def build_multi_agent_execution_graph(
    episode: AgentEpisode,
) -> MultiAgentExecutionGraph:
    """Build a graph only from explicit per-event Agent metadata."""

    events = tuple(episode.events)
    source_event_ids = tuple(event.event_id for event in events)
    records: dict[str, dict[str, Any]] = {}
    missing_evidence: list[str] = []
    unowned: list[str] = []
    invalid = False
    required_owner_types = {
        EventType.MODEL_REQUEST,
        EventType.MODEL_RESPONSE,
        EventType.TOOL_CALL,
        EventType.TOOL_RESULT,
    }
    for event in events:
        attributes = event.attributes
        agent_id = attributes.get("agent_id")
        if not isinstance(agent_id, str) or not agent_id.strip():
            if event.event_type in required_owner_types:
                unowned.append(event.event_id)
            continue
        record = records.setdefault(
            agent_id,
            {
                "event_ids": [],
                "parent_values": [],
                "roles": [],
                "policy_ids": [],
                "policy_revisions": [],
                "branch_ids": [],
                "parallel_group_ids": [],
            },
        )
        record["event_ids"].append(event.event_id)
        if "parent_agent_id" in attributes:
            parent = attributes.get("parent_agent_id")
            if parent is not None and (not isinstance(parent, str) or not parent.strip()):
                invalid = True
            else:
                record["parent_values"].append(parent)
        else:
            missing_evidence.append(f"event:{event.event_id}:parent_agent_id")
        for key, target in (
            ("agent_role", "roles"),
            ("policy_id", "policy_ids"),
            ("policy_revision", "policy_revisions"),
            ("branch_id", "branch_ids"),
            ("parallel_group_id", "parallel_group_ids"),
        ):
            value = attributes.get(key)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    invalid = True
                else:
                    record[target].append(value)

    if not records:
        return MultiAgentExecutionGraph(
            episode_id=episode.episode_id,
            episode_checksum=episode.checksum,
            graph_status=EvidenceStatus.NOT_OBSERVABLE,
            field_status={
                "agent_identity": EvidenceStatus.NOT_OBSERVABLE,
                "parent_child": EvidenceStatus.NOT_OBSERVABLE,
                "message_routing": EvidenceStatus.NOT_OBSERVABLE,
                "branch_concurrency": EvidenceStatus.NOT_OBSERVABLE,
                "reward_ownership": EvidenceStatus.NOT_OBSERVABLE,
                "credit_assignment": EvidenceStatus.NOT_OBSERVABLE,
                "policy_identity": EvidenceStatus.NOT_OBSERVABLE,
            },
            agents=(),
            relations=(),
            unowned_event_ids=tuple(unowned),
            missing_evidence=("no explicit agent_id was observed",),
            source_event_ids=source_event_ids,
        )

    agent_ids = set(records)
    parent_status = EvidenceStatus.OBSERVED
    role_status = EvidenceStatus.OBSERVED
    policy_status = EvidenceStatus.OBSERVED
    branch_values: set[str] = set()
    parallel_group_values: set[str] = set()
    nodes: list[AgentNode] = []
    parent_by_agent: dict[str, str | None] = {}
    for agent_id in sorted(records):
        record = records[agent_id]
        parents = set(record["parent_values"])
        if not record["parent_values"]:
            parent_status = EvidenceStatus.INSUFFICIENT_EVIDENCE
            missing_evidence.append(f"agent:{agent_id}:parent_agent_id")
            parent = None
        elif len(parents) > 1:
            invalid = True
            parent_status = EvidenceStatus.INVALID
            parent = sorted(str(item) for item in parents)[0]
        else:
            parent = next(iter(parents))
        if parent is not None and parent not in agent_ids:
            invalid = True
            parent_status = EvidenceStatus.INVALID
        parent_by_agent[agent_id] = parent
        roles = set(record["roles"])
        policies = set(record["policy_ids"])
        revisions = set(record["policy_revisions"])
        if len(roles) > 1 or len(policies) > 1 or len(revisions) > 1:
            invalid = True
        if not roles:
            role_status = EvidenceStatus.NOT_OBSERVABLE
        if not policies or not revisions:
            policy_status = EvidenceStatus.NOT_OBSERVABLE
        branch_values.update(record["branch_ids"])
        parallel_group_values.update(record["parallel_group_ids"])
        nodes.append(
            AgentNode(
                agent_id=agent_id,
                parent_agent_id=parent,
                role=sorted(roles)[0] if roles else None,
                policy_id=sorted(policies)[0] if policies else None,
                policy_revision=sorted(revisions)[0] if revisions else None,
                event_ids=tuple(record["event_ids"]),
                branch_ids=tuple(record["branch_ids"]),
                parallel_group_ids=tuple(record["parallel_group_ids"]),
            )
        )

    identity_status = (
        EvidenceStatus.INSUFFICIENT_EVIDENCE
        if unowned
        else EvidenceStatus.OBSERVED
    )
    message_relations: list[AgentRelation] = []
    reward_relations: list[AgentRelation] = []
    credit_relations: list[AgentRelation] = []
    tool_relations: list[AgentRelation] = []
    message_status = EvidenceStatus.NOT_OBSERVABLE
    reward_status = EvidenceStatus.NOT_OBSERVABLE
    credit_status = EvidenceStatus.NOT_OBSERVABLE
    for event in events:
        attributes = event.attributes
        agent_id = attributes.get("agent_id")
        if isinstance(agent_id, str) and event.event_type is EventType.TOOL_CALL:
            tool_relations.append(
                AgentRelation(
                    relation_type=RelationType.TOOL_OWNERSHIP,
                    source_agent_id=agent_id,
                    target_agent_id=None,
                    evidence_event_ids=(event.event_id,),
                    attributes={"tool_name": attributes.get("tool_name")},
                )
            )
        recipient = attributes.get("message_to_agent_id")
        if recipient is not None:
            message_status = EvidenceStatus.OBSERVED
            if not isinstance(agent_id, str) or recipient not in agent_ids:
                invalid = True
            else:
                message_relations.append(
                    AgentRelation(
                        relation_type=RelationType.MESSAGE_ROUTE,
                        source_agent_id=agent_id,
                        target_agent_id=recipient,
                        evidence_event_ids=(event.event_id,),
                        attributes={"message_id": attributes.get("message_id")},
                    )
                )
        credit_target = attributes.get("credit_to_agent_id")
        if credit_target is not None:
            credit_status = EvidenceStatus.OBSERVED
            if (
                not isinstance(agent_id, str)
                or not isinstance(credit_target, str)
                or credit_target not in agent_ids
            ):
                invalid = True
                credit_status = EvidenceStatus.INVALID
            else:
                credit_relations.append(
                    AgentRelation(
                        relation_type=RelationType.CREDIT_ASSIGNMENT,
                        source_agent_id=agent_id,
                        target_agent_id=credit_target,
                        evidence_event_ids=(event.event_id,),
                        attributes={"credit_weight": attributes.get("credit_weight")},
                    )
                )
        owner = attributes.get("reward_owner_agent_id")
        if owner is not None:
            reward_status = EvidenceStatus.OBSERVED
            if not isinstance(owner, str) or owner not in agent_ids:
                invalid = True
            else:
                reward_relations.append(
                    AgentRelation(
                        relation_type=RelationType.REWARD_OWNERSHIP,
                        source_agent_id=owner,
                        target_agent_id=None,
                        evidence_event_ids=(event.event_id,),
                        attributes={"reward_scope": attributes.get("reward_scope")},
                    )
                )

    parent_relations = [
        AgentRelation(
            relation_type=RelationType.PARENT_CHILD,
            source_agent_id=parent,
            target_agent_id=agent_id,
            evidence_event_ids=tuple(records[agent_id]["event_ids"]),
        )
        for agent_id, parent in sorted(parent_by_agent.items())
        if parent is not None
    ]
    cycle = _has_parent_cycle(parent_by_agent)
    if cycle:
        invalid = True
        parent_status = EvidenceStatus.INVALID
        missing_evidence.append("parent-child cycle detected")

    if not branch_values or not parallel_group_values:
        branch_status = EvidenceStatus.NOT_OBSERVABLE
    elif any(
        not record["branch_ids"] or not record["parallel_group_ids"]
        for record in records.values()
    ):
        branch_status = EvidenceStatus.INSUFFICIENT_EVIDENCE
    else:
        branch_status = EvidenceStatus.OBSERVED

    if invalid:
        graph_status = EvidenceStatus.INVALID
    elif unowned or parent_status is EvidenceStatus.INSUFFICIENT_EVIDENCE:
        graph_status = EvidenceStatus.INSUFFICIENT_EVIDENCE
    else:
        graph_status = EvidenceStatus.OBSERVED
    return MultiAgentExecutionGraph(
        episode_id=episode.episode_id,
        episode_checksum=episode.checksum,
        graph_status=graph_status,
        field_status={
            "agent_identity": identity_status,
            "parent_child": parent_status,
            "message_routing": message_status,
            "branch_concurrency": branch_status,
            "reward_ownership": reward_status,
            "policy_identity": policy_status,
            "tool_ownership": (
                EvidenceStatus.OBSERVED if tool_relations else EvidenceStatus.NOT_OBSERVABLE
            ),
            "credit_assignment": credit_status,
        },
        agents=tuple(nodes),
        relations=tuple(
            parent_relations
            + message_relations
            + tool_relations
            + reward_relations
            + credit_relations
        ),
        unowned_event_ids=tuple(unowned),
        missing_evidence=tuple(missing_evidence),
        source_event_ids=source_event_ids,
    )


def _has_parent_cycle(parents: Mapping[str, str | None]) -> bool:
    for start in parents:
        seen: set[str] = set()
        current: str | None = start
        while current is not None:
            if current in seen:
                return True
            seen.add(current)
            current = parents.get(current)
    return False
