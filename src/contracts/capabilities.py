"""Capability declarations and precondition checks."""

from __future__ import annotations

from enum import Enum
from typing import Iterable, TYPE_CHECKING

from src.errors import CapabilityMissingError

if TYPE_CHECKING:
    from src.contracts.rollout_record import RolloutRecord


class Capability(str, Enum):
    TOKEN_IDS = "token_ids"
    ACTION_MASK = "action_or_loss_mask"
    OLD_LOGPROBS = "old_logprobs"
    REWARD = "reward"
    GROUP_ID = "group_id"
    POLICY_VERSION = "policy_version"
    VERIFIER_EVIDENCE = "verifier_evidence"
    TOOL_EVENTS = "tool_events"


TRAINING_CORE_CAPABILITIES = frozenset(
    {
        Capability.TOKEN_IDS,
        Capability.ACTION_MASK,
        Capability.REWARD,
        Capability.GROUP_ID,
        Capability.POLICY_VERSION,
    }
)


def capabilities_for_record(record: "RolloutRecord") -> frozenset[Capability]:
    available: set[Capability] = set()
    if record.token_ids is not None:
        available.add(Capability.TOKEN_IDS)
    if record.action_mask is not None or record.loss_mask is not None:
        available.add(Capability.ACTION_MASK)
    if record.old_logprobs is not None:
        available.add(Capability.OLD_LOGPROBS)
    if record.reward is not None:
        available.add(Capability.REWARD)
    if record.group_id is not None:
        available.add(Capability.GROUP_ID)
    if record.policy_version is not None:
        available.add(Capability.POLICY_VERSION)
    if record.verifier_evidence_ref is not None:
        available.add(Capability.VERIFIER_EVIDENCE)
    if record.tool_events is not None:
        available.add(Capability.TOOL_EVENTS)
    return frozenset(available)


def common_capabilities(records: Iterable["RolloutRecord"]) -> frozenset[Capability]:
    iterator = iter(records)
    try:
        common = set(capabilities_for_record(next(iterator)))
    except StopIteration:
        return frozenset()
    for record in iterator:
        common.intersection_update(capabilities_for_record(record))
    return frozenset(common)


def require_capabilities(
    available: Iterable[Capability],
    required: Iterable[Capability],
    *,
    context: str = "consumer",
) -> None:
    missing = frozenset(required) - frozenset(available)
    if missing:
        raise CapabilityMissingError(missing, context)
