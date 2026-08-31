"""ARCHIVED: fail-closed trust helper for the retired GRPO prototype.

The GRPO experiment script is a GPU/bare-metal runner; its logic that governs
*which rollouts are trainable* and *whether the data is honestly certifiable as
on-policy* is isolated here so it can be unit-tested without GPU/model/torch.

Rules
-----
- A rollout infra exception must NOT be turned into reward 0; it is invalid and
  excluded from group baselines and advantage computation.
- Current temperature/top_p sampling with behavior logprobs recomputed offline
  at temp=1 is NOT exact behavior-policy evidence.  We fail closed:
  the artifacts are marked NOT_CERTIFIED_FOR_ON_POLICY_RL instead of being
  presented as strict on-policy RL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

CERTIFIED_FOR_ON_POLICY_RL = "CERTIFIED_FOR_ON_POLICY_RL"
NOT_CERTIFIED_FOR_ON_POLICY_RL = "NOT_CERTIFIED_FOR_ON_POLICY_RL"

ROLLOUT_TRUST_VERSION = "rollout-trust/v1"


@dataclass(frozen=True, slots=True)
class RolloutTrustReport:
    total: int
    valid: int
    invalid: int
    reasons: tuple[str, ...]
    on_policy_certification: str
    version: str = ROLLOUT_TRUST_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "total": self.total,
            "valid": self.valid,
            "invalid": self.invalid,
            "reasons": list(self.reasons),
            "on_policy_certification": self.on_policy_certification,
        }


def reward_with_exception(
    reward: float | None,
    exception: BaseException | None = None,
) -> tuple[float | None, bool]:
    """Return (reward, valid). Any infra exception invalidates the rollout.

    An exception may never be mapped to reward 0.0 for training.
    """
    if exception is not None:
        return None, False
    if reward is None:
        return None, False
    return reward, True


def evaluate_on_policy_certification(
    rollouts: Iterable[Mapping[str, Any]],
    *,
    exact_behavior_logprobs: bool,
) -> RolloutTrustReport:
    """Classify rollouts and decide on-policy certification (fail-closed).

    ``exact_behavior_logprobs`` is True only when behavior logprobs were
    computed by the exact sampling distribution (same temperature/top_p) that
    produced the tokens.  The current GRPO loop recomputes old logprobs at
    temp=1, so callers should pass False unless they actually fix that.
    """
    rollouts = list(rollouts)
    total = len(rollouts)
    valid = 0
    invalid = 0
    reasons: list[str] = []
    for rollout in rollouts:
        reward = rollout.get("reward")
        if reward is None:
            invalid += 1
        else:
            valid += 1

    if not exact_behavior_logprobs:
        reasons.append(
            "behavior logprobs not computed under the exact sampled "
            "temperature/top_p distribution"
        )
    if invalid:
        reasons.append(f"{invalid} rollout(s) invalid (infra exception) and excluded")

    if exact_behavior_logprobs and invalid == 0 and total > 0:
        certification = CERTIFIED_FOR_ON_POLICY_RL
    else:
        certification = NOT_CERTIFIED_FOR_ON_POLICY_RL
    return RolloutTrustReport(
        total=total,
        valid=valid,
        invalid=invalid,
        reasons=tuple(reasons),
        on_policy_certification=certification,
    )
