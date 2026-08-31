"""Unified fail-closed training eligibility tests."""

from __future__ import annotations

import unittest
from dataclasses import replace

from src.assembly.episode_assembler import EpisodeAssembler
from src.contracts.agent_episode import (
    EpisodeOutcome,
    EpisodeVerifierStatus,
    ExecutionValidity,
    TaskStatus,
)
from src.contracts.trace_event import EventStatus, EventType
from src.training.eligibility import (
    TrainingEligibilityStatus,
    evaluate_training_eligibility,
)
from src.training.policy_fingerprint import PolicyFingerprint
from tests.execution_fixtures import make_complete_event_stream, make_episode_context


def _assemble(events, episode_id="ep-elig", task_id="task-elig-1"):
    return EpisodeAssembler().assemble(
        events, contexts={episode_id: make_episode_context(task_id=task_id)}
    ).episodes[0]


def _with_fields(episode, token_ids, logprobs=None, action_mask=None):
    events = []
    for event in episode.events:
        if event.event_type is EventType.MODEL_RESPONSE:
            attrs = dict(event.attributes)
            if token_ids is not None:
                attrs["token_ids"] = list(token_ids)
            if logprobs is not None:
                attrs["logprobs"] = list(logprobs)
            if action_mask is not None:
                attrs["action_mask"] = list(action_mask)
            events.append(replace(event, attributes=attrs))
        else:
            events.append(event)
    return replace(episode, events=tuple(events))


def _full_fp(**overrides):
    base = {
        "provider": "fixture-provider",
        "model_id": "fixture-model",
        "base_model_revision": "fixture-model-r1",
        "adapter_revision": "adapter-r1",
        "tokenizer_revision": "fixture-tokenizer-r1",
        "chat_template_checksum": "chat-c1",
        "tool_schema_checksum": "tools-c1",
        "sampling_config": {"temperature": 0.0},
        "temperature": 0.0,
        "top_p": 1.0,
        "policy_generation": "gen-1",
    }
    base.update(overrides)
    return PolicyFingerprint(**base)


def _base_episode(episode_id="ep-elig"):
    return _assemble(make_complete_event_stream(episode_id=episode_id), episode_id)


class TrainingEligibilityTest(unittest.TestCase):
    def test_normal_certified_sft_episode_is_sft_eligible(self) -> None:
        ep = _base_episode("ep-sft")
        eligibility = evaluate_training_eligibility(ep)
        self.assertIs(eligibility.status, TrainingEligibilityStatus.SFT_ELIGIBLE)
        self.assertTrue(eligibility.sft_eligible)
        self.assertFalse(eligibility.on_policy_rl_eligible)

    def test_declared_logprobs_without_array_is_not_rl(self) -> None:
        ep = _with_fields(_base_episode("ep-nologp"), token_ids=[1, 2, 3], action_mask=[1, 1, 1])
        eligibility = evaluate_training_eligibility(ep)
        self.assertTrue(eligibility.sft_eligible)
        self.assertFalse(eligibility.on_policy_rl_eligible)
        self.assertTrue(
            any("no real behavior logprob evidence" in r for r in eligibility.reasons)
        )

    def test_token_logprob_length_mismatch_is_not_rl(self) -> None:
        ep = _with_fields(
            _base_episode("ep-misalign"),
            token_ids=[1, 2, 3, 4],
            logprobs=[-0.1, -0.2, -0.3],  # shorter -> misaligned
            action_mask=[1, 1, 1, 1],
        )
        eligibility = evaluate_training_eligibility(ep)
        self.assertFalse(eligibility.on_policy_rl_eligible)
        self.assertTrue(any("not aligned" in r for r in eligibility.reasons))

    def test_same_model_different_adapter_rejected_on_policy(self) -> None:
        ep = _with_fields(
            _base_episode("ep-adapter"),
            token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3],
            action_mask=[1, 1, 1],
        )
        target = _full_fp(model_id="fixture-model")
        behavior = _full_fp(model_id="fixture-model", adapter_revision="adapter-OTHER")
        eligibility = evaluate_training_eligibility(
            ep, target_policy=target, behavior_policy=behavior
        )
        self.assertFalse(eligibility.on_policy_rl_eligible)

    def test_temperature_difference_rejected_on_policy(self) -> None:
        ep = _with_fields(
            _base_episode("ep-temp"),
            token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3],
            action_mask=[1, 1, 1],
        )
        target = _full_fp(temperature=0.0)
        behavior = _full_fp(temperature=0.8)
        eligibility = evaluate_training_eligibility(
            ep, target_policy=target, behavior_policy=behavior
        )
        self.assertFalse(eligibility.on_policy_rl_eligible)

    def test_full_fields_and_matching_policy_is_on_policy(self) -> None:
        ep = _with_fields(
            _base_episode("ep-onpol"),
            token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3],
            action_mask=[1, 1, 1],
        )
        target = _full_fp()
        behavior = _full_fp()
        eligibility = evaluate_training_eligibility(
            ep, target_policy=target, behavior_policy=behavior
        )
        self.assertTrue(eligibility.on_policy_rl_eligible)
        self.assertIs(
            eligibility.status, TrainingEligibilityStatus.ON_POLICY_RL_ELIGIBLE
        )

    def test_verifier_attested_valid_failure_reward_zero_is_rl_signal(self) -> None:
        base = _with_fields(
            _base_episode("ep-valid-failure"),
            token_ids=[1, 2, 3],
            logprobs=[-0.1, -0.2, -0.3],
            action_mask=[1, 1, 1],
        )
        events = []
        verifier_id = None
        for event in base.events:
            if event.event_type is EventType.VERIFICATION_FINISHED:
                verifier_id = event.event_id
                events.append(
                    replace(event, status=EventStatus.FAILED, attributes={"passed": False})
                )
            else:
                events.append(event)
        self.assertIsNotNone(verifier_id)
        failure = replace(
            base,
            events=tuple(events),
            outcome=EpisodeOutcome(
                task_status=TaskStatus.FAILURE,
                execution_validity=ExecutionValidity.VALID,
                verifier_status=EpisodeVerifierStatus.FAILED,
                score=0.0,
                evidence_event_ids=(verifier_id,),
            ),
        )
        fp = _full_fp()
        eligibility = evaluate_training_eligibility(
            failure, target_policy=fp, behavior_policy=fp
        )
        self.assertFalse(eligibility.sft_eligible)
        self.assertTrue(eligibility.preference_eligible)
        self.assertTrue(eligibility.on_policy_rl_eligible)

    def test_infra_invalid_never_trains_and_never_reward_zero(self) -> None:
        base = _base_episode("ep-infra")
        invalid = replace(
            base,
            outcome=EpisodeOutcome(
                task_status=TaskStatus.UNKNOWN,
                execution_validity=ExecutionValidity.INFRA_INVALID,
                verifier_status=EpisodeVerifierStatus.NOT_RUN,
                score=None,
                evidence_event_ids=base.outcome.evidence_event_ids,
            ),
        )
        eligibility = evaluate_training_eligibility(invalid)
        self.assertFalse(eligibility.sft_eligible)
        self.assertFalse(eligibility.preference_eligible)
        self.assertFalse(eligibility.on_policy_rl_eligible)
        self.assertIn(
            eligibility.status,
            (TrainingEligibilityStatus.REJECTED, TrainingEligibilityStatus.INSUFFICIENT_EVIDENCE),
        )


if __name__ == "__main__":
    unittest.main()
