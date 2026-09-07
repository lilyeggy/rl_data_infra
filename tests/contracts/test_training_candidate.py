from __future__ import annotations

import unittest

from src.assembly.episode_assembler import EpisodeAssembler
from src.contracts.agent_episode import CaptureCapability
from src.contracts.training_candidate import build_training_candidate_view
from src.training.policy_fingerprint import PolicyFingerprint
from tests.execution_fixtures import make_complete_event_stream, make_episode_context


class TrainingCandidateViewTest(unittest.TestCase):
    def test_gpt_teacher_success_is_sft_candidate_but_not_on_policy_rl(self) -> None:
        episode = EpisodeAssembler().assemble(
            make_complete_event_stream(),
            contexts={"episode-001": make_episode_context()},
        ).episodes[0]
        view = build_training_candidate_view(episode, target_policy_model="local-qwen-4b")
        self.assertTrue(view.sft_candidate)
        self.assertFalse(view.on_policy_rl_candidate)
        self.assertEqual(view.policy_relation, "OFF_POLICY")
        self.assertIn("MODEL_LOGPROBS", view.missing_rl_capabilities)
        self.assertIn("MODEL_TOKEN_IDS", view.missing_rl_capabilities)

    def test_capability_declaration_without_real_fields_is_not_rl_candidate(self) -> None:
        # Adding MODEL_TOKEN_IDS/MODEL_LOGPROBS to the *declared capabilities*
        # set is NOT proof of on-policy eligibility.  The current data plane
        # carries no actual token/logprob arrays and no complete policy
        # fingerprint, so on-policy RL must fail closed.
        context = make_episode_context(
            capabilities=make_episode_context().capabilities
            | {CaptureCapability.MODEL_TOKEN_IDS, CaptureCapability.MODEL_LOGPROBS}
        )
        episode = EpisodeAssembler().assemble(
            make_complete_event_stream(), contexts={"episode-001": context}
        ).episodes[0]
        view = build_training_candidate_view(
            episode, target_policy_model=episode.model_manifest.model_id
        )
        self.assertTrue(view.sft_candidate)
        self.assertFalse(view.on_policy_rl_candidate)
        self.assertEqual(view.policy_relation, "OFF_POLICY")
        self.assertIn("MODEL_LOGPROBS", view.missing_rl_capabilities)
        self.assertIn("MODEL_TOKEN_IDS", view.missing_rl_capabilities)

    def test_full_fingerprint_and_real_fields_yield_on_policy_candidate(self) -> None:
        # A closed-loop fixture that carries real token/logprob/mask evidence
        # AND a complete matching policy fingerprint is the only way to be an
        # on-policy RL candidate.
        from dataclasses import replace

        from src.contracts.trace_event import EventType
        from src.training.eligibility import evaluate_training_eligibility

        episode = EpisodeAssembler().assemble(
            make_complete_event_stream(),
            contexts={"episode-001": make_episode_context()},
        ).episodes[0]
        # inject real token/logprob/mask fields into the MODEL_RESPONSE event
        events = []
        for event in episode.events:
            if event.event_type is EventType.MODEL_RESPONSE:
                attrs = dict(event.attributes)
                attrs["token_ids"] = [1, 2, 3, 4]
                attrs["logprobs"] = [-0.1, -0.2, -0.3, -0.4]
                attrs["action_mask"] = [1, 1, 1, 1]
                events.append(replace(event, attributes=attrs))
            else:
                events.append(event)
        episode = replace(episode, events=tuple(events))

        target = PolicyFingerprint(
            provider="fixture-provider",
            model_id="fixture-model",
            base_model_revision="fixture-model-r1",
            adapter_revision="adapter-r1",
            tokenizer_revision="fixture-tokenizer-r1",
            chat_template_checksum="chat-c1",
            tool_schema_checksum="tools-c1",
            sampling_config={"temperature": 0.0},
            temperature=0.0,
            top_p=1.0,
            policy_generation="gen-1",
        )
        behavior = PolicyFingerprint(
            provider="fixture-provider",
            model_id="fixture-model",
            base_model_revision="fixture-model-r1",
            adapter_revision="adapter-r1",
            tokenizer_revision="fixture-tokenizer-r1",
            chat_template_checksum="chat-c1",
            tool_schema_checksum="tools-c1",
            sampling_config={"temperature": 0.0},
            temperature=0.0,
            top_p=1.0,
            policy_generation="gen-1",
        )

        eligibility = evaluate_training_eligibility(
            episode, target_policy=target, behavior_policy=behavior
        )
        view = build_training_candidate_view(
            episode, target_policy=target, eligibility=eligibility
        )
        self.assertTrue(view.on_policy_rl_candidate)
        self.assertEqual(view.policy_relation, "ON_POLICY")
        self.assertEqual(view.training_role, "TARGET_POLICY")


if __name__ == "__main__":
    unittest.main()
