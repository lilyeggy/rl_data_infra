from __future__ import annotations

import unittest

from src.assembly.episode_assembler import EpisodeAssembler
from src.contracts.agent_episode import CaptureCapability
from src.contracts.training_candidate import build_training_candidate_view
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

    def test_exact_target_policy_capabilities_can_be_rl_candidate(self) -> None:
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
        self.assertTrue(view.on_policy_rl_candidate)
        self.assertEqual(view.policy_relation, "ON_POLICY")


if __name__ == "__main__":
    unittest.main()
