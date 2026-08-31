from __future__ import annotations

import unittest

from src.assembly import assemble_execution_bundle
from src.assembly.episode_assembler import EpisodeAssembler
from src.certification import (
    ConsumerProfile,
    ConsumerVerdict,
    EligibilityDecision,
    certify_for,
)
from src.contracts.execution_identity import ExecutionIdentity
from src.producers.base import (
    ProducerArtifact,
    ProducerCapability,
    ProducerExecutionStatus,
)
from tests.execution_fixtures import make_complete_event_stream, make_episode_context


class CertificationEngineTest(unittest.TestCase):
    def _episode(self):
        episode_id = "episode-consumer-cert"
        return EpisodeAssembler().assemble(
            make_complete_event_stream(episode_id=episode_id),
            contexts={episode_id: make_episode_context(task_id="task-cert")},
        ).episodes[0]

    def test_harness_and_sft_use_one_public_entrypoint(self) -> None:
        episode = self._episode()
        harness = certify_for(episode, ConsumerProfile.HARNESS_ANALYSIS)
        sft = certify_for(episode, ConsumerProfile.SFT)
        on_policy = certify_for(episode, ConsumerProfile.ON_POLICY_RL)
        self.assertIs(harness.verdict, ConsumerVerdict.ELIGIBLE)
        self.assertIs(sft.verdict, ConsumerVerdict.ELIGIBLE)
        self.assertIs(on_policy.verdict, ConsumerVerdict.INSUFFICIENT_EVIDENCE)
        self.assertTrue(on_policy.reasons)

    def test_on_policy_rl_requires_bundle_bound_policy_evidence(self) -> None:
        episode = self._episode()
        identity = ExecutionIdentity(
            run_id=episode.run_id,
            task_id=episode.task_id,
            episode_id=episode.episode_id,
            attempt_id=episode.attempt,
            producer_id="polar",
            producer_version="stable@abc",
            group_id="group-1",
            policy_fingerprint="a" * 64,
            sampling_fingerprint="b" * 64,
        )
        artifact = ProducerArtifact(
            identity=identity,
            status=ProducerExecutionStatus.COMPLETED,
            capabilities=frozenset(
                {
                    ProducerCapability.TOKEN_IDS,
                    ProducerCapability.ACTION_MASK,
                    ProducerCapability.BEHAVIOR_LOGPROBS,
                    ProducerCapability.POLICY_VERSION,
                    ProducerCapability.VERIFIER_EVIDENCE,
                }
            ),
            payload={"trajectory": {"traces": [{"reward": 1.0}]}},
        )
        bundle = assemble_execution_bundle(
            identity=identity,
            episode=episode,
            producer_artifacts=(artifact,),
            verifier_report_checksum="c" * 64,
        )

        decision = certify_for(
            episode,
            ConsumerProfile.ON_POLICY_RL,
            execution_bundle=bundle,
            policy_artifact=artifact,
            target_policy_fingerprint="a" * 64,
        )
        self.assertIs(decision.verdict, ConsumerVerdict.ELIGIBLE)
        self.assertEqual(decision.execution_bundle_checksum, bundle.checksum)
        self.assertEqual(decision.policy_artifact_checksum, artifact.checksum)
        self.assertEqual(EligibilityDecision.from_dict(decision.to_dict()), decision)

        mismatch = certify_for(
            episode,
            ConsumerProfile.ON_POLICY_RL,
            execution_bundle=bundle,
            policy_artifact=artifact,
            target_policy_fingerprint="d" * 64,
        )
        self.assertIs(mismatch.verdict, ConsumerVerdict.REJECTED)


if __name__ == "__main__":
    unittest.main()
