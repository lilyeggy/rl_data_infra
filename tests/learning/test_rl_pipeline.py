from __future__ import annotations

import unittest

from src.assembly import assemble_execution_bundle
from src.assembly.episode_assembler import EpisodeAssembler
from src.certification import ConsumerProfile, ConsumerVerdict, certify_for
from src.contracts.dataset import DatasetPurpose, DatasetRole, DatasetSplit
from src.contracts.execution_identity import ExecutionIdentity
from src.learning import CertifiedArtifact, compile_dataset
from src.producers.base import (
    ProducerArtifact,
    ProducerCapability,
    ProducerExecutionStatus,
)
from tests.execution_fixtures import make_complete_event_stream, make_episode_context


class OfflineRlPipelineTest(unittest.TestCase):
    def test_bundle_certification_and_dataset_manifest_are_one_way_linked(self) -> None:
        episode_id = "episode-offline-rl"
        episode = EpisodeAssembler().assemble(
            make_complete_event_stream(
                run_id="run-offline-rl", episode_id=episode_id
            ),
            contexts={episode_id: make_episode_context(task_id="task-offline-rl")},
        ).episodes[0]
        identity = ExecutionIdentity(
            run_id=episode.run_id,
            task_id=episode.task_id,
            episode_id=episode.episode_id,
            attempt_id=episode.attempt,
            producer_id="polar",
            producer_version="stable@abc",
            group_id="group-offline-rl",
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
            payload={
                "trajectory": {
                    "traces": [
                        {
                            "response_ids": [1, 2],
                            "loss_mask": [1, 1],
                            "response_logprobs": [-0.1, -0.2],
                            "reward": 1.0,
                        }
                    ]
                }
            },
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

        manifest = compile_dataset(
            dataset_id="dataset-offline-rl",
            revision="r1",
            purpose=DatasetPurpose.ON_POLICY_RL,
            selection_policy_version="rl-selection/v1",
            artifacts=(
                CertifiedArtifact(
                    identity=identity,
                    decision=decision,
                    artifact_checksum=artifact.checksum,
                    split=DatasetSplit.TRAIN,
                    role=DatasetRole.TRAJECTORY,
                ),
            ),
        )
        member = manifest.members[0]
        self.assertEqual(member.execution_bundle_checksum, bundle.checksum)
        self.assertEqual(member.policy_artifact_checksum, artifact.checksum)
        self.assertNotIn("certification_checksums", bundle.to_dict())


if __name__ == "__main__":
    unittest.main()
