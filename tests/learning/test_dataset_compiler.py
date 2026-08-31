from __future__ import annotations

import unittest

from src.assembly.episode_assembler import EpisodeAssembler
from src.certification import ConsumerProfile, certify_for
from src.contracts.dataset import DatasetManifest, DatasetPurpose, DatasetRole, DatasetSplit
from src.contracts.execution_identity import ExecutionIdentity
from src.errors import ContractValidationError
from src.learning.dataset_compiler import CertifiedArtifact, compile_dataset
from tests.execution_fixtures import make_complete_event_stream, make_episode_context


def _episode(episode_id: str, task_id: str):
    return EpisodeAssembler().assemble(
        make_complete_event_stream(episode_id=episode_id),
        contexts={episode_id: make_episode_context(task_id=task_id)},
    ).episodes[0]


def _identity(
    episode_id: str,
    task_id: str,
    attempt: int = 1,
    *,
    group_id: str | None = None,
):
    return ExecutionIdentity(
        run_id="run-dataset",
        task_id=task_id,
        episode_id=episode_id,
        attempt_id=attempt,
        producer_id="pi-direct",
        producer_version="pi-direct/v1",
        group_id=group_id,
    )


class DatasetCompilerTest(unittest.TestCase):
    def test_compiles_only_matching_eligible_profile(self) -> None:
        episode = _episode("ep-dataset", "task-dataset")
        decision = certify_for(episode, ConsumerProfile.SFT)
        manifest = compile_dataset(
            dataset_id="dataset-sft",
            revision="r1",
            purpose=DatasetPurpose.SFT,
            selection_policy_version="sft-selection/v1",
            artifacts=(
                CertifiedArtifact(
                    identity=_identity(episode.episode_id, episode.task_id),
                    decision=decision,
                    artifact_checksum="a" * 64,
                    split=DatasetSplit.TRAIN,
                    role=DatasetRole.EXAMPLE,
                ),
            ),
        )
        self.assertEqual(len(manifest.members), 1)
        self.assertEqual(manifest.members[0].episode_checksum, episode.checksum)
        self.assertEqual(DatasetManifest.from_dict(manifest.to_dict()), manifest)

    def test_rejects_profile_mismatch(self) -> None:
        episode = _episode("ep-profile", "task-profile")
        decision = certify_for(episode, ConsumerProfile.HARNESS_ANALYSIS)
        with self.assertRaises(ContractValidationError):
            compile_dataset(
                dataset_id="dataset-sft",
                revision="r1",
                purpose=DatasetPurpose.SFT,
                selection_policy_version="sft-selection/v1",
                artifacts=(
                    CertifiedArtifact(
                        identity=_identity(episode.episode_id, episode.task_id),
                        decision=decision,
                        artifact_checksum="a" * 64,
                        split=DatasetSplit.TRAIN,
                        role=DatasetRole.EXAMPLE,
                    ),
                ),
            )

    def test_rejects_task_leakage_across_splits(self) -> None:
        first = _episode("ep-leak-1", "task-leak")
        second = _episode("ep-leak-2", "task-leak")
        artifacts = (
            CertifiedArtifact(
                identity=_identity(first.episode_id, first.task_id, 1),
                decision=certify_for(first, ConsumerProfile.SFT),
                artifact_checksum="a" * 64,
                split=DatasetSplit.TRAIN,
                role=DatasetRole.EXAMPLE,
            ),
            CertifiedArtifact(
                identity=_identity(second.episode_id, second.task_id, 2),
                decision=certify_for(second, ConsumerProfile.SFT),
                artifact_checksum="b" * 64,
                split=DatasetSplit.TEST,
                role=DatasetRole.EXAMPLE,
            ),
        )
        with self.assertRaises(ContractValidationError):
            compile_dataset(
                dataset_id="dataset-leak",
                revision="r1",
                purpose=DatasetPurpose.SFT,
                selection_policy_version="sft-selection/v1",
                artifacts=artifacts,
            )

    def test_rejects_unpaired_preference_member(self) -> None:
        episode = _episode("ep-chosen-only", "task-pref")
        decision = certify_for(episode, ConsumerProfile.PREFERENCE)
        with self.assertRaises(ContractValidationError):
            compile_dataset(
                dataset_id="dataset-pref",
                revision="r1",
                purpose=DatasetPurpose.PREFERENCE,
                selection_policy_version="pref-selection/v1",
                artifacts=(
                    CertifiedArtifact(
                        identity=_identity(
                            episode.episode_id,
                            episode.task_id,
                            group_id="pair-1",
                        ),
                        decision=decision,
                        artifact_checksum="a" * 64,
                        split=DatasetSplit.TRAIN,
                        role=DatasetRole.CHOSEN,
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
