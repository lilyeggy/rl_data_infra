from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from src.assembly import assemble_execution_bundle
from src.assembly.episode_assembler import EpisodeAssembler
from src.certification import ConsumerProfile, certify_for
from src.cli import _admit_slime
from src.contracts.dataset import DatasetPurpose, DatasetRole, DatasetSplit
from src.contracts.execution_identity import ExecutionIdentity
from src.errors import ContractValidationError
from src.integrations.slime import admit_on_policy_manifest
from src.learning import CertifiedArtifact, compile_dataset
from src.producers.base import (
    ProducerArtifact,
    ProducerCapability,
    ProducerExecutionStatus,
)
from tests.execution_fixtures import make_complete_event_stream, make_episode_context


def _entry(index: int):
    episode_id = f"episode-slime-{index}"
    episode = EpisodeAssembler().assemble(
        make_complete_event_stream(run_id="run-slime", episode_id=episode_id),
        contexts={
            episode_id: make_episode_context(task_id="task-slime", attempt=index)
        },
    ).episodes[0]
    identity = ExecutionIdentity(
        run_id=episode.run_id,
        task_id=episode.task_id,
        episode_id=episode.episode_id,
        attempt_id=episode.attempt,
        producer_id="polar",
        producer_version="stable@abc",
        group_id="group-slime",
        policy_fingerprint="a" * 64,
        sampling_fingerprint="b" * 64,
    )
    trace = {
        "prompt_ids": [10, 11],
        "response_ids": [20, 21],
        "loss_mask": [1, 1],
        "response_logprobs": [-0.1, -0.2],
        "reward": float(index % 2),
    }
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
        payload={"trajectory": {"traces": [trace, trace]}},
    )
    bundle = assemble_execution_bundle(
        identity=identity,
        episode=episode,
        producer_artifacts=(artifact,),
        verifier_report_checksum=(str(index) * 64),
    )
    decision = certify_for(
        episode,
        ConsumerProfile.ON_POLICY_RL,
        execution_bundle=bundle,
        policy_artifact=artifact,
        target_policy_fingerprint="a" * 64,
    )
    return identity, artifact, bundle, decision


class SlimeAdmissionTest(unittest.TestCase):
    def _pipeline(self):
        entries = (_entry(1), _entry(2))
        manifest = compile_dataset(
            dataset_id="dataset-slime",
            revision="r1",
            purpose=DatasetPurpose.ON_POLICY_RL,
            selection_policy_version="selection/v1",
            artifacts=tuple(
                CertifiedArtifact(
                    identity=identity,
                    decision=decision,
                    artifact_checksum=artifact.checksum,
                    split=DatasetSplit.TRAIN,
                    role=DatasetRole.TRAJECTORY,
                )
                for identity, artifact, _, decision in entries
            ),
        )
        decisions = {decision.checksum: decision for _, _, _, decision in entries}
        bundles = {bundle.checksum: bundle for _, _, bundle, _ in entries}
        artifacts = {artifact.checksum: artifact for _, artifact, _, _ in entries}
        return manifest, decisions, bundles, artifacts

    def test_admits_complete_group_and_counts_trajectories_not_traces(self) -> None:
        manifest, decisions, bundles, artifacts = self._pipeline()
        batch = admit_on_policy_manifest(
            manifest,
            decisions_by_checksum=decisions,
            bundles_by_checksum=bundles,
            artifacts_by_checksum=artifacts,
        )
        self.assertEqual(batch.trajectory_count, 2)
        self.assertEqual(batch.group_sizes, {"group-slime": 2})
        self.assertEqual(len(batch.traces), 4)
        self.assertEqual(batch.policy_fingerprint, "a" * 64)

    def test_rejects_missing_artifact_and_undersized_group(self) -> None:
        manifest, decisions, bundles, artifacts = self._pipeline()
        with self.assertRaises(ContractValidationError):
            admit_on_policy_manifest(
                manifest,
                decisions_by_checksum=decisions,
                bundles_by_checksum=bundles,
                artifacts_by_checksum={},
            )
        with self.assertRaises(ContractValidationError):
            admit_on_policy_manifest(
                manifest,
                decisions_by_checksum=decisions,
                bundles_by_checksum=bundles,
                artifacts_by_checksum=artifacts,
                minimum_group_size=3,
            )

    def test_cli_rehydrates_persisted_chain_and_writes_admission_atomically(self) -> None:
        manifest, decisions, bundles, artifacts = self._pipeline()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "dataset-manifest.json"
            decisions_path = root / "eligibility-decisions.jsonl"
            bundles_path = root / "execution-bundles.jsonl"
            artifacts_path = root / "producer-artifacts.jsonl"
            output_path = root / "slime-admission.json"
            manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
            decisions_path.write_text(
                "".join(json.dumps(item.to_dict()) + "\n" for item in decisions.values()),
                encoding="utf-8",
            )
            bundles_path.write_text(
                "".join(json.dumps(item.to_dict()) + "\n" for item in bundles.values()),
                encoding="utf-8",
            )
            artifacts_path.write_text(
                "".join(json.dumps(item.to_dict()) + "\n" for item in artifacts.values()),
                encoding="utf-8",
            )
            with redirect_stdout(StringIO()):
                result = _admit_slime(
                    Namespace(
                        manifest=str(manifest_path),
                        decisions=str(decisions_path),
                        bundles=str(bundles_path),
                        artifacts=str(artifacts_path),
                        output=str(output_path),
                        minimum_group_size=2,
                    )
                )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result, 0)
            self.assertEqual(payload["trajectory_count"], 2)
            self.assertEqual(len(payload["traces"]), 4)
            self.assertEqual(len(payload["checksum"]), 64)


if __name__ == "__main__":
    unittest.main()
