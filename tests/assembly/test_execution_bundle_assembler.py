from __future__ import annotations

import unittest

from src.assembly import assemble_execution_bundle
from src.assembly.episode_assembler import EpisodeAssembler
from src.contracts.execution_identity import ExecutionIdentity
from src.errors import ContractValidationError
from src.producers.base import (
    ProducerArtifact,
    ProducerCapability,
    ProducerExecutionStatus,
)
from tests.execution_fixtures import make_complete_event_stream, make_episode_context


def _episode():
    events = make_complete_event_stream(
        run_id="run-bundle-live", episode_id="episode-bundle-live"
    )
    return EpisodeAssembler().assemble(
        events,
        contexts={
            "episode-bundle-live": make_episode_context(
                task_id="task-bundle-live", attempt=1
            )
        },
    ).episodes[0]


def _identity() -> ExecutionIdentity:
    return ExecutionIdentity(
        run_id="run-bundle-live",
        task_id="task-bundle-live",
        episode_id="episode-bundle-live",
        attempt_id=1,
        producer_id="polar",
        producer_version="stable@abc",
        group_id="group-1",
        policy_fingerprint="a" * 64,
        sampling_fingerprint="b" * 64,
    )


def _artifact(identity=None, *, verifier=True):
    capabilities = {
        ProducerCapability.TOKEN_IDS,
        ProducerCapability.ACTION_MASK,
        ProducerCapability.BEHAVIOR_LOGPROBS,
        ProducerCapability.POLICY_VERSION,
    }
    if verifier:
        capabilities.add(ProducerCapability.VERIFIER_EVIDENCE)
    return ProducerArtifact(
        identity=identity or _identity(),
        status=ProducerExecutionStatus.COMPLETED,
        capabilities=frozenset(capabilities),
        payload={"trajectory": {"traces": [{"reward": 1.0}]}},
    )


class ExecutionBundleAssemblerTest(unittest.TestCase):
    def test_joins_episode_policy_trace_and_verifier_without_certification_cycle(self) -> None:
        artifact = _artifact()
        bundle = assemble_execution_bundle(
            identity=_identity(),
            episode=_episode(),
            producer_artifacts=(artifact,),
            source_artifact_checksums=("c" * 64,),
            verifier_report_checksum="d" * 64,
        )
        self.assertEqual(bundle.episode_checksum, _episode().checksum)
        self.assertIn(artifact.checksum, bundle.source_artifact_checksums)
        self.assertEqual(bundle.policy_trace_checksums, (artifact.checksum,))
        self.assertNotIn("certification_checksums", bundle.to_dict())

    def test_rejects_cross_session_artifact(self) -> None:
        wrong = ExecutionIdentity(
            run_id="run-bundle-live",
            task_id="task-bundle-live",
            episode_id="other-episode",
            attempt_id=1,
            producer_id="polar",
            producer_version="stable@abc",
        )
        with self.assertRaises(ContractValidationError):
            assemble_execution_bundle(
                identity=_identity(),
                episode=_episode(),
                producer_artifacts=(_artifact(wrong, verifier=False),),
            )

    def test_verifier_capability_requires_report_checksum(self) -> None:
        with self.assertRaises(ContractValidationError):
            assemble_execution_bundle(
                identity=_identity(),
                episode=_episode(),
                producer_artifacts=(_artifact(),),
            )


if __name__ == "__main__":
    unittest.main()
