from __future__ import annotations

import unittest
from dataclasses import replace

from src.assembly import assemble_execution_bundle
from src.assembly.episode_assembler import EpisodeAssembler
from src.certification import ConsumerProfile, ConsumerVerdict, certify_for
from src.contracts._json import sha256_json
from src.contracts.dataset import DatasetPurpose, DatasetRole, DatasetSplit
from src.contracts.trace_event import EventStatus, EventType
from src.integrations.polar import (
    IDENTITY_METADATA_KEY,
    PolarBatchContext,
    adapt_polar_task_result,
)
from src.integrations.slime import admit_on_policy_manifest
from src.learning import CertifiedArtifact, compile_dataset
from tests.execution_fixtures import make_complete_event_stream, make_episode_context


def _trace(reward: float):
    return {
        "prompt_ids": [10, 11],
        "response_ids": [20, 21],
        "loss_mask": [1, 1],
        "response_logprobs": [-0.1, -0.2],
        "reward": reward,
        "prompt_messages": [{"role": "user", "content": "fix it"}],
        "response_messages": [{"role": "assistant", "content": "done"}],
        "metadata": {},
    }


def _session(context, session_id: str, reward: float):
    return {
        "session_id": session_id,
        "task_id": context.polar_task_id,
        "status": "COMPLETED",
        "trajectory": {
            "status": "COMPLETED",
            "metadata": {IDENTITY_METADATA_KEY: context.identity_metadata()},
            "traces": [_trace(reward)],
            "error": None,
        },
        "timing": {"run_ms": 100.0},
        "node_id": "node-1",
        "error": None,
        "metadata": {},
    }


def _episode_for(artifact, *, success: bool):
    events = list(
        make_complete_event_stream(
            run_id=artifact.identity.run_id,
            episode_id=artifact.identity.episode_id,
            trace_id=f"trace-{artifact.identity.attempt_id}",
        )
    )
    if not success:
        for index, event in enumerate(events):
            if event.event_type is EventType.VERIFICATION_FINISHED:
                events[index] = replace(
                    event,
                    status=EventStatus.FAILED,
                    attributes={"passed": False, "latency_ms": 50},
                )
            elif event.event_type is EventType.EPISODE_FINISHED:
                events[index] = replace(
                    event,
                    attributes={
                        "task_status": "FAILURE",
                        "execution_validity": "VALID",
                        "verifier_status": "FAILED",
                        "score": 0.0,
                        "termination_reason": "VERIFIER_FAILED",
                        "evidence_event_ids": ["evt-verification-finished"],
                    },
                )
    return EpisodeAssembler().assemble(
        events,
        contexts={
            artifact.identity.episode_id: make_episode_context(
                task_id=artifact.identity.task_id,
                attempt=artifact.identity.attempt_id,
            )
        },
    ).episodes[0]


class OfflinePolarPipelineTest(unittest.TestCase):
    def test_task_result_to_slime_admission_preserves_valid_failure_signal(self) -> None:
        context = PolarBatchContext(
            run_id="run-polar-offline",
            logical_task_id="swebench/task-1",
            polar_task_id="polar-task-offline",
            expected_samples=2,
            producer_version="stable@abc",
            group_id="group-polar-offline",
            policy_fingerprint="a" * 64,
            sampling_fingerprint="b" * 64,
            evaluator_fingerprint=sha256_json({"strategy": "swebench_harness"}),
        )
        raw = {
            "task_id": context.polar_task_id,
            "status": "completed",
            "total_sessions": 2,
            "completed_sessions": 2,
            "results": [
                _session(context, "session-success", 1.0),
                _session(context, "session-valid-failure", 0.0),
            ],
            "result_paths": [],
        }
        artifacts = adapt_polar_task_result(raw, context=context)

        pipeline = []
        for artifact in artifacts:
            success = artifact.payload["polar_session_id"] == "session-success"
            episode = _episode_for(artifact, success=success)
            bundle = assemble_execution_bundle(
                identity=artifact.identity,
                episode=episode,
                producer_artifacts=(artifact,),
                verifier_report_checksum=("c" if success else "d") * 64,
            )
            decision = certify_for(
                episode,
                ConsumerProfile.ON_POLICY_RL,
                execution_bundle=bundle,
                policy_artifact=artifact,
                target_policy_fingerprint="a" * 64,
            )
            self.assertIs(decision.verdict, ConsumerVerdict.ELIGIBLE)
            pipeline.append((artifact, bundle, decision))

        manifest = compile_dataset(
            dataset_id="dataset-polar-offline",
            revision="r1",
            purpose=DatasetPurpose.ON_POLICY_RL,
            selection_policy_version="selection/v1",
            artifacts=tuple(
                CertifiedArtifact(
                    identity=artifact.identity,
                    decision=decision,
                    artifact_checksum=artifact.checksum,
                    split=DatasetSplit.TRAIN,
                    role=DatasetRole.TRAJECTORY,
                )
                for artifact, _, decision in pipeline
            ),
        )
        admission = admit_on_policy_manifest(
            manifest,
            decisions_by_checksum={
                decision.checksum: decision for _, _, decision in pipeline
            },
            bundles_by_checksum={bundle.checksum: bundle for _, bundle, _ in pipeline},
            artifacts_by_checksum={
                artifact.checksum: artifact for artifact, _, _ in pipeline
            },
        )
        self.assertEqual(admission.group_sizes, {"group-polar-offline": 2})
        self.assertEqual(sorted(trace.reward for trace in admission.traces), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
