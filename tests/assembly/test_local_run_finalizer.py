from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.assembly import finalize_local_run
from src.capture import (
    ArtifactStore,
    EventWriter,
    ModelBackendResponse,
    ModelEndpointKind,
    ModelEvidenceJsonlWriter,
    ModelProxyRequest,
    capture_model_call,
)
from src.contracts._json import sha256_json
from src.contracts.agent_episode import CaptureCapability, IntegrityState
from src.contracts.execution_identity import ExecutionIdentity
from src.contracts.manifests import (
    EnvironmentManifest,
    EvaluatorManifest,
    HarnessManifest,
    ModelManifest,
)
from src.contracts.run_manifest import ExecutionRunManifest
from src.errors import ContractValidationError
from src.producers import ProducerArtifact, ProducerCapability, ProducerExecutionStatus
from tests.execution_fixtures import make_complete_event_stream


def _identity() -> ExecutionIdentity:
    return ExecutionIdentity(
        run_id="run-finalize",
        task_id="task-finalize",
        episode_id="episode-finalize",
        attempt_id=1,
        producer_id="local-docker",
        producer_version="local-docker-launcher/v1",
        policy_fingerprint="a" * 64,
        sampling_fingerprint="b" * 64,
    )


def _manifest() -> ExecutionRunManifest:
    digest = sha256_json({})
    return ExecutionRunManifest(
        identity=_identity(),
        harness_manifest=HarnessManifest(
            name="fixture-harness",
            version="v1",
            revision="harness-r1",
            config_digest=digest,
        ),
        model_manifest=ModelManifest(
            provider="self-hosted",
            model_id="example-14b",
            revision="checkpoint-v0",
            sampling_config={"temperature": 0.7},
            tokenizer_revision="tokenizer-r1",
        ),
        environment_manifest=EnvironmentManifest(
            runtime_type="docker",
            revision="docker-r1",
            image="example/harness@sha256:" + "c" * 64,
            resource_limits={"memory_mb": 4096},
            network_policy="bridge-unrestricted",
            task_snapshot="task-snapshot-r1",
        ),
        evaluator_manifest=EvaluatorManifest(
            name="fixture-verifier",
            revision="verifier-r1",
            config_digest=digest,
        ),
        experiment_manifest_ref="experiment-local-r1",
        capture_capabilities=frozenset(
            {
                CaptureCapability.MODEL_IO,
                CaptureCapability.MODEL_TOKEN_USAGE,
                CaptureCapability.TOOL_IO,
                CaptureCapability.VERIFIER_EVIDENCE,
            }
        ),
        launcher_plan_checksum="d" * 64,
        created_at="2026-08-22T00:00:00Z",
    )


class LocalRunFinalizerTest(unittest.TestCase):
    def test_strictly_joins_manifest_events_model_evidence_and_launcher_artifact(self) -> None:
        manifest = _manifest()
        self.assertEqual(ExecutionRunManifest.from_dict(manifest.to_dict()), manifest)
        producer = ProducerArtifact(
            identity=manifest.identity,
            status=ProducerExecutionStatus.COMPLETED,
            capabilities=frozenset({ProducerCapability.RAW_HARNESS_TRACE}),
            payload={"launch_plan_checksum": manifest.launcher_plan_checksum},
        )
        evidence = capture_model_call(
            ModelProxyRequest(
                identity=manifest.identity,
                request_id="request-finalize",
                endpoint_kind=ModelEndpointKind.CONTROLLED,
                model_id="example-14b",
                messages=({"role": "user", "content": "fix"},),
            ),
            ModelBackendResponse(
                response={"text": "done"},
                latency_ms=10,
                status_code=200,
                backend_model_revision="checkpoint-v0",
                response_token_ids=(20,),
                response_logprobs=(-0.1,),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            event_path = root / "events.jsonl"
            evidence_path = root / "model-evidence.jsonl"
            EventWriter(event_path).append_many(
                make_complete_event_stream(
                    run_id=manifest.identity.run_id,
                    episode_id=manifest.identity.episode_id,
                )
            )
            ModelEvidenceJsonlWriter(evidence_path).append(evidence)
            result = finalize_local_run(
                manifest=manifest,
                producer_artifact=producer,
                events_path=event_path,
                model_evidence_path=evidence_path,
            )
        self.assertIs(result.episode.integrity.state, IntegrityState.COMPLETE)
        self.assertEqual(result.execution_bundle.identity, manifest.identity)
        self.assertIn(evidence.checksum, result.execution_bundle.source_artifact_checksums)

    def test_rejects_corrupted_local_artifact_object(self) -> None:
        manifest = _manifest()
        producer = ProducerArtifact(
            identity=manifest.identity,
            status=ProducerExecutionStatus.COMPLETED,
            capabilities=frozenset({ProducerCapability.RAW_HARNESS_TRACE}),
            payload={"launch_plan_checksum": manifest.launcher_plan_checksum},
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events = root / "events.jsonl"
            EventWriter(events).append_many(
                make_complete_event_stream(
                    run_id=manifest.identity.run_id,
                    episode_id=manifest.identity.episode_id,
                )
            )
            artifact = ArtifactStore(root / "objects").put(
                b"original",
                kind="stdout",
                media_type="text/plain",
                created_at="2026-08-22T00:00:00Z",
            )
            manifest_path = root / "artifact-refs.json"
            manifest_path.write_text(json.dumps([artifact.to_dict()]))
            Path(artifact.uri).write_bytes(b"tampered")
            with self.assertRaisesRegex(ContractValidationError, "failed checksum"):
                finalize_local_run(
                    manifest=manifest,
                    producer_artifact=producer,
                    events_path=events,
                    model_evidence_path=root / "model-evidence.jsonl",
                    artifacts_path=manifest_path,
                )


if __name__ == "__main__":
    unittest.main()
