from __future__ import annotations

import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.capture import HarnessTraceClient, ModelEndpointKind
from src.contracts._json import sha256_json
from src.contracts.agent_episode import CaptureCapability, IntegrityState, TaskStatus
from src.contracts.execution_identity import ExecutionIdentity
from src.contracts.manifests import EvaluatorManifest, HarnessManifest, ModelManifest
from src.contracts.trace_event import EventComponent, EventStatus, EventType
from src.launchers import DockerLocalLauncher, SandboxLimits
from src.orchestration import (
    LocalExecutionOrchestrator,
    LocalExecutionSpec,
    prepare_local_execution,
)
from src.producers import ProducerCapability, ProducerExecutionStatus


def _spec(workspace: str) -> LocalExecutionSpec:
    digest = sha256_json({})
    return LocalExecutionSpec(
        identity=ExecutionIdentity(
            run_id="run-one-shot",
            task_id="task-one-shot",
            episode_id="episode-one-shot",
            attempt_id=1,
            producer_id="local-docker",
            producer_version="local-docker-launcher/v1",
            policy_fingerprint="a" * 64,
            sampling_fingerprint="b" * 64,
        ),
        harness_manifest=HarnessManifest(
            name="fixture-harness",
            version="v1",
            revision="harness-r1",
            config_digest=digest,
            hook_version="harness-event-ingress/v1",
        ),
        model_manifest=ModelManifest(
            provider="self-hosted",
            model_id="example-14b",
            revision="checkpoint-v0",
            sampling_config={"temperature": 0.7},
            tokenizer_revision="tokenizer-r1",
        ),
        evaluator_manifest=EvaluatorManifest(
            name="fixture-verifier",
            revision="verifier-r1",
            config_digest=digest,
        ),
        experiment_manifest_ref="experiment-local-v1",
        image="example/harness",
        image_digest="c" * 64,
        workspace=workspace,
        harness_argv=("harness",),
        verifier_argv=("verify",),
        upstream_chat_completions_url="http://127.0.0.1:9/v1/chat/completions",
        endpoint_kind=ModelEndpointKind.CONTROLLED,
        task_snapshot="task-snapshot-r1",
        capture_capabilities=frozenset(
            {CaptureCapability.TOOL_IO, CaptureCapability.HARNESS_DECISIONS}
        ),
        limits=SandboxLimits(cpus=1, memory_mb=512, pids=64, tmpfs_mb=64),
        trace_id="trace-one-shot",
    )


class LocalExecutionTest(unittest.TestCase):
    def test_spec_round_trip_and_prepare_freezes_secret_free_plan(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            spec = _spec(workspace)
            self.assertEqual(LocalExecutionSpec.from_dict(spec.to_dict()), spec)
            prepared = prepare_local_execution(
                spec,
                proxy_base_url="http://host.docker.internal:8088",
                access_token="must-not-persist",
                created_at="2026-08-22T00:00:00Z",
            )
        self.assertNotIn("must-not-persist", str(prepared.launch_plan.to_dict()))
        self.assertEqual(
            prepared.manifest.launcher_plan_checksum,
            prepared.launch_plan.checksum,
        )
        self.assertIn(
            CaptureCapability.VERIFIER_EVIDENCE,
            prepared.manifest.capture_capabilities,
        )

    def test_linux_gateway_is_frozen_into_harness_launch_plan(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            spec = replace(
                _spec(workspace),
                docker_host_gateway=True,
                proxy_bind_host="172.17.0.1",
            )
            prepared = prepare_local_execution(
                spec,
                proxy_base_url="http://host.docker.internal:8088",
                access_token="must-not-persist",
            )
        self.assertIn("--add-host", prepared.launch_plan.argv)
        self.assertEqual(spec.proxy_bind_host, "172.17.0.1")

    def test_one_shot_owns_harness_verifier_artifacts_and_finalization(self) -> None:
        timestamps = iter(f"2026-08-22T00:00:{second:02d}Z" for second in range(30))

        def runner(argv, **kwargs):
            command = argv[-1]
            if command == "harness":
                client = HarnessTraceClient(
                    endpoint=kwargs["env"]["AGENT_TRACE_URL"].replace(
                        "host.docker.internal", "127.0.0.1"
                    ),
                    access_token=kwargs["env"]["AGENT_TRACE_API_KEY"],
                )
                client.emit(
                    event_type=EventType.TOOL_CALL,
                    component=EventComponent.TOOL,
                    status=EventStatus.STARTED,
                    span_id="span-tool",
                    attributes={"tool_name": "shell", "arguments": {"cmd": "false"}},
                )
                client.emit(
                    event_type=EventType.TOOL_RESULT,
                    component=EventComponent.TOOL,
                    status=EventStatus.FAILED,
                    span_id="span-tool",
                    attributes={"tool_name": "shell", "exit_code": 1},
                )
                return subprocess.CompletedProcess(argv, 0, stdout="harness output", stderr="")
            return subprocess.CompletedProcess(argv, 1, stdout="tests failed", stderr="assertion")

        def launcher_factory(**kwargs):
            return DockerLocalLauncher(runner=runner, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            result = LocalExecutionOrchestrator(
                launcher_factory=launcher_factory,
                access_token_factory=lambda: "execution-secret",
                clock=lambda: next(timestamps),
            ).run(_spec(str(workspace)), output_dir=root / "run")

            episode = result.finalization.episode
            self.assertIs(episode.integrity.state, IntegrityState.COMPLETE)
            self.assertIs(episode.outcome.task_status, TaskStatus.FAILURE)
            self.assertEqual(episode.outcome.verifier_status.value, "FAILED")
            self.assertIs(
                result.producer_artifact.status,
                ProducerExecutionStatus.COMPLETED,
            )
            self.assertIn(
                ProducerCapability.VERIFIER_EVIDENCE,
                result.producer_artifact.capabilities,
            )
            self.assertIsNotNone(result.finalization.execution_bundle.verifier_report_checksum)
            self.assertTrue((root / "run" / "finalized" / "episode.json").is_file())
            self.assertTrue((root / "run" / "objects").is_dir())
            self.assertNotIn(
                "harness output",
                (root / "run" / "producer-artifact.json").read_text(),
            )


if __name__ == "__main__":
    unittest.main()
