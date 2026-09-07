from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from src.capture import ArtifactStore
from src.contracts.execution_identity import ExecutionIdentity
from src.errors import ContractValidationError
from src.launchers import (
    DockerLocalLauncher,
    LocalHarnessRequest,
    SandboxNetworkPolicy,
    SandboxProfile,
    SandboxRuntime,
    build_docker_launch_plan,
)
from src.producers import ProducerExecutionStatus


def _identity() -> ExecutionIdentity:
    return ExecutionIdentity(
        run_id="run-local",
        task_id="task-local",
        episode_id="episode-local",
        attempt_id=1,
        producer_id="local-docker",
        producer_version="local-docker-launcher/v1",
        policy_fingerprint="a" * 64,
        sampling_fingerprint="b" * 64,
    )


class DockerLocalLauncherTest(unittest.TestCase):
    def test_plan_injects_identity_and_default_isolation_without_shell(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            request = LocalHarnessRequest(
                identity=_identity(),
                image="example/harness",
                image_digest="c" * 64,
                harness_argv=("python", "-m", "harness", "task.json"),
                host_workspace=workspace,
                timeout_seconds=60,
            )
            plan = build_docker_launch_plan(request)
            self.assertIn("--read-only", plan.argv)
            self.assertIn("linux/amd64", plan.argv)
            self.assertIn("no-new-privileges", plan.argv)
            self.assertIn("--user", plan.argv)
            self.assertIn("ALL", plan.argv)
            self.assertIn("none", plan.argv)
            self.assertEqual(plan.environment["AGENT_RUN_ID"], "run-local")
            self.assertEqual(plan.argv[-4:], request.harness_argv)

    def test_model_proxy_requires_explicit_network_access(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            with self.assertRaises(ContractValidationError):
                LocalHarnessRequest(
                    identity=_identity(),
                    image="example/harness",
                    image_digest="c" * 64,
                    harness_argv=("harness",),
                    host_workspace=workspace,
                    timeout_seconds=60,
                    model_proxy_url="http://host.docker.internal:8088",
                    model_proxy_api_key="proxy-secret",
                )

    def test_secure_profile_requires_runsc(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            with self.assertRaises(ContractValidationError):
                LocalHarnessRequest(
                    identity=_identity(),
                    image="example/harness",
                    image_digest="c" * 64,
                    harness_argv=("harness",),
                    host_workspace=workspace,
                    timeout_seconds=60,
                    sandbox_profile=SandboxProfile.SECURE,
                    sandbox_runtime=SandboxRuntime.RUNC,
                )

    def test_secure_profile_fails_closed_when_runsc_is_not_registered(self) -> None:
        launch_calls = []

        def runner(argv, **_):
            launch_calls.append(argv)
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        def runtime_probe(argv, **_):
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout='{"runc":{"path":"runc"}}',
                stderr="",
            )

        with tempfile.TemporaryDirectory() as workspace:
            artifact = DockerLocalLauncher(
                runner=runner,
                runtime_probe_runner=runtime_probe,
            ).run(
                LocalHarnessRequest(
                    identity=_identity(),
                    image="example/harness",
                    image_digest="c" * 64,
                    harness_argv=("harness",),
                    host_workspace=workspace,
                    timeout_seconds=60,
                    sandbox_profile=SandboxProfile.SECURE,
                    sandbox_runtime=SandboxRuntime.RUNSC,
                )
            )
        self.assertIs(artifact.status, ProducerExecutionStatus.INFRA_INVALID)
        self.assertEqual(launch_calls, [])
        self.assertFalse(artifact.payload["runtime_evidence"]["registration_verified"])

    def test_secure_profile_verifies_and_selects_runsc(self) -> None:
        def runner(argv, **_):
            return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

        def runtime_probe(argv, **_):
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout='{"runc":{"path":"runc"},"runsc":{"path":"runsc"}}',
                stderr="",
            )

        with tempfile.TemporaryDirectory() as workspace:
            request = LocalHarnessRequest(
                identity=_identity(),
                image="example/harness",
                image_digest="c" * 64,
                harness_argv=("harness",),
                host_workspace=workspace,
                timeout_seconds=60,
                sandbox_profile=SandboxProfile.SECURE,
                sandbox_runtime=SandboxRuntime.RUNSC,
            )
            plan = build_docker_launch_plan(request)
            artifact = DockerLocalLauncher(
                runner=runner,
                runtime_probe_runner=runtime_probe,
            ).run(request)
        runtime_index = plan.argv.index("--runtime")
        self.assertEqual(plan.argv[runtime_index + 1], "runsc")
        self.assertIs(artifact.status, ProducerExecutionStatus.COMPLETED)
        self.assertTrue(artifact.payload["runtime_evidence"]["registration_verified"])
        self.assertTrue(artifact.payload["runtime_evidence"]["runtime_request_accepted"])

    def test_linux_host_gateway_alias_is_opt_in_and_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            plan = build_docker_launch_plan(
                LocalHarnessRequest(
                    identity=_identity(),
                    image="example/harness",
                    image_digest="c" * 64,
                    harness_argv=("harness",),
                    host_workspace=workspace,
                    timeout_seconds=60,
                    add_host_gateway=True,
                )
            )
        gateway_index = plan.argv.index("--add-host")
        self.assertEqual(
            plan.argv[gateway_index : gateway_index + 2],
            ("--add-host", "host.docker.internal:host-gateway"),
        )

    def test_local_content_addressed_image_is_not_rewritten_as_registry_digest(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            image_id = "c" * 64
            plan = build_docker_launch_plan(
                LocalHarnessRequest(
                    identity=_identity(),
                    image=f"sha256:{image_id}",
                    image_digest=image_id,
                    harness_argv=("harness",),
                    host_workspace=workspace,
                    timeout_seconds=60,
                )
            )
        self.assertEqual(plan.argv[-2], f"sha256:{image_id}")

    def test_launcher_returns_evidence_and_passes_shell_false(self) -> None:
        calls = []

        def runner(*args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args[0], 0, stdout="trace proxy-secret\n", stderr="")

        with tempfile.TemporaryDirectory() as workspace:
            request = LocalHarnessRequest(
                identity=_identity(),
                image="example/harness",
                image_digest="c" * 64,
                harness_argv=("harness",),
                host_workspace=workspace,
                timeout_seconds=60,
                model_proxy_url="http://host.docker.internal:8088",
                model_proxy_api_key="proxy-secret",
                network_policy=SandboxNetworkPolicy.BRIDGE_UNRESTRICTED,
            )
            artifact = DockerLocalLauncher(runner=runner).run(request)
        self.assertIs(artifact.status, ProducerExecutionStatus.COMPLETED)
        self.assertEqual(artifact.payload["stdout"], "trace [REDACTED]\n")
        self.assertNotIn("proxy-secret", str(artifact.to_dict()))
        self.assertFalse(calls[0][1]["shell"])
        self.assertEqual(calls[0][1]["env"]["AGENT_EPISODE_ID"], "episode-local")
        self.assertEqual(calls[0][1]["env"]["OPENAI_API_KEY"], "proxy-secret")
        self.assertNotIn("proxy-secret", str(build_docker_launch_plan(request).to_dict()))

    def test_docker_cli_failure_is_infrastructure_invalid(self) -> None:
        def runner(argv, **_):
            return subprocess.CompletedProcess(argv, 125, stdout="", stderr="bad mount")

        with tempfile.TemporaryDirectory() as workspace:
            artifact = DockerLocalLauncher(runner=runner).run(
                LocalHarnessRequest(
                    identity=_identity(),
                    image="example/harness",
                    image_digest="c" * 64,
                    harness_argv=("harness",),
                    host_workspace=workspace,
                    timeout_seconds=60,
                )
            )
        self.assertIs(artifact.status, ProducerExecutionStatus.INFRA_INVALID)

    def test_timeout_force_removes_only_the_identity_bound_container(self) -> None:
        cleanup_calls = []

        def runner(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(("docker", "run"), 1)

        def cleanup_runner(argv, **kwargs):
            cleanup_calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as workspace:
            artifact = DockerLocalLauncher(
                runner=runner,
                cleanup_runner=cleanup_runner,
            ).run(
                LocalHarnessRequest(
                    identity=_identity(),
                    image="example/harness",
                    image_digest="c" * 64,
                    harness_argv=("harness",),
                    host_workspace=workspace,
                    timeout_seconds=1,
                )
            )
        self.assertIs(artifact.status, ProducerExecutionStatus.TIMEOUT)
        self.assertEqual(
            cleanup_calls[0][0],
            ("docker", "rm", "--force", "agent-run-" + _identity().checksum[:16]),
        )
        self.assertFalse(cleanup_calls[0][1]["shell"])

    def test_artifact_store_keeps_process_output_out_of_producer_payload(self) -> None:
        def runner(argv, **_):
            return subprocess.CompletedProcess(argv, 0, stdout="large output", stderr="warn")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = DockerLocalLauncher(
                runner=runner,
                artifact_store=ArtifactStore(root / "objects"),
                clock=lambda: "2026-08-22T00:00:00Z",
            ).run(
                LocalHarnessRequest(
                    identity=_identity(),
                    image="example/harness",
                    image_digest="c" * 64,
                    harness_argv=("harness",),
                    host_workspace=temporary,
                    timeout_seconds=60,
                )
            )
            refs = artifact.payload["output_artifacts"]
            self.assertNotIn("stdout", artifact.payload)
            self.assertNotIn("large output", str(artifact.to_dict()))
            self.assertEqual(len(refs), 2)
            self.assertEqual((root / "objects" / refs[0]["sha256"]).read_text(), "large output")


if __name__ == "__main__":
    unittest.main()
