"""Local Docker launcher with explicit identity and isolation controls."""

from __future__ import annotations

import os
import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.capture.event_writer import ArtifactStore
from src.contracts._json import freeze_json, sha256_json, thaw_json, validate_sha256
from src.contracts._validation import optional_text, required_text
from src.contracts.execution_identity import ExecutionIdentity
from src.errors import ContractValidationError
from src.producers import (
    ProducerArtifact,
    ProducerCapability,
    ProducerExecutionStatus,
)

LOCAL_DOCKER_LAUNCHER_VERSION = "local-docker-launcher/v1"
_PRODUCER_ID = "local-docker"
_SUPPORTED_PLATFORMS = frozenset({"linux/amd64", "linux/arm64"})


def _resolve_image_reference(image: str, image_digest: str) -> str:
    """Pin registry images, but accept a matching locally content-addressed tag.

    Docker image IDs are not registry RepoDigests. Appending an image ID to a tag
    as ``tag@sha256:<id>`` makes Docker try to pull it from a registry. When the
    local tag already resolves to the requested image ID, using the tag is safe
    and keeps offline/local Harness runs working.
    """
    if image.startswith("sha256:"):
        return image
    try:
        inspected = subprocess.run(
            ("docker", "image", "inspect", "--format", "{{.Id}}", image),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        inspected = None
    if inspected is not None and inspected.returncode == 0:
        local_id = inspected.stdout.strip()
        if local_id == f"sha256:{image_digest}":
            return image
    return f"{image}@sha256:{image_digest}"


class SandboxNetworkPolicy(str, Enum):
    NONE = "NONE"
    BRIDGE_UNRESTRICTED = "BRIDGE_UNRESTRICTED"


class SandboxProfile(str, Enum):
    """Execution profiles keep throughput and untrusted-code claims separate."""

    FAST = "FAST"
    SECURE = "SECURE"


class SandboxRuntime(str, Enum):
    RUNC = "runc"
    RUNSC = "runsc"


@dataclass(frozen=True, slots=True, kw_only=True)
class SandboxLimits:
    cpus: float = 2.0
    memory_mb: int = 4096
    pids: int = 256
    tmpfs_mb: int = 512

    def __post_init__(self) -> None:
        if isinstance(self.cpus, bool) or not isinstance(self.cpus, (int, float)):
            raise ContractValidationError("cpus must be numeric")
        if not 0.25 <= float(self.cpus) <= 64:
            raise ContractValidationError("cpus must be between 0.25 and 64")
        object.__setattr__(self, "cpus", float(self.cpus))
        for name in ("memory_mb", "pids", "tmpfs_mb"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ContractValidationError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpus": self.cpus,
            "memory_mb": self.memory_mb,
            "pids": self.pids,
            "tmpfs_mb": self.tmpfs_mb,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalHarnessRequest:
    identity: ExecutionIdentity
    image: str
    image_digest: str
    harness_argv: tuple[str, ...]
    host_workspace: str
    timeout_seconds: float
    platform: str = "linux/amd64"
    model_proxy_url: str | None = None
    model_proxy_api_key: str | None = field(default=None, repr=False)
    trace_ingest_url: str | None = None
    trace_ingest_api_key: str | None = field(default=None, repr=False)
    add_host_gateway: bool = False
    run_as_host_user: bool = True
    network_policy: SandboxNetworkPolicy = SandboxNetworkPolicy.NONE
    sandbox_profile: SandboxProfile = SandboxProfile.FAST
    sandbox_runtime: SandboxRuntime = SandboxRuntime.RUNC
    limits: SandboxLimits = field(default_factory=SandboxLimits)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = LOCAL_DOCKER_LAUNCHER_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ExecutionIdentity):
            raise ContractValidationError("identity must be an ExecutionIdentity")
        if (
            self.identity.producer_id != _PRODUCER_ID
            or self.identity.producer_version != LOCAL_DOCKER_LAUNCHER_VERSION
        ):
            raise ContractValidationError("identity does not belong to local Docker launcher")
        required_text(self.image, "image")
        validate_sha256(self.image_digest, "image_digest")
        argv = tuple(self.harness_argv)
        if not argv or any(not isinstance(item, str) or not item for item in argv):
            raise ContractValidationError("harness_argv must contain non-empty strings")
        object.__setattr__(self, "harness_argv", argv)
        workspace = Path(required_text(self.host_workspace, "host_workspace"))
        if not workspace.is_absolute():
            raise ContractValidationError("host_workspace must be an absolute path")
        if workspace.is_symlink() or not workspace.is_dir():
            raise ContractValidationError("host_workspace must be a real directory")
        object.__setattr__(self, "host_workspace", str(workspace.resolve()))
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or float(self.timeout_seconds) <= 0
        ):
            raise ContractValidationError("timeout_seconds must be positive")
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        if self.platform not in _SUPPORTED_PLATFORMS:
            raise ContractValidationError(f"platform must be one of {sorted(_SUPPORTED_PLATFORMS)}")
        optional_text(self.model_proxy_url, "model_proxy_url")
        optional_text(self.model_proxy_api_key, "model_proxy_api_key")
        if (self.model_proxy_url is None) != (self.model_proxy_api_key is None):
            raise ContractValidationError(
                "model_proxy_url and model_proxy_api_key must be configured together"
            )
        optional_text(self.trace_ingest_url, "trace_ingest_url")
        optional_text(self.trace_ingest_api_key, "trace_ingest_api_key")
        if (self.trace_ingest_url is None) != (self.trace_ingest_api_key is None):
            raise ContractValidationError(
                "trace_ingest_url and trace_ingest_api_key must be configured together"
            )
        if not isinstance(self.add_host_gateway, bool):
            raise ContractValidationError("add_host_gateway must be a boolean")
        if not isinstance(self.run_as_host_user, bool):
            raise ContractValidationError("run_as_host_user must be a boolean")
        if not isinstance(self.network_policy, SandboxNetworkPolicy):
            raise ContractValidationError("network_policy must be a SandboxNetworkPolicy")
        if not isinstance(self.sandbox_profile, SandboxProfile):
            raise ContractValidationError("sandbox_profile must be a SandboxProfile")
        if not isinstance(self.sandbox_runtime, SandboxRuntime):
            raise ContractValidationError("sandbox_runtime must be a SandboxRuntime")
        if (
            self.sandbox_profile is SandboxProfile.SECURE
            and self.sandbox_runtime is not SandboxRuntime.RUNSC
        ):
            raise ContractValidationError("SECURE sandbox profile requires the runsc runtime")
        if self.model_proxy_url is not None and self.network_policy is SandboxNetworkPolicy.NONE:
            raise ContractValidationError("model proxy requires an explicit network policy")
        if self.trace_ingest_url is not None and self.network_policy is SandboxNetworkPolicy.NONE:
            raise ContractValidationError("trace ingestion requires an explicit network policy")
        if not isinstance(self.limits, SandboxLimits):
            raise ContractValidationError("limits must be SandboxLimits")
        frozen = freeze_json(self.metadata)
        if not isinstance(frozen, Mapping):
            raise ContractValidationError("metadata must be an object")
        object.__setattr__(self, "metadata", frozen)
        if self.schema_version != LOCAL_DOCKER_LAUNCHER_VERSION:
            raise ContractValidationError("unsupported LocalHarnessRequest schema_version")


@dataclass(frozen=True, slots=True, kw_only=True)
class DockerLaunchPlan:
    identity_checksum: str
    container_name: str
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    secret_environment: Mapping[str, str] = field(repr=False)
    network_policy: SandboxNetworkPolicy
    image_digest: str
    platform: str
    sandbox_profile: SandboxProfile
    sandbox_runtime: SandboxRuntime
    schema_version: str = LOCAL_DOCKER_LAUNCHER_VERSION

    def __post_init__(self) -> None:
        validate_sha256(self.identity_checksum, "identity_checksum")
        required_text(self.container_name, "container_name")
        validate_sha256(self.image_digest, "image_digest")
        if self.platform not in _SUPPORTED_PLATFORMS:
            raise ContractValidationError("unsupported container platform")
        if not isinstance(self.sandbox_profile, SandboxProfile):
            raise ContractValidationError("sandbox_profile must be a SandboxProfile")
        if not isinstance(self.sandbox_runtime, SandboxRuntime):
            raise ContractValidationError("sandbox_runtime must be a SandboxRuntime")
        argv = tuple(self.argv)
        if not argv or any(not isinstance(item, str) or not item for item in argv):
            raise ContractValidationError("argv must contain non-empty strings")
        object.__setattr__(self, "argv", argv)
        environment = dict(self.environment)
        if any(not key or not isinstance(value, str) for key, value in environment.items()):
            raise ContractValidationError("environment must contain string keys and values")
        object.__setattr__(self, "environment", environment)
        secret_environment = dict(self.secret_environment)
        if any(
            not key or not isinstance(value, str) or not value
            for key, value in secret_environment.items()
        ):
            raise ContractValidationError("secret_environment contains invalid values")
        if set(environment) & set(secret_environment):
            raise ContractValidationError("public and secret environment keys overlap")
        object.__setattr__(self, "secret_environment", secret_environment)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity_checksum": self.identity_checksum,
            "container_name": self.container_name,
            "argv": list(self.argv),
            "environment": dict(sorted(self.environment.items())),
            "secret_environment_keys": sorted(self.secret_environment),
            "network_policy": self.network_policy.value,
            "image_digest": self.image_digest,
            "platform": self.platform,
            "sandbox_profile": self.sandbox_profile.value,
            "sandbox_runtime": self.sandbox_runtime.value,
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


def build_docker_launch_plan(request: LocalHarnessRequest) -> DockerLaunchPlan:
    if not isinstance(request, LocalHarnessRequest):
        raise TypeError("request must be a LocalHarnessRequest")
    identity = request.identity
    container_name = f"agent-run-{identity.checksum[:16]}"
    network = {
        SandboxNetworkPolicy.NONE: "none",
        SandboxNetworkPolicy.BRIDGE_UNRESTRICTED: "bridge",
    }[request.network_policy]
    environment = {
        "AGENT_RUN_ID": identity.run_id,
        "AGENT_TASK_ID": identity.task_id,
        "AGENT_EPISODE_ID": identity.episode_id,
        "AGENT_ATTEMPT_ID": str(identity.attempt_id),
        "AGENT_EXECUTION_IDENTITY_CHECKSUM": identity.checksum,
    }
    if request.model_proxy_url is not None:
        environment["AGENT_MODEL_PROXY_URL"] = request.model_proxy_url
        environment["OPENAI_BASE_URL"] = request.model_proxy_url
    if request.trace_ingest_url is not None:
        environment["AGENT_TRACE_URL"] = request.trace_ingest_url
    secret_environment = {}
    if request.model_proxy_api_key is not None:
        secret_environment = {
            "AGENT_MODEL_PROXY_API_KEY": request.model_proxy_api_key,
            "OPENAI_API_KEY": request.model_proxy_api_key,
        }
    if request.trace_ingest_api_key is not None:
        secret_environment["AGENT_TRACE_API_KEY"] = request.trace_ingest_api_key
    image_reference = _resolve_image_reference(request.image, request.image_digest)
    argv = (
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--platform",
        request.platform,
        "--runtime",
        request.sandbox_runtime.value,
        "--label",
        f"agent.run_id={identity.run_id}",
        "--label",
        f"agent.episode_id={identity.episode_id}",
        "--network",
        network,
        *(
            ("--add-host", "host.docker.internal:host-gateway")
            if request.add_host_gateway
            else ()
        ),
        *(
            ("--user", f"{os.getuid()}:{os.getgid()}")
            if request.run_as_host_user
            else ()
        ),
        "--cpus",
        str(request.limits.cpus),
        "--memory",
        f"{request.limits.memory_mb}m",
        "--pids-limit",
        str(request.limits.pids),
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        f"/tmp:rw,noexec,nosuid,size={request.limits.tmpfs_mb}m",
        "--mount",
        f"type=bind,source={request.host_workspace},target=/workspace",
        "--workdir",
        "/workspace",
        *(
            item
            for key in sorted(set(environment) | set(secret_environment))
            for item in ("--env", key)
        ),
        image_reference,
        *request.harness_argv,
    )
    return DockerLaunchPlan(
        identity_checksum=identity.checksum,
        container_name=container_name,
        argv=argv,
        environment=environment,
        secret_environment=secret_environment,
        network_policy=request.network_policy,
        image_digest=request.image_digest,
        platform=request.platform,
        sandbox_profile=request.sandbox_profile,
        sandbox_runtime=request.sandbox_runtime,
    )


class DockerLocalLauncher:
    producer_id = _PRODUCER_ID
    producer_version = LOCAL_DOCKER_LAUNCHER_VERSION
    capabilities = frozenset({ProducerCapability.RAW_HARNESS_TRACE})

    def __init__(
        self,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        cleanup_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        runtime_probe_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        artifact_store: ArtifactStore | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self._runner = runner
        self._cleanup_runner = cleanup_runner
        self._runtime_probe_runner = runtime_probe_runner
        self._artifact_store = artifact_store
        self._clock = clock or (
            lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )

    def run(self, request: LocalHarnessRequest) -> ProducerArtifact:
        plan = build_docker_launch_plan(request)
        runtime_evidence, runtime_issue = self._runtime_evidence(request)
        if runtime_issue is not None:
            return self._artifact(
                request,
                plan,
                ProducerExecutionStatus.INFRA_INVALID,
                payload={"runtime_evidence": runtime_evidence},
                issues=(runtime_issue,),
            )
        try:
            completed = self._runner(
                plan.argv,
                env=(os.environ | dict(plan.environment) | dict(plan.secret_environment)),
                capture_output=True,
                text=True,
                timeout=request.timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            cleanup_issue = self._cleanup_container(plan.container_name)
            return self._artifact(
                request,
                plan,
                ProducerExecutionStatus.TIMEOUT,
                payload={
                    "timeout_seconds": request.timeout_seconds,
                    "runtime_evidence": runtime_evidence,
                },
                issues=tuple(
                    item
                    for item in (f"container execution timed out: {exc}", cleanup_issue)
                    if item is not None
                ),
            )
        except KeyboardInterrupt:
            self._cleanup_container(plan.container_name)
            raise
        except OSError as exc:
            return self._artifact(
                request,
                plan,
                ProducerExecutionStatus.INFRA_INVALID,
                payload={"runtime_evidence": runtime_evidence},
                issues=(f"Docker process could not start: {exc}",),
            )
        except Exception:
            self._cleanup_container(plan.container_name)
            raise
        if completed.returncode == 0:
            status = ProducerExecutionStatus.COMPLETED
            issues = ()
        elif completed.returncode in (125, 126, 127):
            status = ProducerExecutionStatus.INFRA_INVALID
            issues = (f"Docker could not start the Harness: exit {completed.returncode}",)
        else:
            status = ProducerExecutionStatus.FAILED
            issues = ()
        output_payload = self._capture_outputs(
            completed.stdout,
            completed.stderr,
            secrets=tuple(plan.secret_environment.values()),
        )
        return self._artifact(
            request,
            plan,
            status,
            payload={
                "returncode": completed.returncode,
                "runtime_evidence": {
                    **runtime_evidence,
                    "runtime_request_accepted": status
                    is not ProducerExecutionStatus.INFRA_INVALID,
                },
                **output_payload,
            },
            issues=issues,
        )

    def _runtime_evidence(
        self,
        request: LocalHarnessRequest,
    ) -> tuple[dict[str, Any], str | None]:
        evidence: dict[str, Any] = {
            "profile": request.sandbox_profile.value,
            "requested_runtime": request.sandbox_runtime.value,
            "selection_mechanism": "docker --runtime",
            "registration_verified": False,
            "runtime_request_accepted": False,
        }
        if request.sandbox_profile is SandboxProfile.FAST:
            evidence["verification_scope"] = "explicit-runtime-selection"
            return evidence, None
        try:
            completed = self._runtime_probe_runner(
                ("docker", "info", "--format", "{{json .Runtimes}}"),
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                shell=False,
            )
        except OSError as exc:
            return evidence, f"Docker runtime preflight could not start: {exc}"
        except subprocess.TimeoutExpired:
            return evidence, "Docker runtime preflight timed out"
        if completed.returncode != 0:
            return evidence, "Docker runtime preflight failed"
        try:
            runtimes = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return evidence, "Docker runtime preflight returned malformed runtime metadata"
        registered = isinstance(runtimes, Mapping) and request.sandbox_runtime.value in runtimes
        evidence["registration_verified"] = registered
        evidence["verification_scope"] = "docker-runtime-registration-plus-explicit-selection"
        if not registered:
            return evidence, f"required Docker runtime is not registered: {request.sandbox_runtime.value}"
        return evidence, None

    def _capture_outputs(
        self,
        stdout: str,
        stderr: str,
        *,
        secrets: tuple[str, ...],
    ) -> dict[str, Any]:
        for secret in sorted(set(secrets), key=len, reverse=True):
            stdout = stdout.replace(secret, "[REDACTED]")
            stderr = stderr.replace(secret, "[REDACTED]")
        if self._artifact_store is None:
            return {"stdout": stdout, "stderr": stderr}
        references = []
        for kind, value in (("process-stdout", stdout), ("process-stderr", stderr)):
            if not value:
                continue
            references.append(
                self._artifact_store.put(
                    value.encode(),
                    kind=kind,
                    media_type="text/plain; charset=utf-8",
                    created_at=self._clock(),
                ).to_dict()
            )
        return {"output_artifacts": references}

    def _cleanup_container(self, container_name: str) -> str | None:
        try:
            completed = self._cleanup_runner(
                ("docker", "rm", "--force", container_name),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                shell=False,
            )
        except OSError as exc:
            return f"container cleanup could not start: {exc}"
        except subprocess.TimeoutExpired:
            return "container cleanup timed out"
        if completed.returncode not in (0, 1):
            return f"container cleanup failed: exit {completed.returncode}"
        return None

    def _artifact(
        self,
        request: LocalHarnessRequest,
        plan: DockerLaunchPlan,
        status: ProducerExecutionStatus,
        *,
        payload: Mapping[str, Any],
        issues: tuple[str, ...] = (),
    ) -> ProducerArtifact:
        return ProducerArtifact(
            identity=request.identity,
            status=status,
            capabilities=self.capabilities,
            payload={
                "launcher_version": self.producer_version,
                "launch_plan_checksum": plan.checksum,
                "image": request.image,
                "image_digest": request.image_digest,
                "platform": request.platform,
                "network_policy": request.network_policy.value,
                "sandbox_profile": request.sandbox_profile.value,
                "sandbox_runtime": request.sandbox_runtime.value,
                "limits": request.limits.to_dict(),
                "metadata": thaw_json(request.metadata),
                **dict(payload),
            },
            issues=issues,
        )
