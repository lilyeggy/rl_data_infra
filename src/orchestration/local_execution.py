"""One-shot Local Launcher transaction from frozen inputs to evidence bundle."""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.assembly import LocalRunFinalization, finalize_local_run
from src.capture import (
    ArtifactStore,
    EnvironmentCapture,
    EventWriter,
    HarnessEventIngress,
    ModelEndpointKind,
    ModelEvidenceJsonlWriter,
    ModelProxyHttpServer,
    ModelProxyService,
    TraceRecorder,
    snapshot_workspace,
    store_workspace_change_evidence,
)
from src.contracts._json import canonical_json_bytes, validate_sha256
from src.contracts._validation import required_text, strict_fields
from src.contracts.agent_episode import CaptureCapability, EpisodeVerifierStatus
from src.contracts.artifacts import ArtifactRef
from src.contracts.execution_identity import ExecutionIdentity
from src.contracts.manifests import (
    EnvironmentManifest,
    EvaluatorManifest,
    HarnessManifest,
    ModelManifest,
)
from src.contracts.run_manifest import ExecutionRunManifest
from src.contracts.trace_event import EventStatus
from src.contracts.verifier_report import LocalVerifierReport, VerifierExecutionStatus
from src.errors import ContractValidationError
from src.launchers import (
    DockerLaunchPlan,
    DockerLocalLauncher,
    LocalHarnessRequest,
    SandboxLimits,
    SandboxNetworkPolicy,
    SandboxProfile,
    SandboxRuntime,
    build_docker_launch_plan,
)
from src.producers import ProducerArtifact, ProducerCapability, ProducerExecutionStatus

LOCAL_EXECUTION_SPEC_VERSION = "local-execution-spec/v5"
LOCAL_EXECUTION_ORCHESTRATOR_VERSION = "local-execution-orchestrator/v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalExecutionSpec:
    """Secret-free, serializable inputs for exactly one local task attempt."""

    identity: ExecutionIdentity
    harness_manifest: HarnessManifest
    model_manifest: ModelManifest
    evaluator_manifest: EvaluatorManifest
    experiment_manifest_ref: str
    image: str
    image_digest: str
    workspace: str
    harness_argv: tuple[str, ...]
    verifier_argv: tuple[str, ...]
    upstream_chat_completions_url: str
    endpoint_kind: ModelEndpointKind
    task_snapshot: str
    capture_capabilities: frozenset[CaptureCapability] = frozenset()
    # Disabling this keeps model/tool observability but makes model calls
    # ineligible for RL because native behavior logprobs are not requested.
    capture_response_logprobs: bool = True
    default_sampling_seed: int | None = None
    platform: str = "linux/amd64"
    # Linux Docker does not provide host.docker.internal unless we add this gateway alias.
    docker_host_gateway: bool = False
    proxy_bind_host: str = "0.0.0.0"
    run_as_host_user: bool = True
    network_policy: SandboxNetworkPolicy = SandboxNetworkPolicy.BRIDGE_UNRESTRICTED
    sandbox_profile: SandboxProfile = SandboxProfile.FAST
    sandbox_runtime: SandboxRuntime = SandboxRuntime.RUNC
    limits: SandboxLimits = SandboxLimits()
    harness_timeout_seconds: float = 300
    verifier_timeout_seconds: float = 300
    trace_id: str = "trace-local"
    schema_version: str = LOCAL_EXECUTION_SPEC_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ExecutionIdentity):
            raise ContractValidationError("identity must be ExecutionIdentity")
        for name, expected in (
            ("harness_manifest", HarnessManifest),
            ("model_manifest", ModelManifest),
            ("evaluator_manifest", EvaluatorManifest),
            ("endpoint_kind", ModelEndpointKind),
            ("network_policy", SandboxNetworkPolicy),
            ("sandbox_profile", SandboxProfile),
            ("sandbox_runtime", SandboxRuntime),
            ("limits", SandboxLimits),
        ):
            if not isinstance(getattr(self, name), expected):
                raise ContractValidationError(f"{name} has invalid type")
        for name in (
            "experiment_manifest_ref",
            "image",
            "workspace",
            "upstream_chat_completions_url",
            "task_snapshot",
            "trace_id",
            "proxy_bind_host",
        ):
            required_text(getattr(self, name), name)
        validate_sha256(self.image_digest, "image_digest")
        if not self.upstream_chat_completions_url.startswith(("http://", "https://")):
            raise ContractValidationError("upstream URL must use http or https")
        for name in (
            "docker_host_gateway",
            "run_as_host_user",
            "capture_response_logprobs",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ContractValidationError(f"{name} must be a boolean")
        if self.default_sampling_seed is not None and (
            isinstance(self.default_sampling_seed, bool)
            or not isinstance(self.default_sampling_seed, int)
            or self.default_sampling_seed < 0
        ):
            raise ContractValidationError("default_sampling_seed must be a non-negative integer or null")
        for name in ("harness_argv", "verifier_argv"):
            argv = tuple(getattr(self, name))
            if not argv or any(not isinstance(item, str) or not item for item in argv):
                raise ContractValidationError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, argv)
        for name in ("harness_timeout_seconds", "verifier_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ContractValidationError(f"{name} must be positive")
            object.__setattr__(self, name, float(value))
        capabilities = frozenset(self.capture_capabilities)
        if any(not isinstance(item, CaptureCapability) for item in capabilities):
            raise ContractValidationError("capture_capabilities contains unknown values")
        object.__setattr__(self, "capture_capabilities", capabilities)
        if self.network_policy is SandboxNetworkPolicy.NONE:
            raise ContractValidationError("one-shot execution requires host proxy network access")
        if self.schema_version != LOCAL_EXECUTION_SPEC_VERSION:
            raise ContractValidationError("unsupported LocalExecutionSpec schema_version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "harness_manifest": self.harness_manifest.to_dict(),
            "model_manifest": self.model_manifest.to_dict(),
            "evaluator_manifest": self.evaluator_manifest.to_dict(),
            "experiment_manifest_ref": self.experiment_manifest_ref,
            "image": self.image,
            "image_digest": self.image_digest,
            "workspace": self.workspace,
            "harness_argv": list(self.harness_argv),
            "verifier_argv": list(self.verifier_argv),
            "upstream_chat_completions_url": self.upstream_chat_completions_url,
            "endpoint_kind": self.endpoint_kind.value,
            "task_snapshot": self.task_snapshot,
            "capture_capabilities": sorted(item.value for item in self.capture_capabilities),
            "capture_response_logprobs": self.capture_response_logprobs,
            "default_sampling_seed": self.default_sampling_seed,
            "platform": self.platform,
            "docker_host_gateway": self.docker_host_gateway,
            "proxy_bind_host": self.proxy_bind_host,
            "run_as_host_user": self.run_as_host_user,
            "network_policy": self.network_policy.value,
            "sandbox_profile": self.sandbox_profile.value,
            "sandbox_runtime": self.sandbox_runtime.value,
            "limits": self.limits.to_dict(),
            "harness_timeout_seconds": self.harness_timeout_seconds,
            "verifier_timeout_seconds": self.verifier_timeout_seconds,
            "trace_id": self.trace_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LocalExecutionSpec":
        payload = strict_fields(
            value,
            {
                "schema_version",
                "identity",
                "harness_manifest",
                "model_manifest",
                "evaluator_manifest",
                "experiment_manifest_ref",
                "image",
                "image_digest",
                "workspace",
                "harness_argv",
                "verifier_argv",
                "upstream_chat_completions_url",
                "endpoint_kind",
                "task_snapshot",
                "capture_capabilities",
                "capture_response_logprobs",
                "default_sampling_seed",
                "platform",
                "docker_host_gateway",
                "proxy_bind_host",
                "run_as_host_user",
                "network_policy",
                "sandbox_profile",
                "sandbox_runtime",
                "limits",
                "harness_timeout_seconds",
                "verifier_timeout_seconds",
                "trace_id",
            },
            "LocalExecutionSpec",
        )
        limits = payload["limits"]
        if not isinstance(limits, Mapping):
            raise ContractValidationError("limits must be an object")
        return cls(
            schema_version=payload["schema_version"],
            identity=ExecutionIdentity.from_dict(payload["identity"]),
            harness_manifest=HarnessManifest.from_dict(payload["harness_manifest"]),
            model_manifest=ModelManifest.from_dict(payload["model_manifest"]),
            evaluator_manifest=EvaluatorManifest.from_dict(payload["evaluator_manifest"]),
            experiment_manifest_ref=payload["experiment_manifest_ref"],
            image=payload["image"],
            image_digest=payload["image_digest"],
            workspace=payload["workspace"],
            harness_argv=tuple(payload["harness_argv"]),
            verifier_argv=tuple(payload["verifier_argv"]),
            upstream_chat_completions_url=payload["upstream_chat_completions_url"],
            endpoint_kind=ModelEndpointKind(payload["endpoint_kind"]),
            task_snapshot=payload["task_snapshot"],
            capture_capabilities=frozenset(
                CaptureCapability(item) for item in payload["capture_capabilities"]
            ),
            capture_response_logprobs=payload["capture_response_logprobs"],
            default_sampling_seed=payload["default_sampling_seed"],
            platform=payload["platform"],
            docker_host_gateway=payload["docker_host_gateway"],
            proxy_bind_host=payload["proxy_bind_host"],
            run_as_host_user=payload["run_as_host_user"],
            network_policy=SandboxNetworkPolicy(payload["network_policy"]),
            sandbox_profile=SandboxProfile(payload.get("sandbox_profile", "FAST")),
            sandbox_runtime=SandboxRuntime(payload.get("sandbox_runtime", "runc")),
            limits=SandboxLimits(**dict(limits)),
            harness_timeout_seconds=payload["harness_timeout_seconds"],
            verifier_timeout_seconds=payload["verifier_timeout_seconds"],
            trace_id=payload["trace_id"],
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedLocalExecution:
    request: LocalHarnessRequest
    launch_plan: DockerLaunchPlan
    manifest: ExecutionRunManifest


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalExecutionResult:
    finalization: LocalRunFinalization
    manifest: ExecutionRunManifest
    producer_artifact: ProducerArtifact
    output_dir: Path


def prepare_local_execution(
    spec: LocalExecutionSpec,
    *,
    proxy_base_url: str,
    access_token: str,
    created_at: str | None = None,
) -> PreparedLocalExecution:
    """Freeze environment metadata and launch plan before the Harness starts."""

    request = LocalHarnessRequest(
        identity=spec.identity,
        image=spec.image,
        image_digest=spec.image_digest,
        harness_argv=spec.harness_argv,
        host_workspace=str(Path(spec.workspace).resolve()),
        timeout_seconds=spec.harness_timeout_seconds,
        platform=spec.platform,
        model_proxy_url=proxy_base_url,
        model_proxy_api_key=access_token,
        trace_ingest_url=f"{proxy_base_url.rstrip('/')}/v1/harness/events",
        trace_ingest_api_key=access_token,
        add_host_gateway=spec.docker_host_gateway,
        run_as_host_user=spec.run_as_host_user,
        network_policy=spec.network_policy,
        sandbox_profile=spec.sandbox_profile,
        sandbox_runtime=spec.sandbox_runtime,
        limits=spec.limits,
        metadata={"orchestrator_version": LOCAL_EXECUTION_ORCHESTRATOR_VERSION},
    )
    plan = build_docker_launch_plan(request)
    guaranteed = {
        CaptureCapability.MODEL_IO,
        CaptureCapability.MODEL_TOKEN_USAGE,
        CaptureCapability.SANDBOX_LIFECYCLE,
        CaptureCapability.SANDBOX_COMMAND_IO,
        CaptureCapability.FILE_ARTIFACTS,
        CaptureCapability.VERIFIER_EVIDENCE,
    }
    manifest = ExecutionRunManifest(
        identity=spec.identity,
        harness_manifest=spec.harness_manifest,
        model_manifest=spec.model_manifest,
        environment_manifest=EnvironmentManifest(
            runtime_type=f"docker-{spec.sandbox_runtime.value}",
            revision=(
                f"{spec.platform}:{spec.image_digest}:"
                f"{spec.sandbox_profile.value}:{spec.sandbox_runtime.value}"
            ),
            image=f"{spec.image}@sha256:{spec.image_digest}",
            resource_limits={
                **spec.limits.to_dict(),
                "harness_timeout_seconds": spec.harness_timeout_seconds,
                "verifier_timeout_seconds": spec.verifier_timeout_seconds,
            },
            network_policy=spec.network_policy.value,
            task_snapshot=spec.task_snapshot,
        ),
        evaluator_manifest=spec.evaluator_manifest,
        experiment_manifest_ref=spec.experiment_manifest_ref,
        capture_capabilities=frozenset(guaranteed | set(spec.capture_capabilities)),
        launcher_plan_checksum=plan.checksum,
        created_at=created_at or _utc_now(),
    )
    return PreparedLocalExecution(request=request, launch_plan=plan, manifest=manifest)


class LocalExecutionOrchestrator:
    """Own proxy, Harness, verifier, finalization and cleanup as one transaction."""

    def __init__(
        self,
        *,
        launcher_factory: Callable[..., DockerLocalLauncher] = DockerLocalLauncher,
        access_token_factory: Callable[[], str] | None = None,
        clock: Callable[[], str] = _utc_now,
    ) -> None:
        self._launcher_factory = launcher_factory
        self._access_token_factory = access_token_factory or (lambda: secrets.token_urlsafe(32))
        self._clock = clock

    def run(
        self,
        spec: LocalExecutionSpec,
        *,
        output_dir: str | Path,
        upstream_authorization: str | None = None,
        stage_observer: Callable[[str], None] | None = None,
    ) -> LocalExecutionResult:
        root = Path(output_dir).resolve()
        self._claim_output_directory(root)
        events_path = root / "raw-events.jsonl"
        evidence_path = root / "model-evidence.jsonl"
        artifact_manifest_path = root / "artifact-refs.json"
        artifact_store = ArtifactStore(root / "objects")
        recorder = TraceRecorder(
            EventWriter(events_path),
            run_id=spec.identity.run_id,
            episode_id=spec.identity.episode_id,
            trace_id=spec.trace_id,
            clock=self._clock,
        )
        access_token = self._access_token_factory()
        service = ModelProxyService(
            identity=spec.identity,
            endpoint_kind=spec.endpoint_kind,
            upstream_chat_completions_url=spec.upstream_chat_completions_url,
            evidence_writer=ModelEvidenceJsonlWriter(evidence_path),
            recorder=recorder,
            upstream_authorization=upstream_authorization,
            access_token=access_token,
            timeout_seconds=spec.harness_timeout_seconds,
            backend_model_revision=spec.model_manifest.revision,
            capture_response_logprobs=spec.capture_response_logprobs,
            default_sampling_seed=spec.default_sampling_seed,
            on_first_model_request=(
                (lambda: stage_observer("RUN")) if stage_observer is not None else None
            ),
        )
        server = ModelProxyHttpServer(
            service,
            harness_ingress=HarnessEventIngress(
                identity=spec.identity,
                recorder=recorder,
            ),
            host=spec.proxy_bind_host,
            port=0,
        )
        server.start_in_thread()
        refs: dict[str, ArtifactRef] = {}
        try:
            proxy_base_url = f"http://host.docker.internal:{server.address[1]}"
            prepared = prepare_local_execution(
                spec,
                proxy_base_url=proxy_base_url,
                access_token=access_token,
                created_at=self._clock(),
            )
            self._write_json(root / "execution-run-manifest.json", prepared.manifest.to_dict())
            self._write_json(
                root / "launch-plan.json",
                prepared.launch_plan.to_dict() | {"checksum": prepared.launch_plan.checksum},
            )
            launcher = self._launcher_factory(
                artifact_store=artifact_store,
                clock=self._clock,
            )
            environment = EnvironmentCapture(recorder)
            workspace_before = snapshot_workspace(spec.workspace)
            sandbox_span = "span-sandbox"
            environment.sandbox_started(
                span_id=sandbox_span,
                runtime_id=prepared.launch_plan.container_name,
                attributes={"launch_plan_checksum": prepared.launch_plan.checksum},
            )
            harness_artifact = launcher.run(prepared.request)
            if stage_observer is not None:
                stage_observer("POSTRUN")
            harness_refs = self._output_refs(harness_artifact)
            workspace_after = snapshot_workspace(spec.workspace)
            workspace_refs = store_workspace_change_evidence(
                workspace_before,
                workspace_after,
                store=artifact_store,
                created_at=self._clock(),
            )
            self._index_refs(refs, (*harness_refs, *workspace_refs))
            environment.sandbox_finished(
                span_id=sandbox_span,
                runtime_id=prepared.launch_plan.container_name,
                status=self._event_status(harness_artifact.status),
                artifact_refs=tuple(item.artifact_id for item in (*harness_refs, *workspace_refs)),
            )

            verifier_artifact: ProducerArtifact | None = None
            verifier_report: ArtifactRef | None = None
            verifier_event_id: str | None = None
            verifier_status = EpisodeVerifierStatus.NOT_RUN
            score: float | None = None
            if harness_artifact.status not in {
                ProducerExecutionStatus.INFRA_INVALID,
                ProducerExecutionStatus.TIMEOUT,
            }:
                verifier_span = "span-verifier"
                environment.verification_started(
                    span_id=verifier_span,
                    verifier=spec.evaluator_manifest.name,
                )
                verifier_request = LocalHarnessRequest(
                    identity=spec.identity,
                    image=spec.image,
                    image_digest=spec.image_digest,
                    harness_argv=spec.verifier_argv,
                    host_workspace=str(Path(spec.workspace).resolve()),
                    timeout_seconds=spec.verifier_timeout_seconds,
                    platform=spec.platform,
                    run_as_host_user=spec.run_as_host_user,
                    network_policy=SandboxNetworkPolicy.NONE,
                    sandbox_profile=spec.sandbox_profile,
                    sandbox_runtime=spec.sandbox_runtime,
                    limits=spec.limits,
                    metadata={"role": "verifier"},
                )
                verifier_artifact = launcher.run(verifier_request)
                verifier_refs = self._output_refs(verifier_artifact)
                self._index_refs(refs, verifier_refs)
                verifier_status, verifier_event_status, score = self._verifier_outcome(
                    verifier_artifact
                )
                report_payload = LocalVerifierReport(
                    identity_checksum=spec.identity.checksum,
                    evaluator=spec.evaluator_manifest,
                    command=spec.verifier_argv,
                    execution_status=VerifierExecutionStatus(verifier_artifact.status.value),
                    verifier_status=verifier_status,
                    score=score,
                    producer_artifact_checksum=verifier_artifact.checksum,
                    output_artifact_checksums=tuple(item.checksum for item in verifier_refs),
                    created_at=self._clock(),
                )
                verifier_report = artifact_store.put(
                    canonical_json_bytes(report_payload.to_dict()),
                    kind="verifier-report",
                    media_type="application/json",
                    created_at=self._clock(),
                )
                self._index_refs(refs, (verifier_report,))
                finished = environment.verification_finished(
                    span_id=verifier_span,
                    verifier=spec.evaluator_manifest.name,
                    passed=(
                        True
                        if verifier_status is EpisodeVerifierStatus.PASSED
                        else False
                        if verifier_status is EpisodeVerifierStatus.FAILED
                        else None
                    ),
                    status=verifier_event_status,
                    score=score,
                    artifact_refs=tuple(
                        item.artifact_id for item in (*verifier_refs, verifier_report)
                    ),
                )
                verifier_event_id = finished.event_id

            task_status, execution_validity, termination_reason = self._terminal_outcome(
                harness_artifact.status,
                verifier_status,
            )
            recorder.finish(
                span_id="span-episode",
                task_status=task_status,
                execution_validity=execution_validity,
                verifier_status=verifier_status.value,
                termination_reason=termination_reason,
                score=score,
                evidence_event_ids=(verifier_event_id,) if verifier_event_id else (),
            )
            producer = self._combined_artifact(
                prepared,
                harness_artifact,
                verifier_artifact,
                verifier_report,
                workspace_issues=workspace_before.issues + workspace_after.issues,
            )
            self._write_json(root / "producer-artifact.json", producer.to_dict())
            self._write_json(
                artifact_manifest_path,
                [refs[key].to_dict() for key in sorted(refs)],
            )
            finalization = finalize_local_run(
                manifest=prepared.manifest,
                producer_artifact=producer,
                events_path=events_path,
                model_evidence_path=evidence_path,
                artifacts_path=artifact_manifest_path,
            )
            finalized = root / "finalized"
            self._write_json(finalized / "episode.json", finalization.episode.to_dict())
            self._write_json(
                finalized / "execution-bundle.json",
                finalization.execution_bundle.to_dict(),
            )
            self._write_json(
                finalized / "finalization.json",
                {
                    "manifest_checksum": prepared.manifest.checksum,
                    "producer_artifact_checksum": producer.checksum,
                    "episode_checksum": finalization.episode.checksum,
                    "execution_bundle_checksum": finalization.execution_bundle.checksum,
                    "warnings": list(finalization.warnings),
                },
            )
            return LocalExecutionResult(
                finalization=finalization,
                manifest=prepared.manifest,
                producer_artifact=producer,
                output_dir=root,
            )
        finally:
            server.close()

    @staticmethod
    def _claim_output_directory(root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        conflicts = sorted(path.name for path in root.iterdir())
        if conflicts:
            raise ContractValidationError(
                f"output directory must be empty for an immutable execution: {conflicts}"
            )

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode()
            + b"\n"
        )
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    @staticmethod
    def _output_refs(artifact: ProducerArtifact) -> tuple[ArtifactRef, ...]:
        values = artifact.payload.get("output_artifacts", ())
        if not isinstance(values, (tuple, list)):
            raise ContractValidationError("launcher output_artifacts must be an array")
        return tuple(ArtifactRef.from_dict(value) for value in values)

    @staticmethod
    def _index_refs(index: dict[str, ArtifactRef], values: tuple[ArtifactRef, ...]) -> None:
        for value in values:
            prior = index.get(value.artifact_id)
            if prior is not None and prior != value:
                raise ContractValidationError("conflicting artifact reference")
            index[value.artifact_id] = value

    @staticmethod
    def _event_status(status: ProducerExecutionStatus) -> EventStatus:
        return {
            ProducerExecutionStatus.COMPLETED: EventStatus.SUCCEEDED,
            ProducerExecutionStatus.FAILED: EventStatus.FAILED,
            ProducerExecutionStatus.INFRA_INVALID: EventStatus.ERROR,
            ProducerExecutionStatus.TIMEOUT: EventStatus.TIMEOUT,
        }[status]

    @staticmethod
    def _verifier_outcome(
        artifact: ProducerArtifact,
    ) -> tuple[EpisodeVerifierStatus, EventStatus, float | None]:
        if artifact.status is ProducerExecutionStatus.COMPLETED:
            return EpisodeVerifierStatus.PASSED, EventStatus.SUCCEEDED, 1.0
        if artifact.status is ProducerExecutionStatus.FAILED:
            return EpisodeVerifierStatus.FAILED, EventStatus.FAILED, 0.0
        if artifact.status is ProducerExecutionStatus.TIMEOUT:
            return EpisodeVerifierStatus.TIMEOUT, EventStatus.TIMEOUT, None
        return EpisodeVerifierStatus.ERROR, EventStatus.ERROR, None

    @staticmethod
    def _terminal_outcome(
        harness_status: ProducerExecutionStatus,
        verifier_status: EpisodeVerifierStatus,
    ) -> tuple[str, str, str]:
        if harness_status in {
            ProducerExecutionStatus.INFRA_INVALID,
            ProducerExecutionStatus.TIMEOUT,
        }:
            return "UNKNOWN", "INFRA_INVALID", f"HARNESS_{harness_status.value}"
        if verifier_status is EpisodeVerifierStatus.PASSED:
            return "SUCCESS", "VALID", "VERIFIER_PASSED"
        if verifier_status is EpisodeVerifierStatus.FAILED:
            return "FAILURE", "VALID", "VERIFIER_FAILED"
        return "UNKNOWN", "INFRA_INVALID", f"VERIFIER_{verifier_status.value}"

    @staticmethod
    def _combined_artifact(
        prepared: PreparedLocalExecution,
        harness: ProducerArtifact,
        verifier: ProducerArtifact | None,
        report: ArtifactRef | None,
        *,
        workspace_issues: tuple[str, ...],
    ) -> ProducerArtifact:
        status = harness.status
        if status is ProducerExecutionStatus.COMPLETED and verifier is not None:
            if verifier.status in {
                ProducerExecutionStatus.INFRA_INVALID,
                ProducerExecutionStatus.TIMEOUT,
            }:
                status = ProducerExecutionStatus.INFRA_INVALID
        capabilities = {
            ProducerCapability.RAW_HARNESS_TRACE,
            ProducerCapability.HARNESS_EVENTS,
        }
        if report is not None:
            capabilities.add(ProducerCapability.VERIFIER_EVIDENCE)
        payload: dict[str, Any] = {
            "orchestrator_version": LOCAL_EXECUTION_ORCHESTRATOR_VERSION,
            "launch_plan_checksum": prepared.launch_plan.checksum,
            "harness_artifact_checksum": harness.checksum,
            "harness_result": harness.to_dict(),
        }
        if verifier is not None:
            payload["verifier_artifact_checksum"] = verifier.checksum
            payload["verifier_result"] = verifier.to_dict()
        if report is not None:
            payload["verifier_report_checksum"] = report.sha256
        if workspace_issues:
            payload["workspace_capture_issues"] = list(workspace_issues)
        return ProducerArtifact(
            identity=prepared.manifest.identity,
            status=status,
            capabilities=frozenset(capabilities),
            payload=payload,
            issues=(
                harness.issues
                + (verifier.issues if verifier is not None else ())
                + workspace_issues
            ),
        )
