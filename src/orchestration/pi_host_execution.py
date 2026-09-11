"""Certified host-side Pi executions for small, reproducible SWE-bench batches.

Pi remains an unmodified black-box Harness.  This orchestrator owns the
observability boundary around it: controlled-model evidence, workspace change
evidence, isolated verifier output and the canonical Episode/ExecutionBundle.
It is deliberately host-local; the task worktree is the sandbox boundary for
the current single-machine MVP.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import signal
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections.abc import Callable

from src.assembly import finalize_local_run
from src.capture import (
    ArtifactStore,
    EnvironmentCapture,
    EventWriter,
    ModelEndpointKind,
    ModelCallEvidence,
    ModelEvidenceCapability,
    ModelEvidenceJsonlWriter,
    ModelProxyHttpServer,
    ModelProxyService,
    PiJsonAdapter,
    PiOutcomeDeclaration,
    PiRunConfig,
    TraceRecorder,
    read_pi_ndjson,
    snapshot_workspace,
    store_workspace_change_evidence,
)
from src.contracts._json import canonical_json_bytes, sha256_json
from src.contracts.agent_episode import (
    CaptureCapability,
    EpisodeVerifierStatus,
    ExecutionValidity,
    TaskStatus,
)
from src.contracts.execution_identity import ExecutionIdentity
from src.contracts.manifests import EnvironmentManifest, EvaluatorManifest, HarnessManifest, ModelManifest
from src.contracts.run_manifest import ExecutionRunManifest
from src.contracts.trace_event import EventStatus, EventType
from src.contracts.verifier_report import LocalVerifierReport, VerifierExecutionStatus
from src.errors import ContractValidationError
from src.producers import ProducerArtifact, ProducerCapability, ProducerExecutionStatus


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    temporary.replace(path)


def _models_config(*, base_url: str, model: str, max_tokens: int = 8192) -> dict[str, object]:
    return {"providers": {"local-qwen-proxy": {
        "baseUrl": f"{base_url}/v1", "api": "openai-completions",
        "apiKey": "$AGENT_MODEL_PROXY_API_KEY",
        "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False,
                   "supportsUsageInStreaming": True, "supportsStore": False,
                   "maxTokensField": "max_tokens", "supportsStrictMode": False},
                   "models": [{"id": model, "name": "controlled data-plane model", "reasoning": False,
                    "input": ["text"], "contextWindow": 32768, "maxTokens": max_tokens,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}],
    }}}


@dataclass(frozen=True, slots=True, kw_only=True)
class PiHostExecutionSpec:
    identity: ExecutionIdentity
    pi: str
    workspace: str
    prompt: str
    tools: tuple[str, ...]
    timeout_seconds: float
    upstream_url: str | None
    provider: str
    provider_api: str
    model: str
    model_revision: str
    harness_manifest: HarnessManifest
    evaluator_manifest: EvaluatorManifest
    verifier_command: tuple[str, ...]
    task_snapshot: str
    experiment_manifest_ref: str
    sampling_config: dict[str, Any]
    require_rl_evidence: bool = True
    # verl closeout injection points (all optional; defaults preserve behavior).
    model_bridge_url: str | None = None
    verified_policy_fingerprint: str | None = None
    injected_training_sequence: dict[str, Any] | None = None
    # Upper bound on tokens Pi may ask a single generation for. The caller that
    # knows the rollout's response budget sets this; None keeps the historical
    # 8192, which exceeds every training window this project uses.
    max_tokens_per_generation: int | None = None
    # Immutable identity of whatever actually served the model calls. Stock
    # vLLM responses carry no revision field, so the caller that resolved the
    # engine's served model supplies it; without it the evidence cannot claim
    # POLICY_VERSION and every episode is rejected.
    backend_model_revision: str | None = None
    upstream_transport: Callable[[dict[str, Any]], tuple[int, dict[str, Any]]] | None = None
    isolate_pi: bool = False
    training_sequence_builder: Callable[[tuple[ModelCallEvidence, ...]], dict[str, Any]] | None = None


class PiHostExecutionOrchestrator:
    """Run one immutable Pi attempt and fail closed on missing evidence."""

    def run(self, spec: PiHostExecutionSpec, *, output_dir: str | Path) -> dict[str, Any]:
        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=False)
        events_path = root / "raw-events.jsonl"
        evidence_path = root / "model-evidence.jsonl"
        artifacts_path = root / "artifact-refs.json"
        store = ArtifactStore(root / "objects")
        recorder = TraceRecorder(EventWriter(events_path), run_id=spec.identity.run_id,
                                 episode_id=spec.identity.episode_id, trace_id=f"trace-{spec.identity.run_id}")
        environment = EnvironmentCapture(recorder)
        controlled_model = spec.upstream_url is not None
        # Optional bridge override: route Pi's model traffic through the
        # verl-managed endpoint while keeping the same evidence capture path.
        # Default (None) preserves the existing upstream behavior.
        bridge_url = spec.model_bridge_url or spec.upstream_url
        plan = {"kind": "pi-host", "pi": spec.pi, "workspace": str(Path(spec.workspace).resolve()),
                "tools": list(spec.tools), "timeout_seconds": spec.timeout_seconds,
                "model_endpoint": "controlled-proxy" if controlled_model else "provider-managed",
                "provider": spec.provider, "provider_api": spec.provider_api, "model": spec.model,
                "model_bridge_configured": bridge_url is not None and bridge_url != spec.upstream_url}
        plan_checksum = sha256_json(plan)
        manifest = ExecutionRunManifest(
            identity=spec.identity, harness_manifest=spec.harness_manifest,
            model_manifest=ModelManifest(provider=spec.provider, model_id=spec.model,
                                         revision=spec.model_revision, sampling_config=spec.sampling_config),
            environment_manifest=EnvironmentManifest(runtime_type="host-worktree", revision="pi-host/v1",
                resource_limits={"timeout_seconds": spec.timeout_seconds}, network_policy="host-local",
                task_snapshot=spec.task_snapshot), evaluator_manifest=spec.evaluator_manifest,
            experiment_manifest_ref=spec.experiment_manifest_ref,
            capture_capabilities=frozenset({CaptureCapability.MODEL_IO, CaptureCapability.MODEL_TOKEN_USAGE,
                CaptureCapability.TOOL_IO, CaptureCapability.SANDBOX_LIFECYCLE,
                CaptureCapability.FILE_ARTIFACTS, CaptureCapability.VERIFIER_EVIDENCE}
                | ({CaptureCapability.MODEL_TOKEN_IDS, CaptureCapability.MODEL_LOGPROBS}
                   if controlled_model else set())),
            launcher_plan_checksum=plan_checksum, created_at=_now())
        _write_json(root / "execution-run-manifest.json", manifest.to_dict())
        _write_json(root / "launch-plan.json", plan | {"checksum": plan_checksum})
        before = snapshot_workspace(spec.workspace)
        sandbox_span = "span-pi-host"
        environment.sandbox_started(span_id=sandbox_span, runtime_id=str(Path(spec.workspace).resolve()),
                                    attributes={"launcher_plan_checksum": plan_checksum, "harness": "pi"})
        server: ModelProxyHttpServer | None = None
        pi_home: Path | None = None
        try:
            child_env = os.environ | {"PATH": f"{Path(spec.pi).parent}:{os.environ.get('PATH', '')}"}
            provider = spec.provider
            if controlled_model:
                token = secrets.token_urlsafe(32)
                server = ModelProxyHttpServer(ModelProxyService(identity=spec.identity,
                    endpoint_kind=ModelEndpointKind.CONTROLLED,
                    upstream_chat_completions_url=bridge_url,
                    upstream_authorization=os.environ.get("AGENT_UPSTREAM_AUTHORIZATION"),
                    evidence_writer=ModelEvidenceJsonlWriter(evidence_path), recorder=recorder,
                    access_token=token, timeout_seconds=spec.timeout_seconds,
                    backend_model_revision=spec.backend_model_revision,
                    upstream_transport=spec.upstream_transport,
                    # A budget closeout is not a model call, so it leaves no
                    # evidence row; without this note an episode cut off by
                    # budget would be indistinguishable from one the policy
                    # chose to finish.
                    closeout_note_path=root / "engine-closeout.jsonl",
                    max_calls=int(os.environ.get("AGENT_MODEL_MAX_CALLS", "128"))), host="127.0.0.1", port=0)
                server.start_in_thread()
                pi_home = Path(tempfile.mkdtemp(prefix="pi-certified-"))
                config_path = pi_home / ".pi" / "agent" / "models.json"
                config_path.parent.mkdir(parents=True)
                config_path.write_text(json.dumps(_models_config(
                    base_url=f"http://127.0.0.1:{server.address[1]}", model=spec.model,
                    max_tokens=spec.max_tokens_per_generation or 8192,
                )))
                if spec.isolate_pi:
                    (config_path.parent / "settings.json").write_text(json.dumps({
                        "compaction": {"enabled": False}, "retry": {"enabled": False},
                    }))
                provider = "local-qwen-proxy"
                child_env |= {"HOME": str(pi_home), "AGENT_MODEL_PROXY_API_KEY": token}
            command = [spec.pi, "--provider", provider, "--model", spec.model, "--mode", "json", "--print",
                       "--no-session", "--no-context-files", "--no-extensions", "--no-skills", "--tools", ",".join(spec.tools),
                       "--thinking", "minimal", spec.prompt]
            if spec.isolate_pi:
                if pi_home is None:
                    raise ContractValidationError("isolated Pi requires an execution-bound proxy")
                from src.integrations.verl.sandbox import pi_bwrap_command
                command = pi_bwrap_command(command, workspace=spec.workspace, home=pi_home,
                                          pi=spec.pi, proxy_token=token)
                result = _run_owned_process(command, cwd=spec.workspace, env=child_env,
                                            timeout=spec.timeout_seconds)
            else:
                result = subprocess.run(command, cwd=spec.workspace, env=child_env,
                    stdin=subprocess.DEVNULL, capture_output=True, text=True,
                    timeout=spec.timeout_seconds, check=False)
        except subprocess.TimeoutExpired as exc:
            result = subprocess.CompletedProcess(command if 'command' in locals() else [spec.pi], 124,
                                                  _subprocess_text(exc.stdout),
                                                  _subprocess_text(exc.stderr) or "timeout")
        finally:
            if server is not None:
                server.close()
            if pi_home is not None:
                shutil.rmtree(pi_home, ignore_errors=True)
        raw_ref = store.put(result.stdout.encode(), kind="pi-ndjson", media_type="application/x-ndjson", created_at=_now())
        stderr_ref = store.put(result.stderr.encode(), kind="pi-stderr", media_type="text/plain; charset=utf-8", created_at=_now())
        records, issues = read_pi_ndjson(result.stdout)
        evidence_rows = [line for line in evidence_path.read_text().splitlines() if line] if evidence_path.exists() else []
        evidence = tuple(ModelCallEvidence.from_dict(json.loads(line)) for line in evidence_rows)
        rl_fields = frozenset({
            ModelEvidenceCapability.TOKEN_IDS,
            ModelEvidenceCapability.BEHAVIOR_LOGPROBS,
            ModelEvidenceCapability.POLICY_VERSION,
        })
        all_model_calls_usable = bool(evidence) and all(
            item.backend.status_code < 400 and rl_fields.issubset(item.capabilities)
            for item in evidence
        )
        evidence_ref = store.put(evidence_path.read_bytes() if evidence_path.exists() else b"", kind="model-evidence",
                                 media_type="application/x-ndjson", created_at=_now())
        after = snapshot_workspace(spec.workspace)
        workspace_refs = store_workspace_change_evidence(before, after, store=store, created_at=_now())
        adapter_config = PiRunConfig(provider=spec.provider, model=spec.model,
            tools=spec.tools, api=spec.provider_api)
        declaration = PiOutcomeDeclaration(task_status=TaskStatus.UNKNOWN, execution_validity=ExecutionValidity.UNKNOWN,
            verifier_status=EpisodeVerifierStatus.NOT_RUN)
        adapted = PiJsonAdapter().convert(records, run_id=spec.identity.run_id,
            episode_id=spec.identity.episode_id, trace_id=f"trace-{spec.identity.run_id}",
            config=adapter_config, declared_outcome=declaration, source_issues=issues)
        # A transient malformed upstream response can happen after the agent
        # has already written and tested a solution. Preserve that observable
        # work so the external verifier can decide the benchmark outcome.
        has_tool_activity = any(
            event.event_type is EventType.TOOL_RESULT for event in adapted.events
        )
        observable_direct_run = bool(records) and (
            not adapted.backend_error_messages or has_tool_activity
        )
        harness_status = ProducerExecutionStatus.COMPLETED if result.returncode == 0 and (
            (all_model_calls_usable if spec.require_rl_evidence else observable_direct_run)
            if controlled_model else observable_direct_run
        ) else (
            ProducerExecutionStatus.TIMEOUT if result.returncode == 124 else ProducerExecutionStatus.INFRA_INVALID)
        all_harness_refs = (raw_ref, stderr_ref, evidence_ref, *workspace_refs)
        environment.sandbox_finished(span_id=sandbox_span, runtime_id=str(Path(spec.workspace).resolve()),
            status={ProducerExecutionStatus.COMPLETED: EventStatus.SUCCEEDED, ProducerExecutionStatus.TIMEOUT: EventStatus.TIMEOUT,
                    ProducerExecutionStatus.INFRA_INVALID: EventStatus.ERROR}[harness_status],
            artifact_refs=tuple(item.artifact_id for item in all_harness_refs))
        # Preserve tool facts from Pi, while model facts remain authoritative from the proxy.
        for event in adapted.events:
            if event.event_type in {
                EventType.EPISODE_FINISHED,
                EventType.VERIFICATION_STARTED,
                EventType.VERIFICATION_FINISHED,
            }:
                continue
            if controlled_model and event.event_type in {EventType.MODEL_REQUEST, EventType.MODEL_RESPONSE}:
                continue
            recorder.emit(event.event_type, event.component, event.status, span_id=f"pi-{event.span_id}",
                          parent_span_id=None, attributes=dict(event.attributes), artifact_refs=event.artifact_refs,
                          attempt=event.attempt)
        verifier_output = root / "verifier-output.json"
        verifier_status = EpisodeVerifierStatus.ERROR
        verifier_exec = VerifierExecutionStatus.INFRA_INVALID
        verifier_event_status = EventStatus.ERROR
        score: float | None = None
        verifier_refs = ()
        verifier_artifact: ProducerArtifact | None = None
        verifier_event_id: str | None = None
        if harness_status is ProducerExecutionStatus.COMPLETED:
            span = "span-swebench-verifier"
            environment.verification_started(span_id=span, verifier=spec.evaluator_manifest.name)
            verify = subprocess.run(spec.verifier_command, capture_output=True, text=True, timeout=930, check=False)
            if not verifier_output.exists():
                verifier_output.write_text(json.dumps({"verifier_process_stdout": verify.stdout[-8000:], "verifier_process_stderr": verify.stderr[-8000:]}))
            verifier_ref = store.put(verifier_output.read_bytes(), kind="verifier-output", media_type="application/json", created_at=_now())
            verifier_refs = (verifier_ref,)
            try:
                verdict = json.loads(verifier_output.read_text())
                resolved = verdict.get("resolved")
                if isinstance(resolved, bool):
                    verifier_status = EpisodeVerifierStatus.PASSED if resolved else EpisodeVerifierStatus.FAILED
                    verifier_exec = VerifierExecutionStatus.COMPLETED if verify.returncode in {0, 1} else VerifierExecutionStatus.FAILED
                    verifier_event_status = EventStatus.SUCCEEDED if resolved else EventStatus.FAILED
                    score = 1.0 if resolved else 0.0
                elif verdict.get("execution_validity") == "INFRA_INVALID":
                    verifier_status = EpisodeVerifierStatus.ERROR
                    verifier_exec = VerifierExecutionStatus.INFRA_INVALID
            except (OSError, json.JSONDecodeError):
                pass
            verifier_artifact = ProducerArtifact(identity=spec.identity,
                status=ProducerExecutionStatus.COMPLETED if verifier_exec is VerifierExecutionStatus.COMPLETED else ProducerExecutionStatus.INFRA_INVALID,
                capabilities=frozenset({ProducerCapability.VERIFIER_EVIDENCE}),
                payload={"output_artifacts": [item.to_dict() for item in verifier_refs], "command": list(spec.verifier_command)})
            report = LocalVerifierReport(identity_checksum=spec.identity.checksum, evaluator=spec.evaluator_manifest,
                command=spec.verifier_command, execution_status=verifier_exec, verifier_status=verifier_status, score=score,
                producer_artifact_checksum=verifier_artifact.checksum, output_artifact_checksums=tuple(item.checksum for item in verifier_refs), created_at=_now())
            report_ref = store.put(canonical_json_bytes(report.to_dict()), kind="verifier-report", media_type="application/json", created_at=_now())
            verifier_refs = (*verifier_refs, report_ref)
            finished = environment.verification_finished(span_id=span, verifier=spec.evaluator_manifest.name,
                passed=True if verifier_status is EpisodeVerifierStatus.PASSED else False if verifier_status is EpisodeVerifierStatus.FAILED else None,
                status=verifier_event_status, score=score, artifact_refs=tuple(item.artifact_id for item in verifier_refs))
            verifier_event_id = finished.event_id
        else:
            report_ref = None
        valid = harness_status is ProducerExecutionStatus.COMPLETED and verifier_status in {EpisodeVerifierStatus.PASSED, EpisodeVerifierStatus.FAILED}
        recorder.finish(span_id="span-episode", task_status="SUCCESS" if verifier_status is EpisodeVerifierStatus.PASSED else "FAILURE" if verifier_status is EpisodeVerifierStatus.FAILED else "UNKNOWN",
            execution_validity="VALID" if valid else "INFRA_INVALID", verifier_status=verifier_status.value,
            termination_reason="VERIFIER_PASSED" if verifier_status is EpisodeVerifierStatus.PASSED else "VERIFIER_FAILED" if verifier_status is EpisodeVerifierStatus.FAILED else "CAPTURE_OR_VERIFIER_INVALID",
            score=score, evidence_event_ids=(verifier_event_id,) if verifier_event_id else ())
        capabilities = {ProducerCapability.RAW_HARNESS_TRACE, ProducerCapability.HARNESS_EVENTS}
        if all_model_calls_usable:
            capabilities |= {ProducerCapability.TOKEN_IDS, ProducerCapability.BEHAVIOR_LOGPROBS, ProducerCapability.POLICY_VERSION}
        if report_ref:
            capabilities.add(ProducerCapability.VERIFIER_EVIDENCE)
        producer = ProducerArtifact(identity=spec.identity, status=ProducerExecutionStatus.COMPLETED if valid else harness_status,
            capabilities=frozenset(capabilities), payload={"launch_plan_checksum": plan_checksum,
            "output_artifacts": [item.to_dict() for item in (*all_harness_refs, *verifier_refs)],
            "verifier_artifact_checksum": verifier_artifact.checksum if verifier_artifact else None,
            "verifier_report_checksum": report_ref.sha256 if report_ref else None,
            "pi_returncode": result.returncode, "pi_parse_issue_count": len(issues), "model_call_count": len(evidence_rows),
            "all_model_calls_rl_usable": all_model_calls_usable,
            "model_endpoint": "controlled-proxy" if controlled_model else "provider-managed"},
            issues=tuple(issue.message for issue in issues) + tuple(adapted.backend_error_messages) + before.issues + after.issues)
        # Keep policy evidence distinct from the harness artifact.  The terminal
        # verifier reward is attached only after a valid verifier run; failed
        # tasks therefore carry an observed 0.0, while infra-invalid attempts
        # carry no invented reward at all.
        policy_traces = []
        if valid and all_model_calls_usable and score is not None:
            for item in evidence:
                policy_traces.append({
                    "request_id": item.request.request_id,
                    "response_ids": list(item.backend.response_token_ids or ()),
                    "action_mask": [1] * len(item.backend.response_token_ids or ()),
                    "response_logprobs": list(item.backend.response_logprobs or ()),
                    "reward": score,
                    "backend_model_revision": item.backend.backend_model_revision,
                })
        policy_capabilities = set()
        if policy_traces:
            policy_capabilities |= {
                ProducerCapability.TOKEN_IDS,
                ProducerCapability.ACTION_MASK,
                ProducerCapability.BEHAVIOR_LOGPROBS,
                ProducerCapability.POLICY_VERSION,
            }
        if report_ref:
            policy_capabilities.add(ProducerCapability.VERIFIER_EVIDENCE)
        training_payload = {}
        if policy_traces and spec.training_sequence_builder is not None:
            training_payload["training_sequence"] = spec.training_sequence_builder(evidence)
        policy_artifact = ProducerArtifact(
            identity=spec.identity,
            status=ProducerExecutionStatus.COMPLETED if policy_traces else ProducerExecutionStatus.INFRA_INVALID,
            capabilities=frozenset(policy_capabilities),
            payload={"trajectory": {"traces": policy_traces},
                     "model_evidence_checksums": [item.checksum for item in evidence],
                     "verifier_report_checksum": report_ref.sha256 if report_ref else None,
                     **training_payload},
            issues=() if policy_traces else ("RL policy evidence is incomplete or lacks a valid verifier reward",),
        )
        refs = (*all_harness_refs, *verifier_refs)
        _write_json(root / "producer-artifact.json", producer.to_dict())
        _write_json(root / "policy-artifact.json", policy_artifact.to_dict())
        if spec.verified_policy_fingerprint is not None:
            # Fail closed when the caller pins a round policy: the executed
            # episode must carry the same behavior fingerprint.
            if spec.identity.policy_fingerprint != spec.verified_policy_fingerprint:
                raise ContractValidationError(
                    "executed episode policy fingerprint does not match "
                    "the verified round policy"
                )
        if spec.injected_training_sequence is not None:
            # A pre-verified training sequence may be attached for audit; it
            # never replaces the evidence-derived policy artifact above.
            _write_json(
                root / "injected-training-sequence.json",
                dict(spec.injected_training_sequence),
            )
        _write_json(artifacts_path, [item.to_dict() for item in refs])
        finalization = finalize_local_run(manifest=manifest, producer_artifact=producer, events_path=events_path,
            policy_artifact=policy_artifact, model_evidence_path=evidence_path, artifacts_path=artifacts_path)
        _write_json(root / "finalized" / "episode.json", finalization.episode.to_dict())
        _write_json(root / "finalized" / "execution-bundle.json", finalization.execution_bundle.to_dict())
        from src.certification import ConsumerProfile, certify_for
        sft_decision = certify_for(finalization.episode, ConsumerProfile.SFT)
        _write_json(root / "finalized" / "sft-eligibility.json", sft_decision.to_dict())
        rl_decision = certify_for(
            finalization.episode, ConsumerProfile.ON_POLICY_RL,
            execution_bundle=finalization.execution_bundle, policy_artifact=policy_artifact,
            target_policy_fingerprint=spec.identity.policy_fingerprint,
        )
        _write_json(root / "finalized" / "on-policy-rl-eligibility.json", rl_decision.to_dict())
        summary = {"run_id": spec.identity.run_id, "task_id": spec.identity.task_id,
                   "episode_id": spec.identity.episode_id,
                   "policy_artifact_checksum": policy_artifact.checksum,
                   "pi_returncode": result.returncode,
                   "model_call_count": len(evidence_rows) if controlled_model else sum(
                       event.event_type is EventType.MODEL_RESPONSE for event in adapted.events
                   ), "verifier_status": verifier_status.value,
                   "execution_validity": finalization.episode.outcome.execution_validity.value,
                   "integrity": finalization.episode.integrity.state.value,
                   "sft_verdict": sft_decision.verdict.value,
                   "on_policy_rl_verdict": rl_decision.verdict.value,
                   "execution_bundle_checksum": finalization.execution_bundle.checksum}
        _write_json(root / "summary.json", summary)
        return summary


def _subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _run_owned_process(command, *, cwd, env, timeout):
    """Cancel only this subprocess session, including child tool processes."""
    with subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          start_new_session=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
