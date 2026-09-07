"""HISTORICAL EXAMPLE: local V2.1 decision-aware execution artifact."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from src.analysis.attribution import AttributionEngine
from src.analysis.metrics import compute_episode_metrics
from src.assembly.episode_assembler import EpisodeAssembler, EpisodeContext
from src.capture.event_writer import EventJsonlReader, EventWriter
from src.capture.recorder import EnvironmentCapture, HarnessHook, TraceRecorder
from src.contracts._json import sha256_json
from src.contracts.agent_episode import (
    CaptureCapability,
    EpisodeVerifierStatus,
    ExecutionValidity,
    TaskStatus,
)
from src.contracts.manifests import (
    EnvironmentManifest,
    EvaluatorManifest,
    HarnessManifest,
    ModelManifest,
)
from src.contracts.trace_event import EventComponent, EventStatus, EventType
from src.recovery.controller import RecoveryController
from src.recovery.execution import (
    LocalRecoveryToolExecutor,
    RecoveryExecutionAdapter,
)


TASKS = (1, 2, 3)
RUN_ID = "run-v21-local-recovery"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _context(task_index: int) -> EpisodeContext:
    policy = {
        "policy": "bounded-file-recovery",
        "policy_version": "file-not-found-recovery/v2.1",
        "max_discovery_per_scope": 1,
        "ambiguous_candidate_action": "TERMINATE",
    }
    digest = sha256_json(policy)
    return EpisodeContext(
        task_id=f"v21-local-recovery/task-{task_index}",
        attempt=1,
        harness_manifest=HarnessManifest(
            name="local-recovery-runtime",
            version="v2.1",
            revision="bounded-recovery-runtime/r1",
            config_digest=digest,
            policy_flags=policy,
            hook_version="recovery-hook/v2.1",
        ),
        model_manifest=ModelManifest(
            provider="deterministic-fixture",
            model_id="fixture-policy-driver",
            revision="r1",
            sampling_config={"temperature": 0, "seed": 0},
            tokenizer_revision=None,
        ),
        environment_manifest=EnvironmentManifest(
            runtime_type="local-recovery-runtime",
            revision="v21-local-workspace/r1",
            image=None,
            resource_limits={"max_discovery_per_scope": 1},
            network_policy="disabled",
            task_snapshot="v21-recovery-fixtures/r1",
        ),
        evaluator_manifest=EvaluatorManifest(
            name="exact-file-content-verifier",
            revision="v21-local-verifier/r1",
            config_digest=sha256_json({"expected_task_id": f"task-{task_index}"}),
        ),
        experiment_manifest_ref="experiment-v21-local-recovery",
        capabilities=frozenset(
            {
                CaptureCapability.MODEL_IO,
                CaptureCapability.TOOL_IO,
                CaptureCapability.VERIFIER_EVIDENCE,
                CaptureCapability.HARNESS_DECISIONS,
            }
        ),
    )


def _run_task(
    task_index: int, output: Path, workspace_root: Path
) -> tuple[Any, Any, Any, Any, Any]:
    episode_id = f"episode-v21-local-{task_index}"
    trace_id = f"trace-v21-local-{task_index}"
    ids = iter(f"{task_index}-{index}" for index in range(100))
    recorder = TraceRecorder(
        EventWriter(output / "raw-events.jsonl"),
        run_id=RUN_ID,
        episode_id=episode_id,
        trace_id=trace_id,
        clock=lambda: "2026-08-18T00:00:00Z",
        id_factory=ids.__next__,
    )
    model_span = f"span-model-{task_index}"
    recorder.model_request(
        span_id=model_span,
        parent_span_id=None,
        model="fixture-policy-driver",
        messages=[{"role": "user", "content": "recover file and return JSON"}],
    )
    recorder.model_response(
        span_id=model_span,
        parent_span_id=None,
        response={"stop_reason": "toolUse"},
        usage={"input_tokens": 10, "output_tokens": 4},
    )

    missing_path = f"workspace/missing-{task_index}.json"
    initial_call = recorder.tool_call(
        span_id=f"span-initial-read-{task_index}",
        parent_span_id=model_span,
        tool_name="read",
        arguments={"path": missing_path, "offset": 1, "limit": 2000},
    )
    # The structured error fields are facts from the runtime's tool protocol.
    initial_error = recorder.emit(
        EventType.TOOL_RESULT,
        EventComponent.TOOL,
        EventStatus.ERROR,
        span_id=initial_call.span_id,
        parent_span_id=model_span,
        attributes={
            "tool_name": "read",
            "error_code": "FILE_NOT_FOUND",
            "error_path": missing_path,
            "error_source": "local-recovery-runtime",
            "result": {
                "isError": True,
                "content": [{"type": "text", "text": "FILE_NOT_FOUND"}],
            },
        },
    )

    controller = RecoveryController(
        hook=HarnessHook(recorder),
        initial_discovery_budget=1,
    )
    executor = LocalRecoveryToolExecutor(workspace=workspace_root, recorder=recorder)
    runtime = RecoveryExecutionAdapter(controller=controller, executor=executor)
    report = runtime.recover(initial_error)

    environment = EnvironmentCapture(recorder)
    verifier_span = f"span-verifier-{task_index}"
    environment.verification_started(
        span_id=verifier_span,
        verifier="exact-file-content-verifier",
    )
    verified = environment.verification_finished(
        span_id=verifier_span,
        verifier="exact-file-content-verifier",
        passed=True,
        status=EventStatus.SUCCEEDED,
        score=1.0,
    )
    recorder.finish(
        span_id=f"span-episode-{task_index}",
        task_status=TaskStatus.SUCCESS.value,
        execution_validity=ExecutionValidity.VALID.value,
        verifier_status=EpisodeVerifierStatus.PASSED.value,
        termination_reason="V21_RECOVERY_VERIFIED",
        score=1.0,
        evidence_event_ids=(verified.event_id,),
    )

    events = tuple(
        event
        for event in EventJsonlReader.read(output / "raw-events.jsonl").events
        if event.episode_id == episode_id
    )
    context = _context(task_index)
    assembled = EpisodeAssembler().assemble(events, contexts={episode_id: context})
    if assembled.warnings or assembled.quarantined_events or len(assembled.episodes) != 1:
        raise RuntimeError(
            f"V2.1 local runtime did not assemble cleanly: {assembled.warnings}, "
            f"quarantined={len(assembled.quarantined_events)}"
        )
    episode = assembled.episodes[0]
    metrics = compute_episode_metrics(episode)
    diagnosis = AttributionEngine().analyze(episode)
    return episode, metrics, diagnosis, report, assembled


def generate_v21_runtime_evidence(output_dir: str | Path) -> dict[str, Any]:
    """Generate complete local decision-aware Episodes for V2.1 integration tests."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw-events.jsonl"
    if raw_path.exists():
        raw_path.unlink()

    with tempfile.TemporaryDirectory() as temporary:
        workspace_root = Path(temporary)
        workspace = workspace_root / "workspace"
        workspace.mkdir()
        episodes = []
        metrics = []
        diagnoses = []
        recovery_reports = []
        assembly_checksums = []
        for task_index in TASKS:
            (workspace / f"task-{task_index}.json").write_text(
                json.dumps({"task_id": f"task-{task_index}", "status": "SUCCESS"}) + "\n"
            )
            episode, metric, diagnosis, recovery, assembly = _run_task(
                task_index, output, workspace_root
            )
            episodes.append(episode)
            metrics.append(metric)
            diagnoses.append(diagnosis)
            recovery_reports.append(recovery.to_dict())
            assembly_checksums.append(assembly.output_checksum)

    _write_jsonl(output / "episodes.jsonl", (episode.to_dict() for episode in episodes))
    _write_json(output / "metrics.json", [metric.to_dict() for metric in metrics])
    _write_jsonl(output / "diagnoses.jsonl", (report.to_dict() for report in diagnoses))
    _write_json(output / "recovery-executions.json", recovery_reports)
    summary = {
        "release": "v2.1",
        "scope": "local decision-aware recovery runtime evidence",
        "runtime_type": "local-recovery-runtime",
        "not_real_pi": True,
        "episode_count": len(episodes),
        "complete_episode_count": sum(
            episode.integrity.state.value == "COMPLETE" for episode in episodes
        ),
        "success_count": sum(
            episode.outcome.task_status is TaskStatus.SUCCESS for episode in episodes
        ),
        "harness_decision_event_count": sum(
            sum(event.event_type is EventType.HARNESS_DECISION for event in episode.events)
            for episode in episodes
        ),
        "recovery_action_sequences": [
            [decision["action"] for decision in report["decisions"]]
            for report in recovery_reports
        ],
        "raw_events_append_only": True,
        "assembly_checksums": assembly_checksums,
        "claim_boundary": (
            "local runtime validates controller/executor/Hook integration; it is not "
            "evidence that black-box Pi emitted these decisions"
        ),
    }
    _write_json(output / "summary.json", summary)
    _write_json(
        output / "artifact-manifest.json",
        {
            "schema_version": "v2.1-runtime-artifact-manifest/v1",
            "files": sorted(
                path.name
                for path in output.iterdir()
                if path.is_file() and path.name != "artifact-manifest.json"
            ),
            "summary_checksum": sha256_json(summary),
        },
    )
    return summary


def _write_jsonl(path: Path, values: Any) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        )
    )
