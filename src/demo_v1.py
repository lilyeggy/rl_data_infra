"""Deterministic V1 evidence generator for the Trace→Episode→Observe slice."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path
from typing import Any, Callable

from src.analysis.attribution import AttributionEngine
from src.analysis.metrics import compute_episode_metrics
from src.assembly.episode_assembler import EpisodeAssembler, EpisodeContext
from src.capture.event_writer import ArtifactStore, EventJsonlReader, EventWriter
from src.capture.recorder import EnvironmentCapture, HarnessHook, TraceRecorder
from src.contracts._json import sha256_json
from src.contracts.agent_episode import CaptureCapability
from src.contracts.manifests import (
    EnvironmentManifest,
    EvaluatorManifest,
    HarnessManifest,
    ModelManifest,
)
from src.contracts.trace_event import EventComponent, EventStatus, EventType


def _clock(start_second: int) -> Callable[[], str]:
    ticks = count(start_second)

    def now() -> str:
        value = datetime(2026, 8, 17, tzinfo=timezone.utc) + timedelta(seconds=next(ticks))
        return value.isoformat().replace("+00:00", "Z")

    return now


def _id_factory(prefix: str) -> Callable[[], str]:
    values = count(1)
    return lambda: f"{prefix}-{next(values):03d}"


def _context(
    *,
    task_id: str,
    capabilities: frozenset[CaptureCapability],
) -> EpisodeContext:
    empty_digest = sha256_json({})
    return EpisodeContext(
        task_id=task_id,
        attempt=1,
        harness_manifest=HarnessManifest(
            name="reference-harness",
            version="v1",
            revision="demo-v1",
            config_digest=sha256_json({"structured_tool_errors": False}),
            policy_flags={"structured_tool_errors": False},
            hook_version="reference-hook/v1",
        ),
        model_manifest=ModelManifest(
            provider="fixture-api",
            model_id="fixed-model",
            revision="fixed-model-r1",
            sampling_config={"temperature": 0, "seed": 7},
            tokenizer_revision="fixed-tokenizer-r1",
        ),
        environment_manifest=EnvironmentManifest(
            runtime_type="reference-process",
            revision="reference-env-r1",
            resource_limits={"timeout_seconds": 30},
            network_policy="disabled",
            task_snapshot="demo-task-set-r1",
        ),
        evaluator_manifest=EvaluatorManifest(
            name="deterministic-reference-verifier",
            revision="reference-verifier-r1",
            config_digest=empty_digest,
        ),
        experiment_manifest_ref="experiment-v1-observability",
        capabilities=capabilities,
    )


def _recorder(writer: EventWriter, episode: str, start_second: int) -> TraceRecorder:
    return TraceRecorder(
        writer,
        run_id="run-v1-observability",
        episode_id=episode,
        trace_id=f"trace-{episode}",
        clock=_clock(start_second),
        id_factory=_id_factory(episode),
    )


def _emit_success(writer: EventWriter) -> EpisodeContext:
    recorder = _recorder(writer, "episode-success", 0)
    environment = EnvironmentCapture(recorder)
    hook = HarnessHook(recorder)
    recorder.model_request(
        span_id="success-model",
        parent_span_id=None,
        model="fixed-model",
        messages=[{"role": "user", "content": "Return the repository status."}],
        tools=[{"name": "shell"}],
    )
    recorder.model_response(
        span_id="success-model",
        parent_span_id=None,
        response={"tool_call": "shell"},
        usage={"input_tokens": 12, "output_tokens": 5},
        latency_ms=80,
    )
    recorder.tool_call(
        span_id="success-tool",
        parent_span_id="success-model",
        tool_name="shell",
        arguments={"cmd": "git status --short"},
    )
    recorder.tool_result(
        span_id="success-tool",
        parent_span_id="success-model",
        tool_name="shell",
        result={"exit_code": 0, "stdout_summary": "clean"},
        status=EventStatus.SUCCEEDED,
        latency_ms=12,
    )
    environment.verification_started(span_id="success-verifier", verifier="exact-match")
    verifier = environment.verification_finished(
        span_id="success-verifier",
        verifier="exact-match",
        passed=True,
        status=EventStatus.SUCCEEDED,
        score=1.0,
    )
    hook.emit_termination_decided(
        span_id="success-termination",
        reason="VERIFIER_PASSED",
        verifier_observed=True,
    )
    recorder.finish(
        span_id="success-episode",
        task_status="SUCCESS",
        execution_validity="VALID",
        verifier_status="PASSED",
        termination_reason="VERIFIER_PASSED",
        score=1.0,
        evidence_event_ids=(verifier.event_id,),
    )
    return _context(
        task_id="demo/success",
        capabilities=frozenset(
            {
                CaptureCapability.MODEL_IO,
                CaptureCapability.MODEL_TOKEN_USAGE,
                CaptureCapability.TOOL_IO,
                CaptureCapability.VERIFIER_EVIDENCE,
                CaptureCapability.HARNESS_DECISIONS,
                CaptureCapability.TERMINATION_DECISIONS,
            }
        ),
    )


def _emit_task_failure(
    writer: EventWriter, artifact_store: ArtifactStore
) -> tuple[EpisodeContext, tuple[Any, ...]]:
    recorder = _recorder(writer, "episode-tool-loop", 20)
    environment = EnvironmentCapture(recorder)
    hook = HarnessHook(recorder)
    recorder.model_request(
        span_id="failure-model",
        parent_span_id=None,
        model="fixed-model",
        messages=[{"role": "user", "content": "Read a missing file and recover."}],
        tools=[{"name": "shell"}],
    )
    recorder.model_response(
        span_id="failure-model",
        parent_span_id=None,
        response={"tool_call": "shell"},
        usage={"input_tokens": 11, "output_tokens": 6},
        latency_ms=85,
    )
    arguments = {"cmd": "cat missing.txt"}
    recorder.tool_call(
        span_id="failure-tool-1",
        parent_span_id="failure-model",
        tool_name="shell",
        arguments=arguments,
    )
    stderr = artifact_store.put(
        b"cat: missing.txt: No such file or directory\n",
        kind="stderr",
        media_type="text/plain",
        created_at="2026-08-17T00:00:23Z",
        producer_event_id="evt-episode-tool-loop-result-1",
    )
    recorder.emit(
        EventType.TOOL_RESULT,
        EventComponent.TOOL,
        EventStatus.FAILED,
        span_id="failure-tool-1",
        parent_span_id="failure-model",
        event_id="evt-episode-tool-loop-result-1",
        attributes={
            "tool_name": "shell",
            "result": {"exit_code": 1, "feedback_format": "raw_stderr"},
            "latency_ms": 9,
        },
        artifact_refs=(stderr.artifact_id,),
    )
    hook.emit_decision(
        span_id="failure-retry-decision",
        parent_span_id="failure-model",
        decision="RETRY_UNCHANGED",
        reason_code="GENERIC_TOOL_RETRY",
    )
    recorder.tool_call(
        span_id="failure-tool-2",
        parent_span_id="failure-model",
        tool_name="shell",
        arguments=arguments,
        attempt=2,
    )
    recorder.tool_result(
        span_id="failure-tool-2",
        parent_span_id="failure-model",
        tool_name="shell",
        result={"exit_code": 1, "feedback_format": "raw_stderr"},
        status=EventStatus.FAILED,
        latency_ms=8,
        attempt=2,
        artifact_refs=(stderr.artifact_id,),
    )
    environment.verification_started(span_id="failure-verifier", verifier="file-exists")
    verifier = environment.verification_finished(
        span_id="failure-verifier",
        verifier="file-exists",
        passed=False,
        status=EventStatus.FAILED,
        score=0.0,
    )
    hook.emit_termination_decided(
        span_id="failure-termination",
        reason="RETRY_BUDGET_EXHAUSTED",
        verifier_observed=True,
    )
    recorder.finish(
        span_id="failure-episode",
        task_status="FAILURE",
        execution_validity="VALID",
        verifier_status="FAILED",
        termination_reason="RETRY_BUDGET_EXHAUSTED",
        score=0.0,
        evidence_event_ids=(verifier.event_id,),
    )
    return (
        _context(
            task_id="demo/tool-error-loop",
            capabilities=frozenset(
                {
                    CaptureCapability.MODEL_IO,
                    CaptureCapability.MODEL_TOKEN_USAGE,
                    CaptureCapability.TOOL_IO,
                    CaptureCapability.FILE_ARTIFACTS,
                    CaptureCapability.VERIFIER_EVIDENCE,
                    CaptureCapability.HARNESS_DECISIONS,
                    CaptureCapability.TERMINATION_DECISIONS,
                }
            ),
        ),
        (stderr,),
    )


def _emit_infra_invalid(writer: EventWriter) -> EpisodeContext:
    recorder = _recorder(writer, "episode-infra-timeout", 40)
    environment = EnvironmentCapture(recorder)
    recorder.model_request(
        span_id="infra-model",
        parent_span_id=None,
        model="fixed-model",
        messages=[{"role": "user", "content": "Run the test suite."}],
    )
    recorder.model_response(
        span_id="infra-model",
        parent_span_id=None,
        response={"tool_call": "shell"},
        usage={"input_tokens": 8, "output_tokens": 4},
        latency_ms=70,
    )
    environment.sandbox_started(span_id="infra-sandbox", runtime_id="sandbox-003")
    timeout = environment.command_result(
        span_id="infra-command",
        command="python3 -m unittest",
        cwd="/workspace",
        exit_code=None,
        status=EventStatus.TIMEOUT,
        latency_ms=30000,
    )
    environment.sandbox_finished(
        span_id="infra-sandbox",
        runtime_id="sandbox-003",
        status=EventStatus.ERROR,
    )
    recorder.finish(
        span_id="infra-episode",
        task_status="UNKNOWN",
        execution_validity="INFRA_INVALID",
        verifier_status="NOT_RUN",
        termination_reason="SANDBOX_TIMEOUT",
        evidence_event_ids=(timeout.event_id,),
    )
    return _context(
        task_id="demo/infra-timeout",
        capabilities=frozenset(
            {
                CaptureCapability.MODEL_IO,
                CaptureCapability.MODEL_TOKEN_USAGE,
                CaptureCapability.SANDBOX_LIFECYCLE,
                CaptureCapability.SANDBOX_COMMAND_IO,
            }
        ),
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for value in values
    )
    path.write_text(payload)


def generate_v1_demo(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    writer = EventWriter(output / "raw-events.jsonl")
    artifact_store = ArtifactStore(output / "blobs", uri_prefix="blobs")

    contexts: dict[str, EpisodeContext] = {}
    contexts["episode-success"] = _emit_success(writer)
    failure_context, failure_artifacts = _emit_task_failure(writer, artifact_store)
    contexts["episode-tool-loop"] = failure_context
    contexts["episode-infra-timeout"] = _emit_infra_invalid(writer)

    read_result = EventJsonlReader.read(output / "raw-events.jsonl")
    assembly = EpisodeAssembler().assemble(
        read_result.events,
        contexts=contexts,
        artifacts=failure_artifacts,
    )
    metrics = [compute_episode_metrics(episode) for episode in assembly.episodes]
    reports = [AttributionEngine().analyze(episode) for episode in assembly.episodes]

    _write_jsonl(output / "episodes.jsonl", [episode.to_dict() for episode in assembly.episodes])
    _write_json(output / "metrics.json", [item.to_dict() for item in metrics])
    _write_jsonl(output / "diagnoses.jsonl", [report.to_dict() for report in reports])
    summary = {
        "release": "v1",
        "scope": "single-run Trace -> Episode -> Metrics/Attribution observability loop",
        "run_id": "run-v1-observability",
        "episode_count": len(assembly.episodes),
        "quarantined_event_count": len(assembly.quarantined_events),
        "reader_issue_count": len(read_result.issues),
        "assembly_input_checksum": assembly.input_checksum,
        "assembly_output_checksum": assembly.output_checksum,
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "task_id": episode.task_id,
                "task_status": episode.outcome.task_status.value,
                "execution_validity": episode.outcome.execution_validity.value,
                "integrity": episode.integrity.state.value,
                "diagnoses": [item.reason_code for item in report.diagnoses],
            }
            for episode, report in zip(assembly.episodes, reports)
        ],
        "known_limits": [
            "V1 does not compare Harness versions or issue a Regression Gate verdict",
            "cost is NOT_OBSERVABLE without a versioned pricing configuration",
            "the reference run is deterministic synthetic evidence, not a benchmark claim",
        ],
    }
    _write_json(output / "summary.json", summary)
    return summary
