"""V2 entry point plus the deterministic synthetic mechanism fixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from src.analysis.attribution import AttributionEngine
from src.analysis.compare import compare_runs
from src.analysis.metrics import compute_episode_metrics
from src.analysis.regression_gate import GateConfig, evaluate_gate
from src.assembly.episode_assembler import EpisodeAssembler, EpisodeContext
from src.capture.event_writer import EventJsonlReader, EventWriter
from src.capture.pi_adapter import (
    PiJsonAdapter,
    PiOutcomeDeclaration,
    PiRunConfig,
    read_pi_ndjson,
)
from src.capture.recorder import EnvironmentCapture, HarnessHook, TraceRecorder
from src.contracts._json import sha256_json
from src.contracts.agent_episode import (
    CaptureCapability,
    EpisodeVerifierStatus,
    ExecutionValidity,
    TaskStatus,
)
from src.contracts.experiment import ExperimentManifest
from src.contracts.manifests import (
    EnvironmentManifest,
    EvaluatorManifest,
    HarnessManifest,
    ModelManifest,
)
from src.contracts.training_candidate import build_training_candidate_view
from src.contracts.trace_event import EventStatus
from src.observatory.report import render_observatory_html


def _clock(base: int) -> Callable[[], str]:
    counter = base

    def now() -> str:
        nonlocal counter
        value = f"2026-08-17T08:00:{counter:02d}Z"
        counter += 1
        return value

    return now


def _ids(prefix: str) -> Callable[[], str]:
    index = 0

    def value() -> str:
        nonlocal index
        index += 1
        return f"{prefix}-{index:03d}"

    return value


def _context(*, task_id: str, variant: str, experiment_ref: str) -> EpisodeContext:
    policy = {
        "structured_tool_errors": variant == "candidate",
        "retry_after_tool_error": "inspect_then_retry" if variant == "candidate" else "retry_unchanged",
    }
    return EpisodeContext(
        task_id=task_id,
        attempt=1,
        harness_manifest=HarnessManifest(
            name="pi",
            version=f"0.84.2-{variant}",
            revision=f"pi-policy-{variant}-r1",
            config_digest=sha256_json(policy),
            policy_flags=policy,
            hook_version="reference-policy-hook/v1",
        ),
        model_manifest=ModelManifest(
            provider="opencode-go",
            model_id="gpt-5.6-luna",
            revision="NOT_OBSERVABLE",
            sampling_config={"thinking": "minimal", "seed": 7},
            tokenizer_revision=None,
        ),
        environment_manifest=EnvironmentManifest(
            runtime_type="reference-process",
            revision="reference-env/v2",
            image=None,
            resource_limits={"timeout_seconds": 30},
            network_policy="disabled",
            task_snapshot="structured-tool-error-suite/r1",
        ),
        evaluator_manifest=EvaluatorManifest(
            name="exact-json-verifier",
            revision="exact-json-verifier/v1",
            config_digest=sha256_json({"strict": True}),
        ),
        experiment_manifest_ref=experiment_ref,
        capabilities=frozenset(
            {
                CaptureCapability.MODEL_IO,
                CaptureCapability.MODEL_TOKEN_USAGE,
                CaptureCapability.TOOL_IO,
                CaptureCapability.VERIFIER_EVIDENCE,
                CaptureCapability.HARNESS_DECISIONS,
                CaptureCapability.RETRY_DECISIONS,
                CaptureCapability.TERMINATION_DECISIONS,
            }
        ),
    )


def _emit_pair_episode(
    writer: EventWriter,
    *,
    task_index: int,
    variant: str,
    experiment_ref: str,
) -> EpisodeContext:
    run_id = f"run-v2-{variant}"
    episode_id = f"episode-v2-{variant}-{task_index}"
    recorder = TraceRecorder(
        writer,
        run_id=run_id,
        episode_id=episode_id,
        trace_id=f"trace-v2-{variant}-{task_index}",
        clock=_clock(task_index * 15 + (0 if variant == "control" else 1)),
        id_factory=_ids(episode_id),
    )
    hook = HarnessHook(recorder)
    environment = EnvironmentCapture(recorder)
    first_model = f"span-{episode_id}-model-1"
    recorder.model_request(
        span_id=first_model,
        parent_span_id=None,
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": f"read missing-{task_index}.json then recover"}],
        tools=[{"name": "read"}, {"name": "find"}],
    )
    recorder.model_response(
        span_id=first_model,
        parent_span_id=None,
        response={"tool_call": "read", "path": f"missing-{task_index}.json"},
        usage={"input_tokens": 100 + task_index, "output_tokens": 20},
        latency_ms=80,
    )
    first_args = {"path": f"missing-{task_index}.json"}
    first_tool = f"span-{episode_id}-tool-1"
    recorder.tool_call(
        span_id=first_tool,
        parent_span_id=first_model,
        tool_name="read",
        arguments=first_args,
    )
    recorder.tool_result(
        span_id=first_tool,
        parent_span_id=first_model,
        tool_name="read",
        result={
            "error_type": "FILE_NOT_FOUND" if variant == "candidate" else None,
            "stderr": f"missing-{task_index}.json: not found",
            "retryable": variant == "candidate",
            "feedback_format": "structured" if variant == "candidate" else "raw_stderr",
        },
        status=EventStatus.FAILED,
        latency_ms=6,
    )
    hook.emit_decision(
        span_id=f"span-{episode_id}-decision",
        parent_span_id=first_model,
        decision="INSPECT_BEFORE_RETRY" if variant == "candidate" else "RETRY_UNCHANGED",
        reason_code="STRUCTURED_FILE_NOT_FOUND" if variant == "candidate" else "GENERIC_RETRY",
    )
    second_model = f"span-{episode_id}-model-2"
    recorder.model_request(
        span_id=second_model,
        parent_span_id=None,
        model="gpt-5.6-luna",
        messages=[{"role": "tool", "content": "structured error" if variant == "candidate" else "raw stderr"}],
        tools=[{"name": "read"}, {"name": "find"}],
        attempt=2,
    )
    recorder.model_response(
        span_id=second_model,
        parent_span_id=None,
        response={
            "tool_call": "read",
            "path": f"task-{task_index}.json" if variant == "candidate" else f"missing-{task_index}.json",
        },
        usage={
            "input_tokens": (72 if variant == "candidate" else 82) + task_index,
            "output_tokens": 18,
        },
        latency_ms=75 if variant == "candidate" else 82,
        attempt=2,
    )
    second_args = (
        {"path": f"task-{task_index}.json"} if variant == "candidate" else first_args
    )
    second_tool = f"span-{episode_id}-tool-2"
    recorder.tool_call(
        span_id=second_tool,
        parent_span_id=second_model,
        tool_name="read",
        arguments=second_args,
        attempt=2,
    )
    succeeded = variant == "candidate"
    recorder.tool_result(
        span_id=second_tool,
        parent_span_id=second_model,
        tool_name="read",
        result={"records": 4} if succeeded else {"stderr": "not found"},
        status=EventStatus.SUCCEEDED if succeeded else EventStatus.FAILED,
        latency_ms=5,
        attempt=2,
    )
    verifier_span = f"span-{episode_id}-verifier"
    environment.verification_started(span_id=verifier_span, verifier="exact-json-verifier")
    verifier = environment.verification_finished(
        span_id=verifier_span,
        verifier="exact-json-verifier",
        passed=succeeded,
        status=EventStatus.SUCCEEDED if succeeded else EventStatus.FAILED,
        score=1.0 if succeeded else 0.0,
    )
    hook.emit_termination_decided(
        span_id=f"span-{episode_id}-termination",
        reason="VERIFIER_PASSED" if succeeded else "RETRY_BUDGET_EXHAUSTED",
        verifier_observed=True,
    )
    recorder.finish(
        span_id=f"span-{episode_id}-episode",
        task_status="SUCCESS" if succeeded else "FAILURE",
        execution_validity="VALID",
        verifier_status="PASSED" if succeeded else "FAILED",
        termination_reason="VERIFIER_PASSED" if succeeded else "RETRY_BUDGET_EXHAUSTED",
        score=1.0 if succeeded else 0.0,
        evidence_event_ids=(verifier.event_id,),
    )
    return _context(
        task_id=f"structured-error/task-{task_index}",
        variant=variant,
        experiment_ref=experiment_ref,
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in values)
    )


def generate_synthetic_v2_demo(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    experiment = ExperimentManifest(
        experiment_id="experiment-structured-tool-errors-v2",
        revision="r1",
        task_dataset_revision="structured-tool-error-suite/r1",
        control_run_id="run-v2-control",
        candidate_run_id="run-v2-candidate",
        target_policy_flag="TOOL_ERROR_FEEDBACK_LOSS",
        minimum_pairs=3,
        metadata={
            "model_policy": "single fixed remote model; no fallback",
            "reference_case": "raw stderr vs structured tool error feedback",
        },
    )
    raw_path = output / "raw-events.jsonl"
    if raw_path.exists():
        raw_path.unlink()
    writer = EventWriter(raw_path)
    contexts: dict[str, EpisodeContext] = {}
    for task_index in range(1, 4):
        for variant in ("control", "candidate"):
            context = _emit_pair_episode(
                writer,
                task_index=task_index,
                variant=variant,
                experiment_ref=experiment.experiment_id,
            )
            contexts[f"episode-v2-{variant}-{task_index}"] = context
    read = EventJsonlReader.read(raw_path)
    assembly = EpisodeAssembler().assemble(read.events, contexts=contexts)
    controls = tuple(item for item in assembly.episodes if item.run_id == "run-v2-control")
    candidates = tuple(item for item in assembly.episodes if item.run_id == "run-v2-candidate")
    control_metrics = tuple(compute_episode_metrics(item) for item in controls)
    candidate_metrics = tuple(compute_episode_metrics(item) for item in candidates)
    control_reports = tuple(AttributionEngine().analyze(item) for item in controls)
    candidate_reports = tuple(AttributionEngine().analyze(item) for item in candidates)
    control_reasons = {
        report.episode_id: tuple(item.reason_code for item in report.diagnoses)
        for report in control_reports
    }
    candidate_reasons = {
        report.episode_id: tuple(item.reason_code for item in report.diagnoses)
        for report in candidate_reports
    }
    comparison = compare_runs(
        controls,
        candidates,
        control_metrics=control_metrics,
        candidate_metrics=candidate_metrics,
        experiment=experiment,
        control_reason_codes=control_reasons,
        candidate_reason_codes=candidate_reasons,
    )
    gate_config = GateConfig()
    gate = evaluate_gate(comparison, gate_config)
    training_views = tuple(
        build_training_candidate_view(item, target_policy_model="local-student-model")
        for item in candidates
    )

    fixture_path = Path(__file__).parents[1] / "tests" / "fixtures" / "pi" / "gpt56_luna_probe.ndjson"
    pi_records, pi_issues = read_pi_ndjson(fixture_path.read_text())
    pi_adapter = PiJsonAdapter().convert(
        pi_records,
        run_id="run-pi-real-probe",
        episode_id="episode-pi-real-probe",
        trace_id="trace-pi-real-probe",
        config=PiRunConfig(),
        declared_outcome=PiOutcomeDeclaration(
            task_status=TaskStatus.SUCCESS,
            execution_validity=ExecutionValidity.VALID,
            verifier_status=EpisodeVerifierStatus.PASSED,
            score=1.0,
        ),
        source_issues=pi_issues,
    )

    _write_json(output / "experiment-manifest.json", experiment.to_dict())
    _write_jsonl(output / "episodes-control.jsonl", [item.to_dict() for item in controls])
    _write_jsonl(output / "episodes-candidate.jsonl", [item.to_dict() for item in candidates])
    _write_json(output / "metrics-control.json", [item.to_dict() for item in control_metrics])
    _write_json(output / "metrics-candidate.json", [item.to_dict() for item in candidate_metrics])
    _write_jsonl(output / "diagnoses-control.jsonl", [item.to_dict() for item in control_reports])
    _write_jsonl(output / "diagnoses-candidate.jsonl", [item.to_dict() for item in candidate_reports])
    _write_json(output / "comparison.json", comparison.to_dict())
    _write_json(output / "gate-config.json", gate_config.to_dict())
    _write_json(output / "gate-result.json", gate.to_dict())
    _write_json(output / "training-candidates.json", [item.to_dict() for item in training_views])
    _write_json(output / "pi-real-probe-conversion.json", pi_adapter.to_dict())
    (output / "observatory.html").write_text(
        render_observatory_html(
            episodes=assembly.episodes,
            metrics=control_metrics + candidate_metrics,
            diagnoses=[item.to_dict() for item in control_reports + candidate_reports],
            comparison=comparison,
            gate=gate,
        )
    )
    summary = {
        "release": "v2",
        "scope": "real Pi adapter + controlled paired Harness decision loop",
        "pi": PiRunConfig().to_dict(),
        "real_pi_probe_source_checksum": pi_adapter.source_checksum,
        "control_episode_count": len(controls),
        "candidate_episode_count": len(candidates),
        "paired_coverage": comparison.paired_coverage,
        "compatibility_mismatch_count": len(comparison.compatibility_mismatches),
        "gate_decision": gate.decision.value,
        "target_slice": {
            "reason_code": experiment.target_policy_flag,
            "control": comparison.aggregate.control_target_slice_count,
            "candidate": comparison.aggregate.candidate_target_slice_count,
        },
        "training_semantics": {
            "teacher_model": "opencode-go/gpt-5.6-luna",
            "policy_relation": "OFF_POLICY",
            "sft_candidates": sum(item.sft_candidate for item in training_views),
            "on_policy_rl_candidates": sum(item.on_policy_rl_candidate for item in training_views),
        },
        "evidence_checksums": {
            "experiment": experiment.checksum,
            "comparison": comparison.checksum,
            "gate": gate.checksum,
            "assembly": assembly.output_checksum,
        },
        "claim_boundary": (
            "three deterministic pairs prove the decision mechanism; they do not establish "
            "statistical generalization or a benchmark improvement claim"
        ),
    }
    _write_json(output / "summary.json", summary)
    return summary


def generate_v2_demo(output_dir: str | Path) -> dict[str, Any]:
    """Generate the release V2 package from the captured real Pi pairs."""

    from src.real_pi_v2 import generate_real_pi_v2

    return generate_real_pi_v2(output_dir)
