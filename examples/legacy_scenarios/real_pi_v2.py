"""HISTORICAL EXAMPLE: build a V2 package from captured Pi executions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.analysis.attribution import AttributionEngine
from src.analysis.compare import compare_runs
from src.analysis.metrics import compute_episode_metrics
from src.analysis.regression_gate import GateConfig, evaluate_gate
from src.assembly.episode_assembler import EpisodeAssembler, EpisodeContext
from src.capture.pi_adapter import (
    PiAdapterResult,
    PiJsonAdapter,
    PiOutcomeDeclaration,
    PiRunConfig,
    read_pi_ndjson,
)
from src.contracts._json import sha256_json, thaw_json
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
from src.observatory.report import render_observatory_html


EXPECTED_FAILURE_COUNTS = {1: 2, 2: 2, 3: 3}
RUN_IDS = {"control": "run-v2-real-control", "candidate": "run-v2-real-candidate"}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n" for value in values)
    )


def _final_text(records: Iterable[Mapping[str, Any]]) -> str | None:
    final: str | None = None
    for record in records:
        if record.get("type") != "message_end":
            continue
        message = record.get("message")
        if not isinstance(message, Mapping):
            continue
        if message.get("role") != "assistant" or message.get("stopReason") != "stop":
            continue
        content = message.get("content")
        if not isinstance(content, (list, tuple)):
            continue
        texts = [
            item.get("text")
            for item in content
            if isinstance(item, Mapping)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        ]
        if texts:
            final = "".join(texts)
    return final


def verify_reference_answer(
    records: Iterable[Mapping[str, Any]], *, task_index: int, expected_counts=None
) -> tuple[PiOutcomeDeclaration, dict[str, Any]]:
    """Strict evaluator independent of the status claimed by the model.

    ``expected_counts`` is an optional override map (e.g. the V3 15-task
    suite); when omitted the frozen V2 counts are used, so V2 behavior is
    unchanged.
    """

    counts = expected_counts if expected_counts is not None else EXPECTED_FAILURE_COUNTS
    text = _final_text(records)
    parsed: Any = None
    parse_error: str | None = None
    try:
        parsed = json.loads(text) if text is not None else None
    except json.JSONDecodeError as exc:
        parse_error = exc.msg
    expected = {
        "task_id": f"task-{task_index}",
        "status": "SUCCESS",
        "failure_count": counts[task_index],
    }
    passed = (
        isinstance(parsed, dict)
        and set(parsed) == set(expected)
        and parsed == expected
    )
    declaration = PiOutcomeDeclaration(
        task_status=TaskStatus.SUCCESS if passed else TaskStatus.FAILURE,
        execution_validity=ExecutionValidity.VALID,
        verifier_status=(
            EpisodeVerifierStatus.PASSED if passed else EpisodeVerifierStatus.FAILED
        ),
        score=1.0 if passed else 0.0,
        termination_reason="EXACT_JSON_VERIFIER_FINISHED",
    )
    return declaration, {
        "task_index": task_index,
        "expected": expected,
        "observed": parsed,
        "raw_final_text": text,
        "parse_error": parse_error,
        "passed": passed,
        "important_boundary": "model-claimed status is not trusted",
    }


def _context(
    *,
    task_index: int,
    variant: str,
    experiment: ExperimentManifest,
    capabilities: frozenset[CaptureCapability],
) -> EpisodeContext:
    if variant == "control":
        policy = {
            "file_not_found": "repeat_identical_read_once_then_terminate",
            "maximum_identical_retries": 1,
        }
    else:
        policy = {
            "file_not_found": "discover_matching_file_then_read",
            "maximum_identical_retries": 0,
        }
    return EpisodeContext(
        task_id=f"pi-error-recovery/task-{task_index}",
        attempt=1,
        harness_manifest=HarnessManifest(
            name="pi",
            version=f"0.84.2-{variant}",
            revision=f"pi-error-recovery-policy/{variant}/r1",
            config_digest=sha256_json(policy),
            policy_flags=policy,
            hook_version=None,
        ),
        model_manifest=ModelManifest(
            provider="opencode-go",
            model_id="gpt-5.6-luna",
            revision="NOT_OBSERVABLE",
            sampling_config={"thinking": "minimal", "seed": "NOT_OBSERVABLE"},
            tokenizer_revision=None,
        ),
        environment_manifest=EnvironmentManifest(
            runtime_type="pi-local-process",
            revision="pi-synthetic-fixture-env/v1",
            image=None,
            resource_limits={"timeout_seconds": 120},
            network_policy="model-provider-only",
            task_snapshot=experiment.task_dataset_revision,
        ),
        evaluator_manifest=EvaluatorManifest(
            name="exact-json-verifier",
            revision="pi-error-recovery-verifier/v1",
            config_digest=sha256_json(
                {"strict_keys": True, "expected_counts": EXPECTED_FAILURE_COUNTS}
            ),
        ),
        experiment_manifest_ref=experiment.experiment_id,
        capabilities=capabilities | frozenset({CaptureCapability.VERIFIER_EVIDENCE}),
    )


def _load_capture(
    fixture: Path,
    *,
    task_index: int,
    variant: str,
) -> tuple[PiAdapterResult, dict[str, Any]]:
    records, issues = read_pi_ndjson(fixture.read_text())
    declaration, verifier_evidence = verify_reference_answer(records, task_index=task_index)
    adapter = PiJsonAdapter().convert(
        records,
        run_id=RUN_IDS[variant],
        episode_id=f"episode-v2-real-{variant}-{task_index}",
        trace_id=f"trace-v2-real-{variant}-{task_index}",
        config=PiRunConfig(),
        declared_outcome=declaration,
        source_issues=issues,
    )
    return adapter, verifier_evidence


def generate_real_pi_v2(
    output_dir: str | Path,
    *,
    fixture_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a reproducible V2 package without making another API call."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # This file belonged to the pre-release single-probe demo. Remove only the
    # exact known obsolete output so rerunning into the same directory cannot
    # leave evidence from two different experiment definitions mixed together.
    obsolete_probe = output / "pi-real-probe-conversion.json"
    if obsolete_probe.exists():
        obsolete_probe.unlink()
    fixtures = (
        Path(fixture_dir)
        if fixture_dir is not None
        else Path(__file__).parents[2] / "tests" / "fixtures" / "pi" / "v2-real"
    )
    experiment = ExperimentManifest(
        experiment_id="experiment-pi-tool-error-recovery-v2-real",
        revision="r1",
        task_dataset_revision="pi-tool-error-recovery-suite/r1",
        control_run_id=RUN_IDS["control"],
        candidate_run_id=RUN_IDS["candidate"],
        target_policy_flag="OBSERVED_TOOL_ERROR_LOOP",
        minimum_pairs=3,
        metadata={
            "capture": "real Pi subprocess NDJSON, sanitized after capture",
            "model_policy": "opencode-go/gpt-5.6-luna only; no fallback",
            "changed_factor": "Harness error-recovery system policy",
            "task_data": "synthetic and non-sensitive",
        },
    )

    adapters: list[PiAdapterResult] = []
    contexts: dict[str, EpisodeContext] = {}
    verifier_evidence: dict[str, Any] = {}
    source_files: dict[str, str] = {}
    for task_index in sorted(EXPECTED_FAILURE_COUNTS):
        for variant in ("control", "candidate"):
            fixture = fixtures / f"task-{task_index}-{variant}.ndjson"
            adapter, evidence = _load_capture(
                fixture, task_index=task_index, variant=variant
            )
            episode_id = f"episode-v2-real-{variant}-{task_index}"
            adapters.append(adapter)
            contexts[episode_id] = _context(
                task_index=task_index,
                variant=variant,
                experiment=experiment,
                capabilities=adapter.capabilities,
            )
            verifier_evidence[episode_id] = evidence
            source_files[episode_id] = fixture.name

    all_events = tuple(event for adapter in adapters for event in adapter.events)
    assembly = EpisodeAssembler().assemble(all_events, contexts=contexts)
    if assembly.warnings or assembly.quarantined_events:
        raise RuntimeError(
            "real Pi reference traces did not assemble cleanly: "
            f"warnings={assembly.warnings}, quarantined={len(assembly.quarantined_events)}"
        )
    controls = tuple(
        episode for episode in assembly.episodes if episode.run_id == RUN_IDS["control"]
    )
    candidates = tuple(
        episode for episode in assembly.episodes if episode.run_id == RUN_IDS["candidate"]
    )
    control_metrics = tuple(compute_episode_metrics(episode) for episode in controls)
    candidate_metrics = tuple(compute_episode_metrics(episode) for episode in candidates)
    attribution = AttributionEngine()
    control_reports = tuple(attribution.analyze(episode) for episode in controls)
    candidate_reports = tuple(attribution.analyze(episode) for episode in candidates)
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
        build_training_candidate_view(episode, target_policy_model="local-student-model")
        for episode in candidates
    )

    _write_json(output / "experiment-manifest.json", experiment.to_dict())
    _write_jsonl(output / "raw-events.jsonl", (event.to_dict() for event in all_events))
    _write_jsonl(output / "episodes-control.jsonl", (item.to_dict() for item in controls))
    _write_jsonl(output / "episodes-candidate.jsonl", (item.to_dict() for item in candidates))
    _write_json(output / "metrics-control.json", [item.to_dict() for item in control_metrics])
    _write_json(output / "metrics-candidate.json", [item.to_dict() for item in candidate_metrics])
    _write_jsonl(output / "diagnoses-control.jsonl", (item.to_dict() for item in control_reports))
    _write_jsonl(output / "diagnoses-candidate.jsonl", (item.to_dict() for item in candidate_reports))
    _write_json(output / "comparison.json", comparison.to_dict())
    _write_json(output / "gate-config.json", gate_config.to_dict())
    _write_json(output / "gate-result.json", gate.to_dict())
    _write_json(output / "training-candidates.json", [item.to_dict() for item in training_views])
    _write_json(
        output / "capture-evidence.json",
        {
            "sources": source_files,
            "source_checksums": {
                f"episode-v2-real-{variant}-{task_index}": next(
                    item.source_checksum
                    for item in adapters
                    if item.events[0].episode_id
                    == f"episode-v2-real-{variant}-{task_index}"
                )
                for task_index in sorted(EXPECTED_FAILURE_COUNTS)
                for variant in ("control", "candidate")
            },
            "verifier": verifier_evidence,
            "adapter_issues": {
                item.events[0].episode_id: [
                    {
                        "code": issue.code.value,
                        "message": issue.message,
                        "details": thaw_json(issue.details),
                    }
                    for issue in item.issues
                ]
                for item in adapters
            },
        },
    )
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
        "scope": "real paired Pi Harness experiment and decision package",
        "pi": PiRunConfig().to_dict(),
        "control_episode_count": len(controls),
        "candidate_episode_count": len(candidates),
        "observed_outcomes": {
            "control_successes": sum(
                episode.outcome.task_status is TaskStatus.SUCCESS for episode in controls
            ),
            "candidate_successes": sum(
                episode.outcome.task_status is TaskStatus.SUCCESS for episode in candidates
            ),
        },
        "paired_coverage": comparison.paired_coverage,
        "compatibility_mismatch_count": len(comparison.compatibility_mismatches),
        "gate_decision": gate.decision.value,
        "gate_note": "three-state Gate applies the predeclared cost and latency bounds",
        "target_slice": {
            "reason_code": experiment.target_policy_flag,
            "control": comparison.aggregate.control_target_slice_count,
            "candidate": comparison.aggregate.candidate_target_slice_count,
        },
        "training_semantics": {
            "teacher_model": "opencode-go/gpt-5.6-luna",
            "policy_relation": "OFF_POLICY",
            "sft_candidates": sum(item.sft_candidate for item in training_views),
            "on_policy_rl_candidates": sum(
                item.on_policy_rl_candidate for item in training_views
            ),
        },
        "evidence_checksums": {
            "experiment": experiment.checksum,
            "comparison": comparison.checksum,
            "gate": gate.checksum,
            "assembly": assembly.output_checksum,
        },
        "claim_boundary": (
            "three real executions per arm establish a reproducible reference case, not "
            "statistical generalization; Pi's internal decision state was not observable"
        ),
    }
    _write_json(output / "summary.json", summary)
    _write_json(
        output / "artifact-manifest.json",
        {
            "schema_version": "v2-artifact-manifest/v1",
            "files": sorted(
                path.name
                for path in output.iterdir()
                if path.is_file() and path.name != "artifact-manifest.json"
            ),
            "summary_checksum": sha256_json(summary),
        },
    )
    return summary
