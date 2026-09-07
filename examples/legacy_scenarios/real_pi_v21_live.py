"""HISTORICAL EXAMPLE: live paired Pi runner for old Harness evidence.

This module invokes the real Pi CLI (0.84.2) with a fixed provider/model and
runs a control/candidate error-recovery experiment against a local synthetic
workspace.  It produces canonical TraceEvents through the Pi adapter and ends
with a frozen Regression Gate decision.

The experiment model is an explicit variant choice (region-constrained) and is
never a silent fallback: both arms use the same model, and the V2 frozen
baseline (gpt-5.6-luna) artifacts remain unchanged.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from src.analysis.attribution import AttributionEngine
from src.analysis.compare import compare_runs
from src.analysis.metrics import compute_episode_metrics
from src.analysis.regression_gate import GateConfig, evaluate_gate
from src.assembly.episode_assembler import EpisodeAssembler, EpisodeContext
from src.capture.pi_adapter import (
    PiAdapterResult,
    PiJsonAdapter,
    PiRunConfig,
    dump_pi_ndjson,
    read_pi_ndjson,
)
from src.capture.pi_runner import run_pi_process
from src.contracts._json import sha256_json, thaw_json
from src.contracts.agent_episode import CaptureCapability
from src.contracts.experiment import ExperimentManifest
from src.contracts.manifests import (
    EnvironmentManifest,
    EvaluatorManifest,
    HarnessManifest,
    ModelManifest,
)
from src.contracts.training_candidate import build_training_candidate_view
from examples.legacy_scenarios.real_pi_v2 import (
    EXPECTED_FAILURE_COUNTS,
    verify_reference_answer,
)


LIVE_EXPERIMENT_ID = "experiment-pi-recovery-live-deepseek-v4-flash"
LIVE_MODEL = "deepseek-v4-flash"
LIVE_PROVIDER = "opencode-go"
LIVE_RUN_IDS = {
    "control": "run-v21-live-control",
    "candidate": "run-v21-live-candidate",
}

CONTROL_POLICY = (
    "Harness policy: if reading the requested file fails because it does not "
    "exist, retry reading the exact same path once, then stop. Never search "
    "for another file."
)
CANDIDATE_POLICY = (
    "Harness policy: if reading the requested file fails because it does not "
    "exist, search this directory with find using pattern task-*.json, then "
    "read the unique file found. If zero or multiple candidates are found, "
    "stop. Never retry the missing path."
)


def build_prompt(task_index: int, variant: str) -> str:
    policy = CONTROL_POLICY if variant == "control" else CANDIDATE_POLICY
    return (
        f"Start by reading missing-{task_index}.json. "
        f"{policy} "
        f"Return only JSON (no markdown, no code fences) with fields "
        f"task_id, status, failure_count. "
        f"status must be exactly the string SUCCESS when you successfully read "
        f"the task file, regardless of the records inside it. "
        f"failure_count is the number of FAILURE records inside that task file. "
        f"Use at most 4 tool calls total."
    )


def task_file_payload(task_index: int) -> dict[str, Any]:
    failure_count = EXPECTED_FAILURE_COUNTS[task_index]
    total = failure_count + 2
    records = [
        {"id": chr(97 + index), "status": "FAILURE" if index < failure_count else "SUCCESS"}
        for index in range(total)
    ]
    return {"task_id": f"task-{task_index}", "records": records}


def run_episode(
    *,
    task_index: int,
    variant: str,
    workspace_root: str | Path,
    model: str = LIVE_MODEL,
    provider: str = LIVE_PROVIDER,
    timeout_seconds: float = 240,
    thinking: str = "minimal",
    raw_output_dir: str | Path | None = None,
) -> PiAdapterResult:
    cwd = Path(workspace_root) / f"task-{task_index}"
    cwd.mkdir(parents=True, exist_ok=True)
    (cwd / f"task-{task_index}.json").write_text(
        json.dumps(task_file_payload(task_index), ensure_ascii=False) + "\n"
    )
    config = PiRunConfig(model=model, provider=provider, thinking=thinking)
    prompt = build_prompt(task_index, variant)
    capture = run_pi_process(
        config=config,
        prompt=prompt,
        cwd=cwd,
        timeout_seconds=timeout_seconds,
    )
    records, issues = read_pi_ndjson(capture.stdout)
    if raw_output_dir is not None:
        raw_dir = Path(raw_output_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{variant}-{task_index}.ndjson").write_text(dump_pi_ndjson(records))
    declaration, _evidence = verify_reference_answer(records, task_index=task_index)
    return PiJsonAdapter().convert(
        records,
        run_id=LIVE_RUN_IDS[variant],
        episode_id=f"episode-v21-live-{variant}-{task_index}",
        trace_id=f"trace-v21-live-{variant}-{task_index}",
        config=config,
        declared_outcome=declaration,
        source_issues=issues,
        normalize_tool_errors=True,
    )


def build_context(
    *,
    task_index: int,
    variant: str,
    experiment: ExperimentManifest,
    adapter: PiAdapterResult,
    model: str = LIVE_MODEL,
    provider: str = LIVE_PROVIDER,
) -> EpisodeContext:
    policy = (
        {"file_not_found": "retry_same_path_once_then_stop", "maximum_identical_retries": 1}
        if variant == "control"
        else {
            "file_not_found": "bounded_discover_unique_then_read",
            "maximum_identical_retries": 0,
            "discovery_budget": 1,
        }
    )
    return EpisodeContext(
        task_id=f"pi-live-recovery/task-{task_index}",
        attempt=1,
        harness_manifest=HarnessManifest(
            name="pi",
            version=f"0.84.2-{variant}",
            revision=f"pi-live-policy/{variant}/r1",
            config_digest=sha256_json(policy),
            policy_flags=policy,
            hook_version=None,
        ),
        model_manifest=ModelManifest(
            provider=provider,
            model_id=model,
            revision="NOT_OBSERVABLE",
            sampling_config={"thinking": "minimal", "seed": "NOT_OBSERVABLE"},
            tokenizer_revision=None,
        ),
        environment_manifest=EnvironmentManifest(
            runtime_type="pi-local-process",
            revision="pi-live-synthetic-env/v1",
            image=None,
            resource_limits={"timeout_seconds": 240},
            network_policy="model-provider-only",
            task_snapshot=experiment.task_dataset_revision,
        ),
        evaluator_manifest=EvaluatorManifest(
            name="exact-json-verifier",
            revision="pi-live-verifier/v1",
            config_digest=sha256_json(
                {"strict_keys": True, "expected_counts": EXPECTED_FAILURE_COUNTS}
            ),
        ),
        experiment_manifest_ref=experiment.experiment_id,
        capabilities=adapter.capabilities | frozenset({CaptureCapability.VERIFIER_EVIDENCE}),
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: Any) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        )
    )


def run_live_experiment(
    output_dir: str | Path,
    *,
    workspace_root: str | Path | None = None,
    model: str = LIVE_MODEL,
    provider: str = LIVE_PROVIDER,
    timeout_seconds: float = 240,
    thinking: str = "minimal",
) -> dict[str, Any]:
    """Run 3 control + 3 candidate real Pi episodes and evaluate the frozen Gate."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    experiment = ExperimentManifest(
        experiment_id=LIVE_EXPERIMENT_ID,
        revision="r1",
        task_dataset_revision="pi-live-error-recovery-suite/r1",
        control_run_id=LIVE_RUN_IDS["control"],
        candidate_run_id=LIVE_RUN_IDS["candidate"],
        target_policy_flag="OBSERVED_TOOL_ERROR_LOOP",
        minimum_pairs=3,
        metadata={
            "model": model,
            "provider": provider,
            "model_change_reason": "gpt-5.6-luna is region-blocked on the live server",
            "model_change_is_not_silent_fallback": True,
            "v2_frozen_baseline_model": "gpt-5.6-luna",
            "pi_version": "0.84.2",
        },
    )

    adapters: dict[str, PiAdapterResult] = {}
    contexts: dict[str, EpisodeContext] = {}
    raw_ndjson: dict[str, str] = {}

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(workspace_root) if workspace_root is not None else Path(temporary) / "workspace"
        root.mkdir(parents=True, exist_ok=True)
        for task_index in sorted(EXPECTED_FAILURE_COUNTS):
            for variant in ("control", "candidate"):
                episode_id = f"episode-v21-live-{variant}-{task_index}"
                adapter = run_episode(
                    task_index=task_index,
                    variant=variant,
                    workspace_root=root,
                    model=model,
                    provider=provider,
                    timeout_seconds=timeout_seconds,
                    thinking=thinking,
                    raw_output_dir=output / "raw-ndjson",
                )
                adapters[episode_id] = adapter
                contexts[episode_id] = build_context(
                    task_index=task_index,
                    variant=variant,
                    experiment=experiment,
                    adapter=adapter,
                    model=model,
                    provider=provider,
                )
                raw_ndjson[episode_id] = adapter.source_checksum

    all_events = tuple(event for adapter in adapters.values() for event in adapter.events)
    assembly = EpisodeAssembler().assemble(all_events, contexts=contexts)
    if assembly.warnings or assembly.quarantined_events:
        raise RuntimeError(
            f"live experiment did not assemble cleanly: warnings={assembly.warnings}, "
            f"quarantined={len(assembly.quarantined_events)}"
        )
    controls = tuple(
        episode for episode in assembly.episodes if episode.run_id == LIVE_RUN_IDS["control"]
    )
    candidates = tuple(
        episode for episode in assembly.episodes if episode.run_id == LIVE_RUN_IDS["candidate"]
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
    gate = evaluate_gate(comparison, GateConfig())
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
    _write_jsonl(
        output / "diagnoses-candidate.jsonl",
        (item.to_dict() for item in candidate_reports),
    )
    _write_json(output / "comparison.json", comparison.to_dict())
    _write_json(output / "gate-config.json", GateConfig().to_dict())
    _write_json(output / "gate-result.json", gate.to_dict())
    _write_json(output / "training-candidates.json", [item.to_dict() for item in training_views])
    _write_json(
        output / "capture-evidence.json",
        {
            "source_checksums": raw_ndjson,
            "model": model,
            "provider": provider,
            "pi_version": "0.84.2",
            "thinking": thinking,
            "adapter_issues": {
                episode_id: [
                    {
                        "code": issue.code.value,
                        "message": issue.message,
                        "details": thaw_json(issue.details),
                    }
                    for issue in adapter.issues
                ]
                for episode_id, adapter in adapters.items()
            },
        },
    )
    summary = {
        "release": "v2.1-live",
        "scope": "real paired Pi experiment on live server",
        "model": model,
        "provider": provider,
        "model_change_not_silent_fallback": True,
        "v2_frozen_baseline_model": "gpt-5.6-luna",
        "control_episode_count": len(controls),
        "candidate_episode_count": len(candidates),
        "observed_outcomes": {
            "control_successes": sum(
                episode.outcome.task_status.value == "SUCCESS" for episode in controls
            ),
            "candidate_successes": sum(
                episode.outcome.task_status.value == "SUCCESS" for episode in candidates
            ),
        },
        "paired_coverage": comparison.paired_coverage,
        "gate_decision": gate.decision.value,
        "target_slice": {
            "reason_code": experiment.target_policy_flag,
            "control": comparison.aggregate.control_target_slice_count,
            "candidate": comparison.aggregate.candidate_target_slice_count,
        },
        "training_semantics": {
            "teacher_model": f"{provider}/{model}",
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
            "real Pi executions with explicit policy prompts; model differs from the "
            "frozen V2 baseline by user decision due to region availability"
        ),
    }
    _write_json(output / "summary.json", summary)
    _write_json(
        output / "artifact-manifest.json",
        {
            "schema_version": "v2.1-live-artifact-manifest/v1",
            "files": sorted(
                path.name for path in output.iterdir() if path.is_file()
            ),
            "summary_checksum": sha256_json(summary),
        },
    )
    return summary
