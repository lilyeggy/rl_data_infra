"""HISTORICAL EXAMPLE: V3 paired Pi runner for the 15-task suite.

Track A (harness self-improvement) compares candidate (V2.1 policy, the new
baseline) against v3 (candidate + counting discipline + immediate-answer
termination). The control arm retains the V2.1 story and provides off-policy
preference pairs (failure vs success).

All arms run on the same provider/model (opencode-go/deepseek-v4-flash, an
explicit variant choice, never a silent fallback). Sanitized NDJSON is written
to the output ``raw-ndjson`` directory and can be replayed offline with
``--replay`` (no further API calls).
"""

from __future__ import annotations

import json
import tempfile
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
from examples.legacy_scenarios.real_pi_v2 import verify_reference_answer
from examples.legacy_scenarios.task_suite import (
    ALL_TASKS,
    expected_counts_map,
    extra_files,
    task_file_payload,
)

V3_EXPERIMENT_ID = "experiment-pi-recovery-v3-15task"
LIVE_MODEL = "deepseek-v4-flash"
LIVE_PROVIDER = "opencode-go"
ARMS = ("control", "candidate", "v3")
RUN_IDS = {
    "control": "run-v3-live-control",
    "candidate": "run-v3-live-candidate",
    "v3": "run-v3-live-v3",
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
V3_POLICY = (
    CANDIDATE_POLICY
    + " After reading the task file, answer immediately: count only records "
    "whose status is exactly FAILURE; ignore SUCCESS, ERROR, UNKNOWN. "
    "Do not make further tool calls."
)


def build_prompt(task_index: int, arm: str) -> str:
    policy = {
        "control": CONTROL_POLICY,
        "candidate": CANDIDATE_POLICY,
        "v3": V3_POLICY,
    }[arm]
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


def run_episode(
    *,
    task_index: int,
    arm: str,
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
    for name, body in extra_files(task_index).items():
        (cwd / name).write_text(body)
    config = PiRunConfig(model=model, provider=provider, thinking=thinking)
    prompt = build_prompt(task_index, arm)
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
        (raw_dir / f"{arm}-{task_index}.ndjson").write_text(dump_pi_ndjson(records))
    declaration, _evidence = verify_reference_answer(
        records,
        task_index=task_index,
        expected_counts=expected_counts_map(ALL_TASKS),
    )
    return PiJsonAdapter().convert(
        records,
        run_id=RUN_IDS[arm],
        episode_id=f"episode-v3-live-{arm}-{task_index}",
        trace_id=f"trace-v3-live-{arm}-{task_index}",
        config=config,
        declared_outcome=declaration,
        source_issues=issues,
        normalize_tool_errors=True,
    )


def build_context(
    *,
    task_index: int,
    arm: str,
    experiment: ExperimentManifest,
    adapter: PiAdapterResult,
    model: str = LIVE_MODEL,
    provider: str = LIVE_PROVIDER,
) -> EpisodeContext:
    policy = {
        "control": {
            "file_not_found": "retry_same_path_once_then_stop",
            "maximum_identical_retries": 1,
        },
        "candidate": {
            "file_not_found": "bounded_discover_unique_then_read",
            "maximum_identical_retries": 0,
            "discovery_budget": 1,
        },
        "v3": {
            "file_not_found": "bounded_discover_unique_then_read",
            "maximum_identical_retries": 0,
            "discovery_budget": 1,
            "counting": "failures_only_ignore_other_statuses",
            "termination": "answer_immediately_after_read",
        },
    }[arm]
    return EpisodeContext(
        task_id=f"pi-live-v3/task-{task_index}",
        attempt=1,
        harness_manifest=HarnessManifest(
            name="pi",
            version=f"0.84.2-{arm}",
            revision=f"pi-live-v3-policy/{arm}/r1",
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
            revision="pi-v3-synthetic-env/v1",
            image=None,
            resource_limits={"timeout_seconds": 240},
            network_policy="model-provider-only",
            task_snapshot=experiment.task_dataset_revision,
        ),
        evaluator_manifest=EvaluatorManifest(
            name="exact-json-verifier",
            revision="pi-v3-verifier/v1",
            config_digest=sha256_json(
                {"strict_keys": True, "expected_counts": expected_counts_map(ALL_TASKS)}
            ),
        ),
        experiment_manifest_ref=experiment.experiment_id,
        capabilities=adapter.capabilities | frozenset({CaptureCapability.VERIFIER_EVIDENCE}),
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(v, ensure_ascii=False, sort_keys=True) + "\n" for v in values
        )
    )


def _adapt_fixture(
    task_index: int,
    arm: str,
    fixture: Path,
    *,
    expected_counts,
) -> PiAdapterResult:
    records, issues = read_pi_ndjson(fixture.read_text())
    declaration, _evidence = verify_reference_answer(
        records, task_index=task_index, expected_counts=expected_counts
    )
    return PiJsonAdapter().convert(
        records,
        run_id=RUN_IDS[arm],
        episode_id=f"episode-v3-live-{arm}-{task_index}",
        trace_id=f"trace-v3-live-{arm}-{task_index}",
        config=PiRunConfig(model=LIVE_MODEL, provider=LIVE_PROVIDER, thinking="minimal"),
        declared_outcome=declaration,
        source_issues=issues,
        normalize_tool_errors=True,
    )


def _experiment(arms, tasks, model=LIVE_MODEL, provider=LIVE_PROVIDER) -> ExperimentManifest:
    return ExperimentManifest(
        experiment_id=V3_EXPERIMENT_ID,
        revision="r1",
        task_dataset_revision="pi-v3-15-task-suite/r1",
        control_run_id=RUN_IDS["control"],
        candidate_run_id=RUN_IDS["candidate"],
        target_policy_flag="OBSERVED_TOOL_ERROR_LOOP",
        minimum_pairs=3,
        metadata={
            "model": model,
            "provider": provider,
            "model_change_reason": "gpt-5.6-luna is region-blocked on the live server",
            "model_change_is_not_silent_fallback": True,
            "pi_version": "0.84.2",
            "task_suite": "examples.legacy_scenarios.task_suite V3 (15 tasks)",
            "arms": list(arms),
            "track_a_primary_comparison": "candidate vs v3",
        },
    )


def _assemble_and_render(
    *,
    adapters: dict[str, PiAdapterResult],
    contexts: dict[str, EpisodeContext],
    experiment: ExperimentManifest,
    output: Path,
    arms,
    tasks,
    raw_ndjson: dict[str, str],
) -> dict[str, Any]:
    all_events = tuple(event for a in adapters.values() for event in a.events)
    assembly = EpisodeAssembler().assemble(all_events, contexts=contexts)
    if assembly.warnings or assembly.quarantined_events:
        raise RuntimeError(
            f"v3 experiment did not assemble cleanly: warnings={assembly.warnings}, "
            f"quarantined={len(assembly.quarantined_events)}"
        )

    def episodes_of(run_id: str):
        return tuple(e for e in assembly.episodes if e.run_id == run_id)

    attribution = AttributionEngine()
    reports: dict[str, dict] = {}
    for arm in arms:
        run_id = RUN_IDS[arm]
        reports[run_id] = {
            r.episode_id: tuple(i.reason_code for i in r.diagnoses)
            for r in (attribution.analyze(e) for e in episodes_of(run_id))
        }

    def compare_pair(name: str, base_arm: str, new_arm: str) -> dict:
        base_eps = episodes_of(RUN_IDS[base_arm])
        new_eps = episodes_of(RUN_IDS[new_arm])
        base_metrics = tuple(compute_episode_metrics(e) for e in base_eps)
        new_metrics = tuple(compute_episode_metrics(e) for e in new_eps)
        comparison = compare_runs(
            base_eps,
            new_eps,
            control_metrics=base_metrics,
            candidate_metrics=new_metrics,
            experiment=experiment,
            control_reason_codes=reports[RUN_IDS[base_arm]],
            candidate_reason_codes=reports[RUN_IDS[new_arm]],
        )
        gate = evaluate_gate(comparison, GateConfig())
        _write_json(output / f"comparison-{name}.json", comparison.to_dict())
        _write_json(output / f"gate-{name}.json", gate.to_dict())
        return {
            "base": RUN_IDS[base_arm],
            "new": RUN_IDS[new_arm],
            "base_successes": sum(
                e.outcome.task_status.value == "SUCCESS" for e in base_eps
            ),
            "new_successes": sum(
                e.outcome.task_status.value == "SUCCESS" for e in new_eps
            ),
            "paired_coverage": comparison.paired_coverage,
            "gate_decision": gate.decision.value,
            "gate_checks": [c.to_dict() for c in gate.checks],
        }

    comparisons = {}
    if "control" in arms and "candidate" in arms:
        comparisons["candidate-vs-control"] = compare_pair(
            "candidate-vs-control", "control", "candidate"
        )
    if "candidate" in arms and "v3" in arms:
        comparisons["v3-vs-candidate"] = compare_pair(
            "v3-vs-candidate", "candidate", "v3"
        )

    training_views = (
        tuple(
            build_training_candidate_view(episode, target_policy_model="local-student-model")
            for episode in episodes_of(RUN_IDS["v3"])
        )
        if "v3" in arms
        else ()
    )

    _write_json(output / "experiment-manifest.json", experiment.to_dict())
    _write_jsonl(output / "raw-events.jsonl", (e.to_dict() for e in all_events))
    for arm in arms:
        _write_jsonl(
            output / f"episodes-{RUN_IDS[arm]}.jsonl",
            (e.to_dict() for e in episodes_of(RUN_IDS[arm])),
        )
    _write_json(
        output / "capture-evidence.json",
        {
            "source_checksums": raw_ndjson,
            "model": LIVE_MODEL,
            "provider": LIVE_PROVIDER,
            "pi_version": "0.84.2",
            "thinking": "minimal",
            "adapter_issues": {
                eid: [
                    {
                        "code": issue.code.value,
                        "message": issue.message,
                        "details": thaw_json(issue.details),
                    }
                    for issue in a.issues
                ]
                for eid, a in adapters.items()
            },
        },
    )
    _write_json(
        output / "training-candidates-v3.json",
        [i.to_dict() for i in training_views],
    )
    summary = {
        "release": "v3-live",
        "scope": "real paired Pi experiment, 15-task suite, arms=" + ",".join(arms),
        "model": f"{LIVE_PROVIDER}/{LIVE_MODEL}",
        "model_change_not_silent_fallback": True,
        "tasks": list(tasks),
        "arms": list(arms),
        "per_arm_successes": {
            arm: sum(e.outcome.task_status.value == "SUCCESS" for e in episodes_of(RUN_IDS[arm]))
            for arm in arms
        },
        "per_arm_by_task": {
            arm: {
                e.task_id: e.outcome.task_status.value
                for e in episodes_of(RUN_IDS[arm])
            }
            for arm in arms
        },
        "comparisons": comparisons,
        "training_semantics": {
            "teacher_model": f"{LIVE_PROVIDER}/{LIVE_MODEL}",
            "sft_candidates": sum(i.sft_candidate for i in training_views),
            "on_policy_rl_candidates": sum(
                i.on_policy_rl_candidate for i in training_views
            ),
        },
        "claim_boundary": (
            "real Pi executions over a graded 15-task suite; Pi's internal decision "
            "state remains black-box; small sample per arm"
        ),
    }
    _write_json(output / "summary.json", summary)
    _write_json(
        output / "artifact-manifest.json",
        {
            "schema_version": "v3-artifact-manifest/v1",
            "files": sorted(p.name for p in output.iterdir() if p.is_file()),
        },
    )
    return summary


def run_live_experiment(
    output_dir: str | Path,
    *,
    workspace_root: str | Path | None = None,
    timeout_seconds: float = 240,
    thinking: str = "minimal",
    tasks=ALL_TASKS,
    arms=ARMS,
    model: str = LIVE_MODEL,
    provider: str = LIVE_PROVIDER,
) -> dict[str, Any]:
    """Run the V3 arms over the task suite (live API calls) and render."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    experiment = _experiment(arms, tasks, model, provider)

    adapters: dict[str, PiAdapterResult] = {}
    contexts: dict[str, EpisodeContext] = {}
    raw_ndjson: dict[str, str] = {}

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(workspace_root) if workspace_root is not None else Path(temporary) / "workspace"
        root.mkdir(parents=True, exist_ok=True)
        for task_index in sorted(tasks):
            for arm in arms:
                episode_id = f"episode-v3-live-{arm}-{task_index}"
                adapter = run_episode(
                    task_index=task_index,
                    arm=arm,
                    workspace_root=root,
                    timeout_seconds=timeout_seconds,
                    thinking=thinking,
                    raw_output_dir=output / "raw-ndjson",
                    model=model,
                    provider=provider,
                )
                adapters[episode_id] = adapter
                contexts[episode_id] = build_context(
                    task_index=task_index,
                    arm=arm,
                    experiment=experiment,
                    adapter=adapter,
                    model=model,
                    provider=provider,
                )
                raw_ndjson[episode_id] = adapter.source_checksum

    return _assemble_and_render(
        adapters=adapters,
        contexts=contexts,
        experiment=experiment,
        output=output,
        arms=arms,
        tasks=tasks,
        raw_ndjson=raw_ndjson,
    )


def build_from_fixtures(
    fixtures_dir: str | Path,
    output_dir: str | Path,
    *,
    tasks=ALL_TASKS,
    arms=ARMS,
) -> dict[str, Any]:
    """Replay already-captured sanitized NDJSON (no further API calls)."""
    fixtures = Path(fixtures_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    experiment = _experiment(arms, tasks)

    adapters: dict[str, PiAdapterResult] = {}
    contexts: dict[str, EpisodeContext] = {}
    raw_ndjson: dict[str, str] = {}

    for task_index in sorted(tasks):
        for arm in arms:
            fixture = fixtures / f"{arm}-{task_index}.ndjson"
            if not fixture.exists():
                raise FileNotFoundError(f"missing fixture {fixture}")
            episode_id = f"episode-v3-live-{arm}-{task_index}"
            adapter = _adapt_fixture(
                task_index,
                arm,
                fixture,
                expected_counts=expected_counts_map(ALL_TASKS),
            )
            adapters[episode_id] = adapter
            contexts[episode_id] = build_context(
                task_index=task_index,
                arm=arm,
                experiment=experiment,
                adapter=adapter,
            )
            raw_ndjson[episode_id] = adapter.source_checksum

    return _assemble_and_render(
        adapters=adapters,
        contexts=contexts,
        experiment=experiment,
        output=output,
        arms=arms,
        tasks=tasks,
        raw_ndjson=raw_ndjson,
    )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--tasks", default=",".join(str(t) for t in ALL_TASKS))
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--replay", default=None, help="dir of {arm}-{task}.ndjson fixtures")
    parser.add_argument("--model", default=LIVE_MODEL)
    parser.add_argument("--provider", default=LIVE_PROVIDER)
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    tasks = tuple(int(t) for t in args.tasks.split(","))
    arms = tuple(a for a in args.arms.split(","))
    if args.replay:
        result = build_from_fixtures(args.replay, args.output, tasks=tasks, arms=arms)
    else:
        result = run_live_experiment(
            args.output,
            workspace_root=args.workspace,
            timeout_seconds=args.timeout,
            tasks=tasks,
            arms=arms,
            model=args.model,
            provider=args.provider,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0)
