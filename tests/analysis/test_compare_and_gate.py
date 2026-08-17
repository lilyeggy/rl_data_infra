from __future__ import annotations

import unittest

from src.analysis.compare import compare_runs
from src.analysis.metrics import compute_episode_metrics
from src.analysis.regression_gate import GateConfig, GateDecision, evaluate_gate
from src.assembly.episode_assembler import EpisodeAssembler
from src.contracts._json import sha256_json
from src.contracts.experiment import ExperimentManifest
from src.contracts.manifests import EnvironmentManifest, HarnessManifest
from src.contracts.trace_event import EventStatus, EventType, TraceEvent
from tests.execution_fixtures import make_complete_event_stream, make_episode_context


def _episode(
    *,
    task: str,
    run: str,
    episode_id: str,
    harness_version: str,
    success: bool,
    environment_revision: str = "fixture-env-r1",
):
    events: list[TraceEvent] = []
    for event in make_complete_event_stream(run_id=run, episode_id=episode_id):
        value = event.to_dict()
        value["event_id"] = f"{event.event_id}-{episode_id}"
        if value["parent_span_id"] is not None:
            value["parent_span_id"] = f"{value['parent_span_id']}-{episode_id}"
        value["span_id"] = f"{value['span_id']}-{episode_id}"
        if event.event_type is EventType.VERIFICATION_FINISHED and not success:
            value["status"] = EventStatus.FAILED.value
            value["attributes"]["passed"] = False
        if event.event_type is EventType.EPISODE_FINISHED:
            value["status"] = (
                EventStatus.SUCCEEDED.value if success else EventStatus.FAILED.value
            )
            value["attributes"].update(
                {
                    "task_status": "SUCCESS" if success else "FAILURE",
                    "verifier_status": "PASSED" if success else "FAILED",
                    "score": 1.0 if success else 0.0,
                    "evidence_event_ids": [f"evt-verification-finished-{episode_id}"],
                }
            )
        events.append(TraceEvent.from_dict(value))
    digest = sha256_json({"version": harness_version})
    harness = HarnessManifest(
        name="pi",
        version=harness_version,
        revision=f"pi-{harness_version}",
        config_digest=digest,
        policy_flags={"structured_tool_errors": harness_version == "candidate"},
        hook_version=None,
    )
    environment = EnvironmentManifest(
        runtime_type="process",
        revision=environment_revision,
        image=None,
        resource_limits={"timeout_seconds": 30},
        network_policy="disabled",
        task_snapshot="dataset-r1",
    )
    context = make_episode_context(
        task_id=task,
        harness_manifest=harness,
        environment_manifest=environment,
        experiment_manifest_ref="experiment-v2",
    )
    result = EpisodeAssembler().assemble(events, contexts={episode_id: context})
    return result.episodes[0]


def _manifest() -> ExperimentManifest:
    return ExperimentManifest(
        experiment_id="experiment-v2",
        revision="r1",
        task_dataset_revision="dataset-r1",
        control_run_id="run-control",
        candidate_run_id="run-candidate",
        target_policy_flag="TOOL_ERROR_FEEDBACK_LOSS",
        minimum_pairs=3,
    )


class CompareAndGateTest(unittest.TestCase):
    def test_compatible_paired_improvement_is_accepted(self) -> None:
        controls = tuple(
            _episode(
                task=f"task-{index}",
                run="run-control",
                episode_id=f"control-{index}",
                harness_version="control",
                success=False,
            )
            for index in range(3)
        )
        candidates = tuple(
            _episode(
                task=f"task-{index}",
                run="run-candidate",
                episode_id=f"candidate-{index}",
                harness_version="candidate",
                success=True,
            )
            for index in range(3)
        )
        report = compare_runs(
            controls,
            candidates,
            control_metrics=tuple(compute_episode_metrics(item) for item in controls),
            candidate_metrics=tuple(compute_episode_metrics(item) for item in candidates),
            experiment=_manifest(),
            control_reason_codes={
                item.episode_id: ("TOOL_ERROR_FEEDBACK_LOSS",) for item in controls
            },
            candidate_reason_codes={},
        )
        self.assertTrue(report.comparable)
        self.assertEqual(report.paired_coverage, 1.0)
        self.assertEqual(report.aggregate.success_rate_delta, 1.0)
        gate = evaluate_gate(report, GateConfig())
        self.assertEqual(gate.decision, GateDecision.ACCEPT)
        self.assertTrue(all(check.passed is True for check in gate.checks))

    def test_environment_confounder_blocks_decision(self) -> None:
        controls = (
            _episode(
                task="task-0",
                run="run-control",
                episode_id="control-0",
                harness_version="control",
                success=False,
            ),
        )
        candidates = (
            _episode(
                task="task-0",
                run="run-candidate",
                episode_id="candidate-0",
                harness_version="candidate",
                success=True,
                environment_revision="changed-environment",
            ),
        )
        report = compare_runs(
            controls,
            candidates,
            control_metrics=(compute_episode_metrics(controls[0]),),
            candidate_metrics=(compute_episode_metrics(candidates[0]),),
            experiment=_manifest(),
        )
        self.assertFalse(report.comparable)
        self.assertEqual(report.compatibility_mismatches[0].field, "environment_manifest")
        gate = evaluate_gate(report, GateConfig(minimum_episode_pairs=1))
        self.assertEqual(gate.decision, GateDecision.INSUFFICIENT_EVIDENCE)

    def test_comparable_regression_is_rejected(self) -> None:
        controls = tuple(
            _episode(
                task=f"task-{index}",
                run="run-control",
                episode_id=f"control-{index}",
                harness_version="control",
                success=True,
            )
            for index in range(3)
        )
        candidates = tuple(
            _episode(
                task=f"task-{index}",
                run="run-candidate",
                episode_id=f"candidate-{index}",
                harness_version="candidate",
                success=False,
            )
            for index in range(3)
        )
        report = compare_runs(
            controls,
            candidates,
            control_metrics=tuple(compute_episode_metrics(item) for item in controls),
            candidate_metrics=tuple(compute_episode_metrics(item) for item in candidates),
            experiment=_manifest(),
        )
        gate = evaluate_gate(
            report,
            GateConfig(require_target_slice_improvement=False),
        )
        self.assertEqual(gate.decision, GateDecision.REJECT)
        failed = {check.rule for check in gate.checks if check.passed is False}
        self.assertIn("success_rate_drop_tolerance", failed)


if __name__ == "__main__":
    unittest.main()
