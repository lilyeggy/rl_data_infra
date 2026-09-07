"""ARCHIVED EXAMPLE: assemble a historical SWE-bench subset capture package.

This module is retained only to reproduce the V3 experiment.  New data-plane
code must use the producer, identity, certification, and dataset contracts in
``src`` instead of importing this scenario-specific packager.

Inputs (produced by archive/experiments/swebench/run_swebench.py):
  results/<instance_id>/trace.ndjson   sanitized real Pi NDJSON
  results/<instance_id>/eval.json      test-verifier outcome (FAIL_TO_PASS)

The verifier is the real SWE-bench test suite (FAIL_TO_PASS tests), i.e. an
external strict evaluator, never the model's own claim. Resolved instances
become offline SFT candidates; all instances produce canonical TraceEvents,
episodes, metrics and verifier evidence.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.analysis.metrics import compute_episode_metrics
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
from src.validation.episode_semantics import certify_episode

RUN_ID = "run-swebench-verified-subset"
# single-arm: no control/candidate pairing; contract requires distinct ids
CONTROL_RUN_ID = "run-swebench-verified-control-sentinel"
MODEL = "deepseek-v4-flash"
PROVIDER = "opencode-go"
PI_TOOLS = ("read", "bash", "write", "edit", "grep", "find", "ls", "glob")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n" for value in values)
    )


def build_package(
    results_root: str | Path,
    output_dir: str | Path,
    *,
    dataset_file: str | Path | None = None,
) -> dict[str, Any]:
    results = Path(results_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    instance_dirs = sorted(
        d for d in results.iterdir()
        if d.is_dir() and (d / "trace.ndjson").exists() and (d / "eval.json").exists()
    )
    if not instance_dirs:
        raise RuntimeError(f"no evaluated captures under {results}")

    dataset = {}
    if dataset_file is not None:
        dataset = json.loads(Path(dataset_file).read_text())

    experiment = ExperimentManifest(
        experiment_id="experiment-swebench-verified-subset-v3",
        revision="r1",
        task_dataset_revision="SWE-bench_Verified/10-instance-subset/r1",
        control_run_id=CONTROL_RUN_ID,
        candidate_run_id=RUN_ID,
        target_policy_flag="SWE_BENCH_FAIL_TO_PASS",
        minimum_pairs=1,
        metadata={
            "arms": "single-arm (no control/candidate pairing)",
            "source": "princeton-nlp/SWE-bench_Verified (validation split)",
            "model": f"{PROVIDER}/{MODEL}",
            "pi_version": "0.84.2",
            "tools": list(PI_TOOLS),
            "verifier": "real test suite FAIL_TO_PASS (no Docker; venv + pytest)",
            "subset_selection": "light pure-Python repos, Python 3.12 compatible",
            "gold_patch_never_shown_to_agent": True,
            "test_patch_never_shown_to_agent": True,
        },
    )

    adapters: dict[str, PiAdapterResult] = {}
    contexts: dict[str, EpisodeContext] = {}
    verifier_evidence: dict[str, Any] = {}
    test_outcomes: dict[str, Any] = {}

    for inst_dir in instance_dirs:
        instance_id = inst_dir.name
        eval_data = json.loads((inst_dir / "eval.json").read_text())
        resolved = bool(eval_data.get("resolved"))
        records, issues = read_pi_ndjson((inst_dir / "trace.ndjson").read_text())
        declared = PiOutcomeDeclaration(
            task_status=TaskStatus.SUCCESS if resolved else TaskStatus.FAILURE,
            execution_validity=ExecutionValidity.VALID,
            verifier_status=(
                EpisodeVerifierStatus.PASSED if resolved else EpisodeVerifierStatus.FAILED
            ),
            score=1.0 if resolved else 0.0,
            termination_reason="SWE_BENCH_TEST_VERIFIER_FINISHED",
        )
        adapter = PiJsonAdapter().convert(
            records,
            run_id=RUN_ID,
            episode_id=f"episode-swebench-{instance_id}",
            trace_id=f"trace-swebench-{instance_id}",
            config=PiRunConfig(model=MODEL, provider=PROVIDER, tools=PI_TOOLS),
            declared_outcome=declared,
            source_issues=issues,
        )
        adapters[instance_id] = adapter
        episode_id = f"episode-swebench-{instance_id}"
        dataset_row = dataset.get(instance_id, {})
        test_outcomes[instance_id] = {
            "resolved": resolved,
            "ftp": eval_data.get("ftp", {}),
            "ptp_sanity": eval_data.get("ptp_sanity", {}),
            "repo": dataset_row.get("repo"),
            "base_commit": dataset_row.get("base_commit"),
            "version": dataset_row.get("version"),
            "difficulty": dataset_row.get("difficulty"),
            "FAIL_TO_PASS": dataset_row.get("FAIL_TO_PASS", []),
        }
        verifier_evidence[instance_id] = {
            "resolved": resolved,
            "ftp_results": eval_data.get("ftp", {}),
            "ptp_sanity_results": eval_data.get("ptp_sanity", {}),
            "expected_ftp": dataset_row.get("FAIL_TO_PASS", []),
            "boundary": "test patch hidden from agent; gold patch unused",
        }
        contexts[episode_id] = EpisodeContext(
            task_id=f"swe-bench-verified/{instance_id}",
            attempt=1,
            harness_manifest=HarnessManifest(
                name="pi",
                version="0.84.2",
                revision="pi-swebench-agent/r1",
                config_digest=sha256_json({"tools": list(PI_TOOLS), "thinking": "minimal"}),
                policy_flags={"tools": list(PI_TOOLS), "thinking": "minimal"},
                hook_version=None,
            ),
            model_manifest=ModelManifest(
                provider=PROVIDER,
                model_id=MODEL,
                revision="NOT_OBSERVABLE",
                sampling_config={"thinking": "minimal", "seed": "NOT_OBSERVABLE"},
                tokenizer_revision=None,
            ),
            environment_manifest=EnvironmentManifest(
                runtime_type="pi-local-process",
                revision="swebench-repo-checkout/v1",
                image=None,
                resource_limits={"timeout_seconds": 600},
                network_policy="model-provider-only",
                task_snapshot=experiment.task_dataset_revision,
            ),
            evaluator_manifest=EvaluatorManifest(
                name="swe-bench-fail-to-pass-verifier",
                revision="swebench-test-runner/v1",
                config_digest=sha256_json(
                    {
                        "fail_to_pass": dataset_row.get("FAIL_TO_PASS", []),
                        "method": "venv + pytest, no Docker",
                    }
                ),
            ),
            experiment_manifest_ref=experiment.experiment_id,
            capabilities=adapter.capabilities
            | frozenset({CaptureCapability.VERIFIER_EVIDENCE}),
        )

    all_events = tuple(event for adapter in adapters.values() for event in adapter.events)
    assembly = EpisodeAssembler().assemble(all_events, contexts=contexts)
    if assembly.warnings or assembly.quarantined_events:
        raise RuntimeError(
            f"swebench package did not assemble cleanly: warnings={assembly.warnings}, "
            f"quarantined={len(assembly.quarantined_events)}"
        )
    episodes = assembly.episodes
    metrics = tuple(compute_episode_metrics(e) for e in episodes)
    # Only verifier-certified episodes may become SFT candidates.  The
    # eval.json ``resolved`` label is the verifier *input*; certification
    # re-validates the assembled episode end-to-end (real verifier PASSED
    # evidence, consistent terminal+verifier, integrity COMPLETE).
    certifications = {
        episode.episode_id: certify_episode(episode) for episode in episodes
    }
    training_views = tuple(
        build_training_candidate_view(episode, target_policy_model="local-student-model")
        for episode in episodes
    )
    sft_candidate_ids = tuple(
        view.episode_id
        for view in training_views
        if view.sft_candidate and certifications[view.episode_id].is_certified
    )

    _write_json(
        output / "episode-certifications.json",
        [cert.to_dict() for cert in certifications.values()],
    )

    _write_json(output / "experiment-manifest.json", experiment.to_dict())
    _write_jsonl(output / "raw-events.jsonl", (event.to_dict() for event in all_events))
    _write_jsonl(output / "episodes.jsonl", (e.to_dict() for e in episodes))
    _write_json(output / "metrics.json", [m.to_dict() for m in metrics])
    _write_json(output / "verifier-evidence.json", verifier_evidence)
    _write_json(output / "test-outcomes.json", test_outcomes)
    _write_json(output / "training-candidates.json", [item.to_dict() for item in training_views])
    _write_json(
        output / "capture-evidence.json",
        {
            "source_checksums": {
                instance_id: adapter.source_checksum
                for instance_id, adapter in adapters.items()
            },
            "model": MODEL,
            "provider": PROVIDER,
            "pi_version": "0.84.2",
            "adapter_issues": {
                instance_id: [
                    {
                        "code": issue.code.value,
                        "message": issue.message,
                        "details": thaw_json(issue.details),
                    }
                    for issue in adapter.issues
                ]
                for instance_id, adapter in adapters.items()
            },
        },
    )
    summary = {
        "release": "v3-swebench",
        "scope": "SWE-bench Verified subset through the data plane",
        "model": f"{PROVIDER}/{MODEL}",
        "instances": len(instance_dirs),
        "resolved": sum(1 for v in test_outcomes.values() if v["resolved"]),
        "test_outcomes": test_outcomes,
        "training_semantics": {
            "teacher_model": f"{PROVIDER}/{MODEL}",
            "sft_candidates": len(sft_candidate_ids),
            "sft_candidate_ids": list(sft_candidate_ids),
            "certified_episodes": sum(
                1 for cert in certifications.values() if cert.is_certified
            ),
            "on_policy_rl_candidates": sum(
                item.on_policy_rl_candidate for item in training_views
            ),
        },
        "verifier_certification": {
            "rule": "only verifier-certified episodes become SFT candidates",
            "evaluator": "SWE-bench test suite (FTP+PTP)",
            "boundary": (
                "eval.json resolved is verifier input; the assembled episode is "
                "re-certified before any training label is trusted"
            ),
        },
        "evidence_checksums": {
            "experiment": experiment.checksum,
            "assembly": assembly.output_checksum,
        },
        "claim_boundary": (
            "real third-party benchmark (SWE-bench Verified subset) executed by real "
            "Pi over a graded difficulty mix; verifier is the actual test suite; "
            "10 instances is an integration sample, not a benchmark claim; "
            "no Docker, venv + pytest evaluation"
        ),
    }
    _write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="/root/rivermind-data/swebench/results")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="/root/rivermind-data/swebench/selected_instances.json")
    args = parser.parse_args()
    print(
        json.dumps(
            build_package(args.results, args.output, dataset_file=args.dataset),
            ensure_ascii=False,
            indent=2,
        )
    )
    sys.exit(0)
