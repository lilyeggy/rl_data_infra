#!/usr/bin/env python3
"""Build a fail-closed base-vs-candidate report from finalized run bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.capture.model_proxy import ModelCallEvidence
from src.contracts._json import canonical_json_bytes
from src.contracts.agent_episode import AgentEpisode, IntegrityState
from src.contracts.execution_bundle import ExecutionBundle
from src.contracts.run_manifest import ExecutionRunManifest


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_run(root: Path) -> dict[str, Any]:
    manifest = ExecutionRunManifest.from_dict(_load_json(root / "execution-run-manifest.json"))
    episode = AgentEpisode.from_dict(_load_json(root / "finalized" / "episode.json"))
    bundle = ExecutionBundle.from_dict(
        _load_json(root / "finalized" / "execution-bundle.json")
    )
    summary = _load_json(root / "summary.json")
    evidence = tuple(
        ModelCallEvidence.from_dict(json.loads(line))
        for line in (root / "model-evidence.jsonl").read_text().splitlines()
        if line
    )

    identities = (manifest.identity, bundle.identity)
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError(f"identity mismatch in {root}")
    if episode.run_id != manifest.identity.run_id or episode.episode_id != manifest.identity.episode_id:
        raise ValueError(f"episode identity mismatch in {root}")
    if bundle.episode_checksum != episode.checksum:
        raise ValueError(f"episode checksum mismatch in {root}")
    if summary.get("execution_bundle_checksum") != bundle.checksum:
        raise ValueError(f"summary bundle checksum mismatch in {root}")
    if episode.integrity.state is not IntegrityState.COMPLETE:
        raise ValueError(f"holdout run is not COMPLETE: {root}")
    if episode.outcome.execution_validity.value != "VALID":
        raise ValueError(f"holdout run is not VALID: {root}")
    if not evidence or not all(item.rl_usable_call for item in evidence):
        raise ValueError(f"holdout run lacks RL-usable model evidence: {root}")
    revision = manifest.model_manifest.revision
    if any(item.backend.backend_model_revision != revision for item in evidence):
        raise ValueError(f"backend/model-manifest revision mismatch in {root}")

    return {
        "root": root.as_posix(),
        "manifest": manifest,
        "episode": episode,
        "bundle": bundle,
        "model_call_count": len(evidence),
        "resolved": episode.outcome.verifier_status.value == "PASSED",
    }


def holdout_decision(*, baseline_resolved: bool, candidate_resolved: bool) -> tuple[str, str]:
    if candidate_resolved and not baseline_resolved:
        return "ACCEPT", "IMPROVEMENT"
    if baseline_resolved and not candidate_resolved:
        return "REJECT", "REGRESSION"
    return "REJECT", "NO_IMPROVEMENT"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = _load_run(args.baseline_run)
    candidate = _load_run(args.candidate_run)
    baseline_manifest = baseline["manifest"]
    candidate_manifest = candidate["manifest"]
    if baseline_manifest.identity.task_id != candidate_manifest.identity.task_id:
        raise ValueError("baseline and candidate task IDs differ")
    if baseline_manifest.harness_manifest != candidate_manifest.harness_manifest:
        raise ValueError("baseline and candidate Harness manifests differ")
    if baseline_manifest.evaluator_manifest != candidate_manifest.evaluator_manifest:
        raise ValueError("baseline and candidate evaluator manifests differ")
    if (
        baseline_manifest.environment_manifest.task_snapshot
        != candidate_manifest.environment_manifest.task_snapshot
    ):
        raise ValueError("baseline and candidate task snapshots differ")
    if baseline_manifest.identity.sampling_fingerprint != candidate_manifest.identity.sampling_fingerprint:
        raise ValueError("baseline and candidate sampling fingerprints differ")

    decision, reason = holdout_decision(
        baseline_resolved=baseline["resolved"],
        candidate_resolved=candidate["resolved"],
    )

    def render(value: dict[str, Any]) -> dict[str, Any]:
        manifest = value["manifest"]
        episode = value["episode"]
        bundle = value["bundle"]
        return {
            "run_directory": value["root"],
            "run_id": manifest.identity.run_id,
            "task_id": manifest.identity.task_id,
            "model_revision": manifest.model_manifest.revision,
            "policy_fingerprint": manifest.identity.policy_fingerprint,
            "episode_checksum": episode.checksum,
            "execution_bundle_checksum": bundle.checksum,
            "model_call_count": value["model_call_count"],
            "integrity": episode.integrity.state.value,
            "execution_validity": episode.outcome.execution_validity.value,
            "verifier_status": episode.outcome.verifier_status.value,
            "resolved": value["resolved"],
        }

    report = {
        "schema_version": "policy-holdout-report/v1",
        "gate": "fixed-task-fixed-harness-policy-update",
        "baseline": render(baseline),
        "candidate": render(candidate),
        "decision": decision,
        "reason": reason,
        "limitations": [
            "This is one selected unseen DEV task, not a benchmark score.",
            "A rejected candidate must not be promoted to RL or production solely from train loss.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(report) + b"\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2))
    return 0 if decision == "ACCEPT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
