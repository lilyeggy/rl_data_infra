#!/usr/bin/env python3
"""Collect on-policy group rollouts for Agentic RL.

Each task is sampled `group_size` times (typically G>=4 for GRPO) under the
current policy. Every attempt captures prompt/response token IDs, action mask,
behavior logprobs, and verifier outcome. Valid attempts are certified for
ON_POLICY_RL and compiled into an admission batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.assembly import assemble_execution_bundle
from src.assembly.episode_assembler import EpisodeAssembler
from src.certification import ConsumerProfile, ConsumerVerdict, certify_for
from src.contracts._json import canonical_json_bytes, sha256_json
from src.contracts.agent_episode import AgentEpisode, ExecutionValidity, TaskStatus
from src.contracts.dataset import DatasetManifest, DatasetPurpose, DatasetRole, DatasetSplit
from src.contracts.execution_bundle import ExecutionBundle
from src.contracts.execution_identity import ExecutionIdentity
from src.contracts.trace_event import EventStatus, TraceEvent
from src.integrations.slime import admit_on_policy_manifest
from src.learning import CertifiedArtifact, compile_dataset
from src.producers.base import (
    ProducerArtifact,
    ProducerCapability,
    ProducerExecutionStatus,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_mock_rollout(
    *,
    run_id: str,
    task_id: str,
    attempt_id: int,
    group_id: str,
    policy_fingerprint: str,
    sampling_fingerprint: str,
    passed: bool,
) -> tuple[AgentEpisode, ProducerArtifact, ExecutionBundle]:
    """Generate a valid, certified rollout tuple for tests and dry runs."""
    from tests.execution_fixtures import make_complete_event_stream, make_episode_context

    episode_id = f"ep-{task_id}-{attempt_id}"
    reward = 1.0 if passed else 0.0

    events = make_complete_event_stream(
        run_id=run_id,
        episode_id=episode_id,
        trace_id=f"trace-{episode_id}",
    )
    context = make_episode_context(
        task_id=task_id,
        attempt=attempt_id,
    )
    assembler = EpisodeAssembler()
    assembled = assembler.assemble(events, contexts={episode_id: context})
    episode = assembled.episodes[0]

    identity = ExecutionIdentity(
        run_id=run_id,
        task_id=task_id,
        episode_id=episode_id,
        attempt_id=attempt_id,
        producer_id="local_docker",
        producer_version="local-orchestrator/v1",
        group_id=group_id,
        policy_fingerprint=policy_fingerprint,
        sampling_fingerprint=sampling_fingerprint,
    )

    trace = {
        "prompt_ids": [101, 202, 303],
        "response_ids": [404, 505, 606, 707],
        "loss_mask": [1, 1, 1, 1],
        "response_logprobs": [-0.25, -0.15, -0.35, -0.05],
        "reward": reward,
    }

    artifact = ProducerArtifact(
        identity=identity,
        status=ProducerExecutionStatus.COMPLETED,
        capabilities=frozenset(
            {
                ProducerCapability.TOKEN_IDS,
                ProducerCapability.ACTION_MASK,
                ProducerCapability.BEHAVIOR_LOGPROBS,
                ProducerCapability.POLICY_VERSION,
                ProducerCapability.VERIFIER_EVIDENCE,
            }
        ),
        payload={"trajectory": {"traces": [trace]}},
    )

    verifier_checksum = _sha256(f"report-{task_id}-{attempt_id}-{passed}")
    bundle = assemble_execution_bundle(
        identity=identity,
        episode=episode,
        producer_artifacts=(artifact,),
        verifier_report_checksum=verifier_checksum,
    )

    return episode, artifact, bundle


def collect_group_rollouts(
    *,
    tasks: Sequence[str],
    group_size: int,
    output_dir: Path,
    run_id: str,
    policy_fingerprint: str,
    mock: bool = False,
) -> dict[str, Any]:
    """Execute G rollouts per task, certify for ON_POLICY_RL and compile Slime batch."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_file = output_dir / "producer-artifacts.jsonl"
    bundles_file = output_dir / "execution-bundles.jsonl"
    decisions_file = output_dir / "eligibility-decisions.jsonl"
    manifest_file = output_dir / "dataset-manifest.json"
    admission_file = output_dir / "slime-admission.json"

    artifacts_by_cs: dict[str, ProducerArtifact] = {}
    bundles_by_cs: dict[str, ExecutionBundle] = {}
    decisions_by_cs: dict[str, Any] = {}
    certified_entries: list[CertifiedArtifact] = []

    sampling_fingerprint = _sha256("temperature=0.8,top_p=0.95")

    for task_idx, task_id in enumerate(tasks):
        group_id = f"group-{run_id}-{task_id}"
        for attempt in range(1, group_size + 1):
            if mock:
                passed = (attempt % 2 == 1)
                episode, artifact, bundle = _make_mock_rollout(
                    run_id=run_id,
                    task_id=task_id,
                    attempt_id=attempt,
                    group_id=group_id,
                    policy_fingerprint=policy_fingerprint,
                    sampling_fingerprint=sampling_fingerprint,
                    passed=passed,
                )
            else:
                raise NotImplementedError(
                    "Live Docker rollouts require running with a configured LocalExecutionSpec."
                )

            decision = certify_for(
                episode,
                ConsumerProfile.ON_POLICY_RL,
                execution_bundle=bundle,
                policy_artifact=artifact,
                target_policy_fingerprint=policy_fingerprint,
            )

            artifacts_by_cs[artifact.checksum] = artifact
            bundles_by_cs[bundle.checksum] = bundle
            decisions_by_cs[decision.checksum] = decision

            if decision.verdict is ConsumerVerdict.ELIGIBLE:
                certified_entries.append(
                    CertifiedArtifact(
                        identity=artifact.identity,
                        decision=decision,
                        artifact_checksum=artifact.checksum,
                        split=DatasetSplit.TRAIN,
                        role=DatasetRole.TRAJECTORY,
                    )
                )

    with open(artifacts_file, "w", encoding="utf-8") as f:
        for a in artifacts_by_cs.values():
            f.write(json.dumps(a.to_dict(), ensure_ascii=False) + "\n")

    with open(bundles_file, "w", encoding="utf-8") as f:
        for b in bundles_by_cs.values():
            f.write(json.dumps(b.to_dict(), ensure_ascii=False) + "\n")

    with open(decisions_file, "w", encoding="utf-8") as f:
        for d in decisions_by_cs.values():
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")

    manifest = compile_dataset(
        dataset_id=f"rl-dataset-{run_id}",
        revision="r1",
        purpose=DatasetPurpose.ON_POLICY_RL,
        selection_policy_version="on-policy-rl/v1",
        artifacts=tuple(certified_entries),
    )
    manifest_file.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n")

    admission_batch = admit_on_policy_manifest(
        manifest,
        decisions_by_checksum=decisions_by_cs,
        bundles_by_checksum=bundles_by_cs,
        artifacts_by_checksum=artifacts_by_cs,
        minimum_group_size=min(2, group_size),
    )
    admission_file.write_text(
        json.dumps(admission_batch.to_dict(), indent=2, ensure_ascii=False) + "\n"
    )

    summary = {
        "run_id": run_id,
        "tasks_count": len(tasks),
        "total_rollouts": len(tasks) * group_size,
        "eligible_trajectories": len(certified_entries),
        "admitted_traces": len(admission_batch.traces),
        "policy_fingerprint": policy_fingerprint,
        "admission_checksum": admission_batch.checksum,
        "output_dir": str(output_dir),
    }
    (output_dir / "collection-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=["task-demo-1", "task-demo-2"], help="Task IDs to rollout")
    parser.add_argument("--group-size", type=int, default=4, help="Number of rollouts per task (G)")
    parser.add_argument("--run-id", default="rl-run-001", help="Unique Run ID")
    parser.add_argument(
        "--policy-fingerprint",
        default="0" * 64,
        help="Target policy fingerprint (64-char hex SHA256)",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for artifacts and admission")
    parser.add_argument("--mock", action="store_true", help="Generate valid mock rollouts for testing")
    args = parser.parse_args()

    summary = collect_group_rollouts(
        tasks=args.tasks,
        group_size=args.group_size,
        output_dir=Path(args.output_dir),
        run_id=args.run_id,
        policy_fingerprint=args.policy_fingerprint,
        mock=args.mock,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
