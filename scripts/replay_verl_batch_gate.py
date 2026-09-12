"""Rehearse the batch gate on a finished round's evidence, without a GPU.

The gate runs inside the Ray trainer, where a rejection is one line in a large
log. Everything it reads is durable -- episodes, bundles, policy artifacts,
`admitted-sequence.json` -- so the whole decision can be replayed in-process.

Usage: PYTHONPATH=<repo> python3 replay_verl_batch_gate.py <round-dir> [attempt]

Prints one line per episode (calls / tool rounds / sequence length / verifier)
and then the batch verdict: certified, or the exact reason it was refused.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

from src.certification import ConsumerProfile, certify_for
from src.contracts.agent_episode import AgentEpisode
from src.contracts.execution_bundle import ExecutionBundle
from src.errors import ContractValidationError
from src.integrations.verl.admission import AdmittedVerlSequence
from src.integrations.verl.manager import CertifiedAgentLoopManager
from src.integrations.verl.sequence import describe_sequence_difference
from src.training.policy_fingerprint import PolicyFingerprint
from src.producers.base import ProducerArtifact


def main() -> int:
    round_dir = sys.argv[1]
    attempt = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    policy = PolicyFingerprint.from_dict(
        json.loads(open(os.path.join(round_dir, "round-policy.json")).read())
    )
    gen = sorted(glob.glob(os.path.join(round_dir, "episodes", "*", "gen-*")))
    if not gen:
        print("no generation directories under", round_dir)
        return 1
    attempt_root = os.path.join(gen[-1], f"attempt-{attempt}")
    episodes = sorted(
        path for path in glob.glob(os.path.join(attempt_root, "*")) if os.path.isdir(path)
    )
    print(f"{len(episodes)} episodes under {attempt_root}")

    from src.integrations.verl.pi_loop import _assemble

    sequences: list[AdmittedVerlSequence] = []
    task_ids: dict[str, str] = {}
    for episode_dir in episodes:
        name = os.path.basename(episode_dir)
        summary = json.loads(open(os.path.join(episode_dir, "summary.json")).read())
        sequence = AdmittedVerlSequence.from_dict(
            json.loads(open(os.path.join(episode_dir, "admitted-sequence.json")).read())
        )
        episode = AgentEpisode.from_dict(
            json.loads(open(os.path.join(episode_dir, "finalized/episode.json")).read())
        )
        bundle = ExecutionBundle.from_dict(
            json.loads(open(os.path.join(episode_dir, "finalized/execution-bundle.json")).read())
        )
        artifact = ProducerArtifact.from_dict(
            json.loads(open(os.path.join(episode_dir, "policy-artifact.json")).read())
        )
        decision = certify_for(
            episode,
            ConsumerProfile.ON_POLICY_RL,
            execution_bundle=bundle,
            policy_artifact=artifact,
            target_policy_fingerprint=policy.checksum(),
        )
        rebuilt = _assemble(Path(episode_dir), sequence.episode_id)
        training = {
            "prompt_ids": list(rebuilt.prompt_ids),
            "response_ids": list(rebuilt.response_ids),
            "loss_mask": list(rebuilt.response_mask),
            "response_logprobs": list(rebuilt.response_logprobs),
        }
        verdict = decision.verdict.value
        note = ""
        if artifact.payload.get("training_sequence") != training:
            note = " ARTIFACT-DRIFT: " + describe_sequence_difference(
                artifact.payload.get("training_sequence"), training
            )
        if verdict != "ELIGIBLE":
            note += f" CERT={verdict}"
        reward = 1.0 if summary["verifier_status"] == "PASSED" else 0.0
        if sequence.reward != reward:
            note += f" REWARD-DRIFT stored={sequence.reward} expected={reward}"
        print(
            f"  {name:34s} calls={sequence.num_model_calls} "
            f"tool_rounds={sequence.num_tool_rounds} len={len(sequence.response_ids)} "
            f"verifier={summary['verifier_status']} reward={sequence.reward}{note}"
        )
        sequences.append(sequence)
        task_ids[sequence.episode_id] = str(summary.get("task_id"))

    rewards = sorted({item.reward for item in sequences})
    print("distinct rewards:", rewards)
    gate = CertifiedAgentLoopManager(policy=policy, minimum_group_size=4)
    try:
        certified = gate.certify_batch(
            sequences, batch_id="replay", task_ids_by_episode=task_ids
        )
    except ContractValidationError as exc:
        print("BATCH REFUSED:", exc)
        return 2
    print(
        f"BATCH CERTIFIED {certified.batch_id} over {len(certified.sequence_checksums)} sequences"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
