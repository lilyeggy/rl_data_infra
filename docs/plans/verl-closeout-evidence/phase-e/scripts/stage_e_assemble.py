#!/usr/bin/env python3
"""Stage E assembly: turn one rollout round's episodes into a training batch.

Reads the per-episode native token evidence produced by the rollout driver,
assembles one verified verl sequence per episode (contiguity-checked), admits
them, certifies the batch through CertifiedAgentLoopManager, and writes the
FULL token arrays the trainer consumes:

  round-<k>.json = {
    policy: {...fingerprint...},
    episodes: [{episode_id, reward, prompt_ids, response_ids, response_mask,
                rollout_logprobs, num_model_calls, num_tool_rounds,
                verifier_status, contiguity}],
    certified_batch: {...}
  }

CPU only; no GPU. Read-only with respect to rollout evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/cxr/agentic/code")
sys.path.insert(0, "/home/cxr/verl-closeout/smoke-src")

from src.integrations.verl import (  # noqa: E402
    AdmittedVerlSequence,
    BridgeCallRecord,
    CertifiedAgentLoopManager,
    assemble_episode_sequence,
    build_per_call_segments,
)
from src.training.policy_fingerprint import PolicyFingerprint  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="dir containing episode subdirs")
    parser.add_argument("--round-index", type=int, required=True)
    parser.add_argument("--policy-generation", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--task-id", default="Mbpp/118")
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-group-size", type=int, default=2)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    adapter = Path(args.adapter)
    episodes = sorted(p for p in run_dir.iterdir() if p.is_dir() and (p / "model-evidence.jsonl").exists())
    if not episodes:
        print(f"NO-EPISODES: {run_dir}", flush=True)
        return 10

    policy = PolicyFingerprint(
        provider="verl-vllm",
        model_id="qwen2.5-coder-14b",
        base_model_revision=args.base_model,
        adapter_revision=sha256_file(adapter / "adapter_model.safetensors"),
        tokenizer_revision=sha256_file(adapter / "tokenizer.json"),
        chat_template_checksum=sha256_file(adapter / "chat_template.jinja"),
        tool_schema_checksum="t" * 64,
        sampling_config={"temperature": 1.0, "top_p": 1.0},
        temperature=1.0,
        top_p=1.0,
        policy_generation=args.policy_generation,
    )
    policy_checksum = policy.checksum()

    report: dict = {
        "round_index": args.round_index,
        "policy": {
            "generation": args.policy_generation,
            "checksum": policy_checksum,
            "adapter_revision": policy.adapter_revision,
        },
        "episodes": [],
    }
    admitted: list[AdmittedVerlSequence] = []
    for episode_dir in episodes:
        episode_id = episode_dir.name
        rows = [
            json.loads(line)
            for line in (episode_dir / "model-evidence.jsonl").read_text().splitlines()
            if line.strip()
        ]
        summary = json.loads((episode_dir / "summary.json").read_text())
        verifier_status = summary["verifier_status"]
        entry: dict = {"episode_id": episode_id, "verifier_status": verifier_status}
        try:
            records = [
                BridgeCallRecord(
                    request_id=row["request"]["request_id"],
                    prompt_token_ids=tuple(row["backend"]["prompt_token_ids"]),
                    response_token_ids=tuple(row["backend"]["response_token_ids"]),
                    response_logprobs=tuple(row["backend"]["response_logprobs"]),
                )
                for row in rows
            ]
            segments, frozen_prompt = build_per_call_segments(records)
            sequence = assemble_episode_sequence(
                episode_id=episode_id, prompt_ids=frozen_prompt, per_call_segments=segments
            )
            # Only a completed verifier verdict yields a GRPO reward. Infra
            # errors stay distinct from a valid failure (closeout §3.5).
            if verifier_status not in ("PASSED", "FAILED"):
                raise ValueError(f"non-terminating verifier status: {verifier_status}")
            reward = 1.0 if verifier_status == "PASSED" else 0.0
            admitted.append(
                AdmittedVerlSequence(
                    episode_id=episode_id,
                    group_id=args.group_id,
                    policy_fingerprint=policy_checksum,
                    prompt_ids=sequence.prompt_ids,
                    response_ids=sequence.response_ids,
                    response_mask=sequence.response_mask,
                    response_logprobs=sequence.response_logprobs,
                    reward=reward,
                    num_model_calls=sequence.num_model_calls,
                    num_tool_rounds=sequence.num_tool_rounds,
                    member_id=episode_id,
                    execution_bundle_checksum="0" * 64,
                    policy_artifact_checksum="0" * 64,
                )
            )
            entry.update({
                "reward": reward,
                "num_model_calls": sequence.num_model_calls,
                "num_tool_rounds": sequence.num_tool_rounds,
                "prompt_ids": list(sequence.prompt_ids),
                "response_ids": list(sequence.response_ids),
                "response_mask": list(sequence.response_mask),
                "rollout_logprobs": list(sequence.response_logprobs),
                "contiguity": "OK",
                "trainable_tokens": int(sum(sequence.response_mask)),
            })
        except Exception as exc:  # noqa: BLE001 - record, never fabricate
            entry["error"] = f"{type(exc).__name__}: {exc}"
        report["episodes"].append(entry)

    ok = [e for e in report["episodes"] if "error" not in e]
    report["assembled"] = len(ok)
    report["reward_values"] = sorted({e["verifier_status"] for e in ok})
    report["has_variance"] = len({e["verifier_status"] for e in ok}) > 1
    if admitted:
        manager = CertifiedAgentLoopManager(policy=policy, minimum_group_size=args.minimum_group_size)
        try:
            certified = manager.certify_batch(
                admitted,
                batch_id=f"e-round{args.round_index}-batch",
                task_ids_by_episode={s.episode_id: args.task_id for s in admitted},
            )
            report["manager_certified"] = True
            report["certified_batch"] = certified.to_dict()
        except Exception as exc:  # noqa: BLE001
            report["manager_certified"] = False
            report["manager_error"] = f"{type(exc).__name__}: {exc}"
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))
    for entry in report["episodes"]:
        print(f"  {entry['episode_id']}: {entry.get('verifier_status')} "
              f"reward={entry.get('reward')} calls={entry.get('num_model_calls')} "
              f"tools={entry.get('num_tool_rounds')} trainable={entry.get('trainable_tokens')} "
              f"{entry.get('error', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
