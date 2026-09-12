#!/usr/bin/env python3
"""Stage D final assembly: turn the dstage13 certified batch into a verl batch.

For each of the 4 certified episodes:
  model-evidence.jsonl (native per-call tokens/logprobs)
    -> BridgeCallRecord list -> build_per_call_segments (contiguity check)
    -> assemble_episode_sequence (single sequence per episode, mask 0 on obs)
  -> attach as policy artifact 'verl_sequence'
Then admit via verl admission and certify the batch via the Manager.

Runs on the CPU side; no GPU. Read-only with respect to GPU evidence.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SMOKE = Path("/home/cxr/verl-closeout/smoke")
sys.path.insert(0, "/home/cxr/agentic/code")
sys.path.insert(0, "/home/cxr/verl-closeout/smoke-src")

from src.integrations.verl import (  # noqa: E402
    BridgeCallRecord,
    CertifiedAgentLoopManager,
    assemble_episode_sequence,
    build_per_call_segments,
)
from src.training.policy_fingerprint import PolicyFingerprint  # noqa: E402

RUN_ROOT = Path(os.environ.get("D_RUN_ROOT", "/home/cxr/verl-closeout/stage-d/run32"))
EPISODES = sorted(p for p in RUN_ROOT.glob("dstage*") if p.is_dir())


def main() -> int:
    policy = PolicyFingerprint(
        provider="verl-vllm", model_id="qwen2.5-coder-14b",
        base_model_revision="/home/cxr/agentic/models/qwen2.5-coder-14b-base",
        adapter_revision="6db6a40c881b1896a061a0a51c36112a482f7dd8a659294c2471c248eb2458ae",
        tokenizer_revision="3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8",
        chat_template_checksum="44d5f08f3f72b837eaad09f13a54c1f9f4eb58d75240334548b7fd52a5437fa5",
        tool_schema_checksum="t" * 64,
        sampling_config={"temperature": 1.0, "top_p": 1.0},
        temperature=1.0, top_p=1.0, policy_generation="P0",
    )
    report = {"policy_checksum": policy.checksum(), "episodes": []}
    sequences = {}
    for episode_dir in EPISODES:
        evidence_path = episode_dir / "model-evidence.jsonl"
        rows = [json.loads(line) for line in evidence_path.read_text().splitlines() if line.strip()]
        summary = json.loads((episode_dir / "summary.json").read_text())
        records = [
            BridgeCallRecord(
                request_id=row["request"]["request_id"],
                prompt_token_ids=tuple(row["backend"]["prompt_token_ids"]),
                response_token_ids=tuple(row["backend"]["response_token_ids"]),
                response_logprobs=tuple(row["backend"]["response_logprobs"]),
            )
            for row in rows
        ]
        episode_id = episode_dir.name
        try:
            segments, frozen_prompt = build_per_call_segments(records)
            sequence = assemble_episode_sequence(
                episode_id=episode_id, prompt_ids=frozen_prompt,
                per_call_segments=segments,
            )
            sequences[episode_id] = sequence
            report["episodes"].append({
                "episode_id": episode_id,
                "verifier_status": summary["verifier_status"],
                "on_policy_rl_verdict": summary["on_policy_rl_verdict"],
                "num_model_calls": sequence.num_model_calls,
                "num_tool_rounds": sequence.num_tool_rounds,
                "prompt_len": len(sequence.prompt_ids),
                "response_len": len(sequence.response_ids),
                "trainable_tokens": sum(sequence.response_mask),
                "contiguity": "OK",
            })
        except Exception as exc:  # noqa: BLE001 - record, never fabricate
            report["episodes"].append({
                "episode_id": episode_id,
                "verifier_status": summary["verifier_status"],
                "error": f"{type(exc).__name__}: {exc}",
            })
    ok = [e for e in report["episodes"] if "error" not in e]
    report["assembled"] = len(ok)
    report["reward_values"] = sorted({e["verifier_status"] for e in ok})
    report["has_variance"] = len({e["verifier_status"] for e in ok}) > 1
    # Manager-level batch certification over the assembled sequences.
    if sequences:
        from src.integrations.verl import AdmittedVerlSequence

        admitted = [
            AdmittedVerlSequence(
                episode_id=s.episode_id, group_id="mbpp118-group",
                policy_fingerprint=policy.checksum(),
                prompt_ids=s.prompt_ids, response_ids=s.response_ids,
                response_mask=s.response_mask, response_logprobs=s.response_logprobs,
                reward=1.0 if next(
                    e["verifier_status"] for e in ok if e["episode_id"] == s.episode_id
                ) == "PASSED" else 0.0,
                num_model_calls=s.num_model_calls, num_tool_rounds=s.num_tool_rounds,
                member_id=s.episode_id, execution_bundle_checksum="0" * 64,
                policy_artifact_checksum="0" * 64,
            )
            for s in sequences.values()
        ]
        manager = CertifiedAgentLoopManager(policy=policy, minimum_group_size=2)
        try:
            certified = manager.certify_batch(
                admitted, batch_id="mbpp118-batch-1",
                task_ids_by_episode={s.episode_id: "Mbpp/118" for s in admitted},
            )
            report["manager_certified"] = True
            report["certified_batch"] = certified.to_dict()
        except Exception as exc:  # noqa: BLE001
            report["manager_certified"] = False
            report["manager_error"] = f"{type(exc).__name__}: {exc}"
    out = RUN_ROOT / "verl-assembly.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
