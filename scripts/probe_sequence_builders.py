"""Diff the two sequence builders that must agree on one episode.

`policy-artifact.json` carries a `training_sequence` built from the in-memory
evidence, and the batch gate re-derives the same sequence by re-reading
`model-evidence.jsonl`. They disagree on real episodes, and this prints exactly
where, so the fix targets the cause instead of relaxing the check.

Usage: PYTHONPATH=<repo> python3 probe_sequence_builders.py <episode-dir>
"""

from __future__ import annotations

import json
import os
import sys

from src.integrations.verl.bridge import BridgeCallRecord, build_per_call_segments
from src.integrations.verl.sequence import assemble_episode_sequence


def rows_of(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def record_from(row: dict, index: int) -> BridgeCallRecord:
    backend = row["backend"]
    return BridgeCallRecord(
        request_id=row["request"]["request_id"],
        prompt_token_ids=tuple(backend["prompt_token_ids"]),
        response_token_ids=tuple(backend["response_token_ids"]),
        response_logprobs=tuple(backend["response_logprobs"]),
    )


def as_training_dict(sequence) -> dict:
    return {
        "prompt_ids": list(sequence.prompt_ids),
        "response_ids": list(sequence.response_ids),
        "loss_mask": list(sequence.response_mask),
        "response_logprobs": list(sequence.response_logprobs),
    }


def main() -> int:
    directory = sys.argv[1]
    evidence = rows_of(os.path.join(directory, "model-evidence.jsonl"))
    artifact = json.loads(open(os.path.join(directory, "policy-artifact.json")).read())
    stored = artifact["payload"].get("training_sequence")

    # The in-memory builder receives exactly these rows, in this order.
    records = [record_from(row, index) for index, row in enumerate(evidence)]
    segments, prompt = build_per_call_segments(records)
    rebuilt = assemble_episode_sequence(
        episode_id=artifact["identity"]["episode_id"],
        prompt_ids=prompt,
        per_call_segments=segments,
    )
    fresh = as_training_dict(rebuilt)

    print("evidence rows:", len(evidence))
    print("stored keys:", sorted(stored) if isinstance(stored, dict) else stored)
    if not isinstance(stored, dict):
        print("no stored training_sequence")
        return 1
    for key in sorted(fresh):
        left, right = stored.get(key), fresh[key]
        if left == right:
            print(f"  {key}: identical (len={len(right)})")
            continue
        print(f"  {key}: DIFFER stored_len={len(left)} fresh_len={len(right)}")
        for index, (a, b) in enumerate(zip(left, right)):
            if a != b:
                print(f"    first difference at {index}: stored={a!r} fresh={b!r}")
                break
    # Where the arrays first diverge tells which call is responsible.
    total = sum(len(row["backend"]["response_token_ids"] or []) for row in evidence)
    prompt_lens = [len(row["backend"]["prompt_token_ids"] or []) for row in evidence]
    print("per-call response lens:", [len(row["backend"]["response_token_ids"] or []) for row in evidence])
    print("per-call prompt lens:", prompt_lens)
    print("sum(response lens):", total, "fresh response len:", len(fresh["response_ids"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
