"""Compare each episode's evidence rows with the call count its own summary claims.

A row count above `model_call_count` means `model-evidence.jsonl` gained a write
after the episode was summarised, which is exactly the window in which the
artifact's stored sequence and the file the batch gate re-reads can disagree.

Usage: python3 probe_evidence_row_race.py <attempt-dir>
"""

from __future__ import annotations

import json
import os
import sys

TOOLS = ()


def main() -> int:
    base = sys.argv[1]
    mismatched = 0
    for episode in sorted(os.listdir(base)):
        directory = os.path.join(base, episode)
        evidence = os.path.join(directory, "model-evidence.jsonl")
        summary = os.path.join(directory, "summary.json")
        turns = os.path.join(directory, "engine-closeout.jsonl")
        if not os.path.exists(evidence):
            print(f"{episode}: NO model-evidence.jsonl")
            continue
        rows = [line for line in open(evidence) if line.strip()]
        calls = None
        if os.path.exists(summary):
            calls = json.loads(open(summary).read()).get("model_call_count")
        verdict = "ok"
        if calls is not None and calls != len(rows):
            verdict = "RACE: evidence has more rows than the summary's call count"
            mismatched += 1
        print(
            f"{episode} evidence_rows={len(rows)} summary_calls={calls} "
            f"closeout={os.path.exists(turns)} -> {verdict}"
        )
    print(f"episodes with a row/call mismatch: {mismatched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
