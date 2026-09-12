"""Report one episode's event sequence, to see how far a real episode got.

Read-only. Prints every recorded event with its status and the tool facts it
carries, so "the tool round never happened" can be distinguished from "the tool
round happened and the episode was cut short by something else".

Usage: python3 probe_episode_events.py <episode-dir> [<episode-dir> ...]
"""

from __future__ import annotations

import json
import os
import sys

INTERESTING = {
    "MODEL_REQUEST",
    "MODEL_RESPONSE",
    "TOOL_CALL",
    "TOOL_RESULT",
    "TERMINATION_DECIDED",
    "EPISODE_FINISHED",
    "SANDBOX_FINISHED",
    "VERIFICATION_FINISHED",
}


def main() -> int:
    for directory in sys.argv[1:]:
        print("===", os.path.basename(directory))
        path = os.path.join(directory, "raw-events.jsonl")
        rows = [json.loads(line) for line in open(path) if line.strip()]
        for row in rows:
            event_type = row.get("event_type")
            if event_type not in INTERESTING:
                continue
            attributes = row.get("attributes") or {}
            detail = ""
            if event_type in {"TOOL_CALL", "TOOL_RESULT"}:
                detail = json.dumps(attributes)[:240]
            elif event_type in {"TERMINATION_DECIDED", "EPISODE_FINISHED"}:
                detail = json.dumps(attributes)[:240]
            print("  {:22s} {:10s} {}".format(event_type, str(row.get("status")), detail))
        summary_path = os.path.join(directory, "summary.json")
        if os.path.exists(summary_path):
            summary = json.loads(open(summary_path).read())
            print("  summary:", json.dumps({
                key: summary.get(key) for key in (
                    "verifier_status", "execution_validity", "on_policy_rl_verdict",
                    "model_call_count", "tool_call_count",
                )
            }))
        else:
            print("  (no summary.json: the episode never reached the finalizer)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
