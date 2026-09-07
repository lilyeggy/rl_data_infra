#!/usr/bin/env python3
"""Compare fixed Harness policies using only finalized certified run bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.capture.model_proxy import ModelCallEvidence, ModelEvidenceCapability
from src.contracts._json import sha256_json


def load_runs(root: Path, policy: str) -> list[dict[str, object]]:
    records = []
    marker = f"-{policy}-"
    for summary_path in sorted(root.glob("*/summary.json")):
        if marker not in summary_path.parent.name:
            continue
        summary = json.loads(summary_path.read_text())
        episode = json.loads((summary_path.parent / "finalized" / "episode.json").read_text())
        bundle = json.loads((summary_path.parent / "finalized" / "execution-bundle.json").read_text())
        evidence_path = summary_path.parent / "model-evidence.jsonl"
        evidence = tuple(
            ModelCallEvidence.from_dict(json.loads(line))
            for line in evidence_path.read_text().splitlines()
            if line
        )
        required = frozenset({
            ModelEvidenceCapability.TOKEN_IDS,
            ModelEvidenceCapability.BEHAVIOR_LOGPROBS,
            ModelEvidenceCapability.POLICY_VERSION,
        })
        # Audit the immutable raw evidence itself. This is intentionally more
        # stringent than trusting a producer's high-level capability claim.
        if not evidence or not all(item.backend.status_code < 400 and required.issubset(item.capabilities) for item in evidence):
            continue
        records.append({"run_id": summary["run_id"], "task_id": summary["task_id"],
                        "valid": episode["outcome"]["execution_validity"] == "VALID",
                        "resolved": episode["outcome"]["verifier_status"] == "PASSED",
                        "integrity": episode["integrity"]["state"], "bundle_checksum": sha256_json(bundle)})
    return records


def metrics(records: list[dict[str, object]]) -> dict[str, object]:
    valid = [item for item in records if item["valid"] and item["integrity"] == "COMPLETE"]
    resolved = [item for item in valid if item["resolved"]]
    return {"attempts": len(records), "complete_valid_attempts": len(valid), "resolved": len(resolved),
            "solve_rate": len(resolved) / len(valid) if valid else None,
            "runs": records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = metrics(load_runs(args.runs_root, args.baseline))
    candidate = metrics(load_runs(args.runs_root, args.candidate))
    comparable = min(baseline["complete_valid_attempts"], candidate["complete_valid_attempts"]) >= 3
    if not comparable:
        decision, reason = "INCONCLUSIVE", "promotion requires at least three COMPLETE and VALID finalized runs per policy"
    elif candidate["solve_rate"] > baseline["solve_rate"]:
        decision, reason = "ACCEPT", "candidate improved solve rate on the fixed isolated verifier"
    else:
        decision, reason = "REJECT", "candidate did not improve solve rate on the fixed isolated verifier"
    report = {"schema_version": "harness-regression-report/v1", "evaluator": "selected-swebench-isolated-test-patch",
              "baseline_policy": args.baseline, "candidate_policy": args.candidate, "baseline": baseline,
              "candidate": candidate, "decision": decision, "reason": reason,
              "limitations": ["This compares a selected fixed task set, not a general SWE-bench score.",
                              "No policy is promoted without a subsequent larger held-out evaluation."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if decision != "INCONCLUSIVE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
