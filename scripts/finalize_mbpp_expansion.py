#!/usr/bin/env python3
"""Summarize an MBPP rollout batch without treating infra errors as model failures."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="MBPP expansion")
    args = parser.parse_args()

    summaries = sorted(args.root.glob("*/summary.json"))
    rows = [json.loads(path.read_text()) for path in summaries]
    verdicts = Counter(row.get("sft_verdict") for row in rows)
    statuses = Counter(row.get("verifier_status") for row in rows)
    validity = Counter(row.get("execution_validity") for row in rows)
    calls = [row["model_call_count"] for row in rows if isinstance(row.get("model_call_count"), int)]
    report = {
        "label": args.label,
        "root": str(args.root),
        "tasks": len(rows),
        "sft_eligible": verdicts.get("ELIGIBLE", 0),
        "rejected": verdicts.get("REJECTED", 0),
        "insufficient_evidence": verdicts.get("INSUFFICIENT_EVIDENCE", 0),
        "verifier_status": dict(statuses),
        "execution_validity": dict(validity),
        "mean_model_calls": round(sum(calls) / len(calls), 2) if calls else None,
        "max_model_calls": max(calls) if calls else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    md = [
        f"# {args.label}", "", f"- Tasks: {report['tasks']}",
        f"- SFT eligible: {report['sft_eligible']}",
        f"- Rejected: {report['rejected']}",
        f"- Insufficient evidence: {report['insufficient_evidence']}",
        f"- Mean model calls: {report['mean_model_calls']}",
        f"- Verifier status: `{dict(statuses)}`",
        f"- Execution validity: `{dict(validity)}`", "",
        "This report keeps infrastructure/evidence errors separate from model/verifier failures.",
    ]
    args.output.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
