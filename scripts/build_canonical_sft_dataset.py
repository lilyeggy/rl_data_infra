#!/usr/bin/env python3
"""Build the canonical generic SFT dataset from migrated canonical episodes.

Consumes ``canonical-teacher/v1/canonical-*.json`` (migration output) and:
- converts each into canonical SFT examples with explicit decision roles;
- runs leak / dedup / verifier / lossy statistics;
- emits a per-role report and a dataset card; writes JSONL.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Mapping

from src.contracts._json import canonical_json_bytes, sha256_json
from src.capture.pi_canonical_adapter import CanonicalEpisode
from src.learning.canonical_sft_dataset import build_canonical_sft_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks-json", type=Path, required=True,
        help="SWE-bench JSON list containing instance_id and problem_statement",
    )
    parser.add_argument(
        "--workspace-root-prefix", type=str,
        help="absolute workspace parent; task suffix is appended before rendering observations",
    )
    args = parser.parse_args()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    episodes: list[CanonicalEpisode] = []
    for path in sorted(glob.glob(str(args.canonical_dir / "canonical-*.json"))):
        episodes.append(CanonicalEpisode.from_dict(json.loads(Path(path).read_text())))

    task_rows = json.loads(args.tasks_json.read_text())
    if isinstance(task_rows, Mapping):
        task_statements = {
            str(task_id): row["problem_statement"]
            for task_id, row in task_rows.items()
            if isinstance(row, Mapping) and row.get("problem_statement")
        }
    elif isinstance(task_rows, list):
        task_statements = {
            str(row.get("instance_id") or row.get("task_id")): row["problem_statement"]
            for row in task_rows
            if isinstance(row, Mapping) and row.get("problem_statement")
        }
    else:
        raise ValueError("--tasks-json must contain an object or a list of task records")

    workspace_roots = {}
    if args.workspace_root_prefix:
        prefix = args.workspace_root_prefix.rstrip("/")
        workspace_roots = {
            ep.episode_id: f"{prefix}/{ep.task_id.split('__', 1)[-1]}"
            for ep in episodes
        }

    examples, report = build_canonical_sft_dataset(
        tuple(episodes),
        task_ids={ep.episode_id: ep.task_id for ep in episodes},
        task_statements=task_statements,
        workspace_roots=workspace_roots,
    )

    lines = [canonical_json_bytes(ex.to_dict()) + b"\n" for ex in examples]
    dataset_bytes = b"".join(lines)
    dataset_sha = sha256_json([json.loads(line) for line in dataset_bytes.splitlines() if line])
    raw_sha = __import__("hashlib", fromlist=["sha256"]).sha256(dataset_bytes).hexdigest()

    (output / "canonical-generic-sft-v1.jsonl").write_bytes(dataset_bytes)
    card = {
        "dataset_name": "canonical-generic-sft-v1",
        "version": "canonical-generic-sft/v1",
        "source": "canonical-teacher/v1 (Pi teacher, 9 TRAIN tasks)",
        "report": report.to_dict(),
        "manifest": {
            "example_count": len(examples),
            "examples_sha256": raw_sha,
            "examples_semantic_checksum": dataset_sha,
            "report_checksum": report.checksum,
        },
        "channels": list(report.channels),
        "policy": "Only certified teacher episodes -> canonical; no Pi-private prompt; "
                  "no host-path leak; assistant-only loss by default.",
    }
    (output / "dataset-card.json").write_bytes(canonical_json_bytes(card) + b"\n")
    print(json.dumps({"examples": len(examples), "report_checksum": report.checksum,
                      "sha256": raw_sha, "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
