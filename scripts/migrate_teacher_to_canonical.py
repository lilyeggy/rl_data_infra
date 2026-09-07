#!/usr/bin/env python3
"""Migrate certified Teacher episodes into harness-neutral canonical episodes.

Reads each finalized ``episode.json`` (produced by the Pi host pipeline) and
re-labels every tool call/result into a ``CanonicalAction``. Raw Pi message data
stays as evidence (original ``arguments``/``observation`` + original TraceEvent
ids in ``evidence_event_ids``). A migration manifest records before/after
checksums, per-episode quarantine reasons, and the exact renderer version.

Nothing is silently dropped: unmappable Pi tools or structurally broken pairs
go to ``quarantine`` in the manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.contracts._json import canonical_json_bytes, sha256_json
from src.contracts.agent_episode import AgentEpisode
from src.capture.pi_canonical_adapter import (
    CANONICAL_EPISODE_SCHEMA_VERSION,
    CanonicalEpisode,
    convert_episode_to_canonical,
)

MIGRATION_VERSION = "pi-canonical-migration/v1"


def _load_workspace(run: Path | dict[str, Any]) -> str | None:
    """Read the workspace root from a launch-plan.json inside a run dir, or
    (when given an episode dict) from an explicit environment field."""

    if isinstance(run, Path):
        plan = run / "launch-plan.json"
        if plan.exists():
            try:
                value = json.loads(plan.read_text()).get("workspace")
                if isinstance(value, str) and value.strip():
                    return value
            except json.JSONDecodeError:
                pass
        return None
    env = run.get("environment_manifest") or {}
    return env.get("workspace") or env.get("workspace_root")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--episode-rel", default="finalized/episode.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": MIGRATION_VERSION,
        "canonical_episode_schema_version": CANONICAL_EPISODE_SCHEMA_VERSION,
        "renderer": "src.capture.pi_canonical_adapter convert_episode_to_canonical",
        "episodes": [],
        "quarantine": [],
    }

    for run_dir in args.run_dirs:
        run = Path(run_dir)
        ep_path = run / args.episode_rel
        if not ep_path.exists():
            manifest["quarantine"].append(
                {"run_dir": str(run), "reason": "missing finalized episode.json"}
            )
            continue
        episode_dict = json.loads(ep_path.read_text())
        episode = AgentEpisode.from_dict(episode_dict)
        before_checksum = episode.checksum
        # Authoritative workspace root from the run's launch plan (file-level
        # cwd), falling back to inference only for runs without a launch plan.
        workspace_root = _load_workspace(run)
        if workspace_root is None:
            workspace_root = _load_workspace(episode_dict)
        try:
            canonical, issues = convert_episode_to_canonical(
                episode, workspace_root=workspace_root
            )
        except Exception as exc:  # noqa: BLE001 - quarantine rather than abort
            manifest["quarantine"].append(
                {
                    "run_dir": str(run),
                    "episode_id": episode.episode_id,
                    "reason": f"conversion failed: {type(exc).__name__}: {exc}",
                }
            )
            continue

        record = {
            "task_id": episode.task_id,
            "episode_id": episode.episode_id,
            "run_dir": str(run),
            "before_checksum": before_checksum,
            "after_checksum": canonical.checksum,
            "action_count": len(canonical.actions),
            "lossy_count": sum(1 for a in canonical.actions if a.lossy),
            "issues": [issue.to_dict() for issue in issues],
            "verifier_status": canonical.verifier_status,
        }
        manifest["episodes"].append(record)

        rel_episode = output / f"canonical-{episode.episode_id}.json"
        rel_episode.write_bytes(canonical_json_bytes(canonical.to_dict()) + b"\n")

        # Per-episode record with the same content as canonical for traceability
        # + the source run path.
        detail = output / f"manifest-{episode.episode_id}.json"
        detail.write_bytes(canonical_json_bytes(record) + b"\n")

    manifest["episodes_count"] = len(manifest["episodes"])
    manifest["quarantine_count"] = len(manifest["quarantine"])
    manifest["checksum"] = sha256_json(manifest)
    (output / "migration-manifest.json").write_bytes(
        canonical_json_bytes(manifest) + b"\n"
    )
    print(json.dumps(
        {
            "migrated": manifest["episodes_count"],
            "quarantined": manifest["quarantine_count"],
            "output": str(output),
            "manifest_checksum": manifest["checksum"],
        },
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())