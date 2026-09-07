"""HISTORICAL EXAMPLE: deterministic V2.1 recovery-policy replay package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.capture.pi_adapter import (
    PiJsonAdapter,
    PiOutcomeDeclaration,
    PiRunConfig,
    read_pi_ndjson,
)
from src.contracts._json import sha256_json
from src.contracts.agent_episode import (
    EpisodeVerifierStatus,
    ExecutionValidity,
    TaskStatus,
)
from src.recovery.file_not_found_policy import FILE_NOT_FOUND_POLICY_VERSION
from src.recovery.replay import replay_file_not_found_recovery


TASKS = (1, 2, 3)
VARIANTS = ("control", "candidate")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def generate_v21_recovery_replay(
    output_dir: str | Path,
    *,
    fixture_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Replay the V2.1 policy against sanitized V2 Pi traces.

    The package is derived planning evidence.  It deliberately does not append
    synthetic decision events to the V2 raw event stream.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fixtures = (
        Path(fixture_dir)
        if fixture_dir is not None
        else Path(__file__).parents[2] / "tests" / "fixtures" / "pi" / "v2-real"
    )
    adapter = PiJsonAdapter()
    reports: dict[str, Any] = {}
    source_checksums: dict[str, str] = {}
    adapter_issues: dict[str, list[dict[str, Any]]] = {}

    for task_index in TASKS:
        for variant in VARIANTS:
            key = f"task-{task_index}-{variant}"
            fixture = fixtures / f"{key}.ndjson"
            records, issues = read_pi_ndjson(fixture.read_text())
            result = adapter.convert(
                records,
                run_id=f"run-v21-{variant}",
                episode_id=f"episode-v21-{key}",
                trace_id=f"trace-v21-{key}",
                config=PiRunConfig(),
                declared_outcome=PiOutcomeDeclaration(
                    task_status=TaskStatus.UNKNOWN,
                    execution_validity=ExecutionValidity.UNKNOWN,
                    verifier_status=EpisodeVerifierStatus.UNKNOWN,
                ),
                source_issues=issues,
                normalize_tool_errors=True,
            )
            report = replay_file_not_found_recovery(result.events)
            reports[key] = report.to_dict()
            source_checksums[key] = result.source_checksum
            adapter_issues[key] = [
                {
                    "code": issue.code.value,
                    "message": issue.message,
                    "source_record_id": issue.source_record_id,
                }
                for issue in result.issues
            ]

    _write_json(output / "recovery-replay.json", reports)
    _write_json(
        output / "capture-evidence.json",
        {
            "source_checksums": source_checksums,
            "adapter_issues": adapter_issues,
            "decision_capture": "NOT_OBSERVABLE",
            "important_boundary": (
                "replay decisions are derived policy plans; they are not synthetic "
                "HARNESS_DECISION raw events"
            ),
        },
    )
    summary = {
        "release": "v2.1",
        "scope": "bounded FILE_NOT_FOUND policy replay and decision-observability preparation",
        "policy_version": FILE_NOT_FOUND_POLICY_VERSION,
        "episode_count": len(reports),
        "decision_capture": "NOT_OBSERVABLE",
        "raw_events_modified": False,
        "candidate_replay_steps_match": all(
            all(
                step["action_alignment"] == "MATCH"
                for step in reports[f"task-{task}-candidate"]["steps"]
            )
            for task in TASKS
        ),
        "control_replay_has_no_claimed_decision": all(
            "HARNESS_DECISION" not in json.dumps(reports[f"task-{task}-control"])
            for task in TASKS
        ),
        "claim_boundary": (
            "policy replay validates deterministic action planning against sanitized traces; "
            "runtime Harness decisions require Hook-enabled capture"
        ),
    }
    _write_json(output / "summary.json", summary)
    _write_json(
        output / "artifact-manifest.json",
        {
            "schema_version": "v2.1-artifact-manifest/v1",
            "files": sorted(
                path.name
                for path in output.iterdir()
                if path.is_file() and path.name != "artifact-manifest.json"
            ),
            "summary_checksum": sha256_json(summary),
        },
    )
    return summary
