#!/usr/bin/env python3
"""Persistent, restart-safe supervisor for a single-GPU learning loop.

The supervisor is deliberately command-agnostic: each stage declares an
immutable completion artifact and (optionally) the command that creates it.
It never starts two GPU stages concurrently, persists every transition, and
stops fail-closed on a missing artifact or a non-zero command.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def required_json_error(stage: dict[str, Any], artifact: Path) -> str | None:
    """Return a fail-closed reason when a stage's JSON gate is not satisfied."""

    required = stage.get("required_json")
    if required is None:
        return None
    if not isinstance(required, dict) or not required:
        return "required_json must be a non-empty object"
    try:
        value = json.loads(artifact.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return f"cannot load gated artifact {artifact}: {exc}"
    if not isinstance(value, dict):
        return f"gated artifact is not a JSON object: {artifact}"
    mismatches = [
        f"{key} expected {expected!r}, got {value.get(key)!r}"
        for key, expected in required.items()
        if value.get(key) != expected
    ]
    return "; ".join(mismatches) if mismatches else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    stages = config["stages"]
    state = json.loads(args.state.read_text()) if args.state.exists() else {
        "schema_version": "learning-loop-state/v1", "started_at": _now(), "stages": []
    }
    for stage in stages:
        name = stage["name"]
        artifact = Path(stage["completion_artifact"])
        latest = next(
            (entry for entry in reversed(state["stages"]) if entry["name"] == name),
            None,
        )
        if artifact.is_file():
            gate_error = required_json_error(stage, artifact)
            if gate_error is not None:
                if latest is None or latest.get("status") != "REJECTED" or latest.get("reason") != gate_error:
                    state["stages"].append({
                        "name": name,
                        "status": "REJECTED",
                        "observed_at": _now(),
                        "artifact": str(artifact),
                        "reason": gate_error,
                    })
                state["status"] = "REJECTED"
                state["stopped_at"] = state.get("stopped_at", _now())
                state["reason"] = gate_error
                _write(args.state, state)
                return 3
            if latest is not None and latest.get("status") == "COMPLETED":
                continue
            state["stages"].append({"name": name, "status": "COMPLETED", "observed_at": _now(), "artifact": str(artifact)})
            state.pop("status", None)
            state.pop("stopped_at", None)
            state.pop("reason", None)
            _write(args.state, state)
            continue
        if latest is not None and latest.get("status") == "COMPLETED":
            state["failed_at"] = _now()
            state["failure"] = f"completed artifact vanished: {artifact}"
            _write(args.state, state)
            return 2
        command = stage.get("command")
        if not command:
            if latest is None:
                state["stages"].append({"name": name, "status": "WAITING", "observed_at": _now(), "artifact": str(artifact)})
            else:
                latest["status"] = "WAITING"
                latest["observed_at"] = _now()
            _write(args.state, state); return 0
        state["stages"].append({"name": name, "status": "RUNNING", "started_at": _now(), "artifact": str(artifact), "command": command})
        _write(args.state, state)
        result = subprocess.run(command, shell=True, check=False)
        state["stages"][-1]["finished_at"] = _now()
        if result.returncode or not artifact.is_file():
            state["stages"][-1]["status"] = "FAILED"
            state["stages"][-1]["returncode"] = result.returncode
            _write(args.state, state); return 1
        state["stages"][-1]["status"] = "COMPLETED"
        _write(args.state, state)
    state["completed_at"] = _now(); state["status"] = "COMPLETED"; _write(args.state, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
