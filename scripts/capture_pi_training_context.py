#!/usr/bin/env python3
"""Freeze the real Pi system prompt and tool contract used for SFT rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.contracts._json import canonical_json_bytes, sha256_json


SCHEMA_VERSION = "pi-training-context/v1"


def _first_request(path: Path) -> dict[str, Any]:
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        request = row.get("request")
        if isinstance(request, dict):
            return request
    raise ValueError(f"no request found in model evidence: {path}")


def capture(entries: list[str]) -> dict[str, Any]:
    contexts: dict[str, Any] = {}
    common_tools: list[dict[str, Any]] | None = None
    for entry in entries:
        if "=" not in entry:
            raise ValueError("--evidence must be TASK_ID=MODEL_EVIDENCE_JSONL")
        task_id, raw_path = entry.split("=", 1)
        if not task_id or task_id in contexts:
            raise ValueError(f"invalid or duplicate task id: {task_id!r}")
        path = Path(raw_path)
        request = _first_request(path)
        messages = request.get("messages")
        tools = request.get("tools")
        if not isinstance(messages, list) or not isinstance(tools, list) or not tools:
            raise ValueError(f"request lacks messages/tools: {path}")
        system = [item for item in messages if item.get("role") == "system"]
        if len(system) != 1 or not isinstance(system[0].get("content"), str):
            raise ValueError(f"expected exactly one text system message: {path}")
        if common_tools is None:
            common_tools = tools
        elif sha256_json(common_tools) != sha256_json(tools):
            raise ValueError(f"Pi tool contract differs across evidence files: {path}")
        contexts[task_id] = {
            "system_message": system[0],
            "source_evidence": str(path.resolve()),
            "source_evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    if not contexts or common_tools is None:
        raise ValueError("at least one --evidence entry is required")
    return {
        "schema_version": SCHEMA_VERSION,
        "tools": common_tools,
        "tools_checksum": sha256_json(common_tools),
        "task_contexts": dict(sorted(contexts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence", action="append", default=[],
        metavar="TASK_ID=MODEL_EVIDENCE_JSONL",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = capture(args.evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(result) + b"\n")
    print(json.dumps({
        "output": str(args.output),
        "task_count": len(result["task_contexts"]),
        "tool_count": len(result["tools"]),
        "tools_checksum": result["tools_checksum"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
