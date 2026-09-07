#!/usr/bin/env python3
"""Run Pi in a sandbox and forward observable tool facts to the Data Plane."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _emit(event: dict[str, Any]) -> None:
    request = urllib.request.Request(
        os.environ["AGENT_TRACE_URL"],
        data=json.dumps(event, ensure_ascii=False).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {os.environ['AGENT_TRACE_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        response.read()


def _tool_events(record: dict[str, Any], pending: dict[str, str]) -> list[dict[str, Any]]:
    if record.get("type") != "message_end":
        return []
    message = record.get("message")
    if not isinstance(message, dict):
        return []
    events: list[dict[str, Any]] = []
    if message.get("role") == "assistant":
        for item in message.get("content") or []:
            if not isinstance(item, dict) or item.get("type") != "toolCall":
                continue
            call_id = str(item.get("id") or f"tool-{len(pending)}")
            tool_name = str(item.get("name") or "unknown")
            pending[call_id] = tool_name
            events.append(
                {
                    "event_type": "TOOL_CALL",
                    "component": "TOOL",
                    "status": "STARTED",
                    "span_id": f"span-pi-tool-{call_id}",
                    "parent_span_id": None,
                    "attributes": {
                        "tool_name": tool_name,
                        "arguments": item.get("arguments") or {},
                        "source": "pi-json-wrapper/v1",
                    },
                    "artifact_refs": [],
                    "attempt": int(os.environ["AGENT_ATTEMPT_ID"]),
                }
            )
    elif message.get("role") == "toolResult":
        call_id = str(message.get("toolCallId") or "unknown")
        tool_name = str(message.get("toolName") or pending.get(call_id) or "unknown")
        failed = bool(message.get("isError"))
        events.append(
            {
                "event_type": "TOOL_RESULT",
                "component": "TOOL",
                "status": "FAILED" if failed else "SUCCEEDED",
                "span_id": f"span-pi-tool-{call_id}",
                "parent_span_id": None,
                "attributes": {
                    "tool_name": tool_name,
                    "is_error": failed,
                    "content": message.get("content") or [],
                    "source": "pi-json-wrapper/v1",
                },
                "artifact_refs": [],
                "attempt": int(os.environ["AGENT_ATTEMPT_ID"]),
            }
        )
        pending.pop(call_id, None)
    return events


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--model", default="qwen2.5-coder-14b-instruct")
    args = parser.parse_args()

    temporary_home = Path(tempfile.mkdtemp(prefix="pi-data-plane-", dir="/tmp"))
    models = temporary_home / ".pi" / "agent" / "models.json"
    models.parent.mkdir(parents=True)
    models.write_text(
        json.dumps(
            {
                "providers": {
                    "data-plane-vllm": {
                        "baseUrl": f"{os.environ['OPENAI_BASE_URL'].rstrip('/')}/v1",
                        "api": "openai-completions",
                        "apiKey": "$OPENAI_API_KEY",
                        "compat": {
                            "supportsStrictMode": False,
                            "supportsStore": False,
                            "maxTokensField": "max_tokens",
                        },
                        "models": [
                            {
                                "id": args.model,
                                "contextWindow": 32768,
                                "maxTokens": 8192,
                                "compat": {
                                    "supportsStrictMode": False,
                                    "supportsStore": False,
                                    "maxTokensField": "max_tokens",
                                },
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["HOME"] = str(temporary_home)
    command = [
        "pi",
        "--provider",
        "data-plane-vllm",
        "--model",
        args.model,
        "--mode",
        "json",
        "--print",
        "--no-session",
        "--thinking",
        "minimal",
        args.prompt,
    ]
    process = subprocess.Popen(
        command,
        cwd="/workspace",
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
    )
    assert process.stdout is not None
    pending: dict[str, str] = {}
    trace_failed = False
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        try:
            record = json.loads(line)
            if isinstance(record, dict):
                for event in _tool_events(record, pending):
                    _emit(event)
        except (json.JSONDecodeError, OSError, urllib.error.URLError, KeyError) as exc:
            trace_failed = True
            print(f"trace forwarding failed: {exc}", file=sys.stderr)
    returncode = process.wait()
    if pending:
        print(f"unsettled tool calls: {sorted(pending)}", file=sys.stderr)
        trace_failed = True
    return returncode if returncode != 0 else (70 if trace_failed else 0)


if __name__ == "__main__":
    raise SystemExit(main())
