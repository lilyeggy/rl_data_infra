"""Minimal subprocess boundary for a fixed-model Pi capture.

The runner does not interpret task success and never invokes a shell.  Its
output must pass through :mod:`src.capture.pi_adapter` and an external verifier.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.capture.pi_adapter import PiRunConfig, read_pi_ndjson
from src.errors import AdapterIssue


@dataclass(frozen=True, slots=True)
class PiProcessCapture:
    command: tuple[str, ...]
    cwd: str
    returncode: int
    stdout: str
    stderr: str
    record_count: int
    issues: tuple[AdapterIssue, ...]
    final_stop_reason: str | None
    backend_error_messages: tuple[str, ...]

    @property
    def protocol_settled(self) -> bool:
        return self.final_stop_reason == "stop" and not self.backend_error_messages


def run_pi_process(
    *,
    config: PiRunConfig,
    prompt: str,
    cwd: str | Path,
    timeout_seconds: float = 300,
) -> PiProcessCapture:
    command = config.command(prompt)
    completed = subprocess.run(
        command,
        cwd=Path(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
        shell=False,
    )
    records, issues = read_pi_ndjson(completed.stdout)
    stop_reasons: list[str] = []
    backend_errors: list[str] = []
    for record in records:
        if record.get("type") != "message_end":
            continue
        message = record.get("message")
        if not isinstance(message, dict) and not hasattr(message, "get"):
            continue
        if message.get("role") != "assistant":
            continue
        reason = message.get("stopReason")
        if isinstance(reason, str):
            stop_reasons.append(reason)
        if reason == "error":
            backend_errors.append(str(message.get("errorMessage") or "Pi backend error"))
    final_reason = stop_reasons[-1] if stop_reasons else None
    # Earlier retryable errors do not invalidate a later settled response, but
    # they remain observable evidence in raw records and the canonical trace.
    terminal_errors = tuple(backend_errors) if final_reason == "error" else ()
    return PiProcessCapture(
        command=command,
        cwd=str(Path(cwd).resolve()),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        record_count=len(records),
        issues=issues,
        final_stop_reason=final_reason,
        backend_error_messages=terminal_errors,
    )
