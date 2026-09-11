"""Timeout and cancellation cleanup for Pi/model/verifier subprocesses.

Only cleans up processes this run registered. Never touches shared Ray
instances, existing training processes, or GPUs outside this run.
"""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

RUN_CLEANUP_VERSION = "verl-run-cleanup/v1"


@dataclass(slots=True, kw_only=True)
class RegisteredProcess:
    label: str
    process: subprocess.Popen[str]
    owned: bool = True


@dataclass(slots=True, kw_only=True)
class RunCleanup:
    """Track and terminate only this run's child processes."""

    registrations: list[RegisteredProcess] = field(default_factory=list)

    def register(self, label: str, process: subprocess.Popen[str]) -> None:
        self.registrations.append(RegisteredProcess(label=label, process=process))

    def terminate_all(self, *, grace_seconds: float = 5.0) -> dict[str, str]:
        """Terminate registered processes; return label -> outcome."""
        outcomes: dict[str, str] = {}
        for registration in self.registrations:
            process = registration.process
            if process.poll() is not None:
                outcomes[registration.label] = f"already-exited:{process.returncode}"
                continue
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:  # pragma: no cover - CI runs on POSIX
                    process.terminate()
            except (ProcessLookupError, PermissionError, OSError) as exc:
                outcomes[registration.label] = f"terminate-failed:{exc}"
                continue
            try:
                process.wait(timeout=grace_seconds)
                outcomes[registration.label] = f"terminated:{process.returncode}"
            except subprocess.TimeoutExpired:
                try:
                    if os.name == "posix":
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    else:  # pragma: no cover - CI runs on POSIX
                        process.kill()
                except (ProcessLookupError, PermissionError, OSError) as exc:
                    outcomes[registration.label] = f"kill-failed:{exc}"
                    continue
                process.wait()
                outcomes[registration.label] = f"killed:{process.returncode}"
        self.registrations.clear()
        return outcomes


def popen_process_group(
    command: Sequence[str],
    *,
    cwd: str,
    env: dict[str, str] | None,
    extra: dict[str, Any] | None = None,
) -> subprocess.Popen[str]:
    """Start a child in its own process group for scoped termination."""
    kwargs: dict[str, Any] = {
        "cwd": cwd,
        "env": env,
        "stdin": subprocess.DEVNULL,
        "capture_output": True,
        "text": True,
    }
    if extra:
        kwargs.update(extra)
    if os.name == "posix":
        kwargs["start_new_session"] = True
    return subprocess.Popen(list(command), **kwargs)
