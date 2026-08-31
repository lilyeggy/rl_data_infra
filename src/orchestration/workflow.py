"""Persistent compare-and-swap state and non-executing command plans."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from src.contracts._json import sha256_json
from src.errors import ContractValidationError
from src.orchestration.single_gpu import CyclePhase, GpuOwner, SingleGpuCycleState

WORKFLOW_PLAN_VERSION = "single-gpu-dry-run-plan/v1"

_RUNNABLE_PHASES = (
    CyclePhase.ROLLOUT_RUNNING,
    CyclePhase.TRAINING_RUNNING,
    CyclePhase.EVALUATION_RUNNING,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class PlannedCommand:
    phase: CyclePhase
    argv: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.phase not in _RUNNABLE_PHASES:
            raise ContractValidationError(f"phase {self.phase.value} is not command-runnable")
        argv = tuple(self.argv)
        if not argv or any(not isinstance(item, str) or not item for item in argv):
            raise ContractValidationError("argv must contain non-empty strings")
        object.__setattr__(self, "argv", argv)

    @property
    def gpu_owner(self) -> GpuOwner:
        return {
            CyclePhase.ROLLOUT_RUNNING: GpuOwner.SERVING,
            CyclePhase.TRAINING_RUNNING: GpuOwner.TRAINING,
            CyclePhase.EVALUATION_RUNNING: GpuOwner.SERVING,
        }[self.phase]

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "gpu_owner": self.gpu_owner.value,
            "argv": list(self.argv),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SingleGpuDryRunPlan:
    commands: tuple[PlannedCommand, ...]
    schema_version: str = WORKFLOW_PLAN_VERSION

    def __post_init__(self) -> None:
        commands = tuple(self.commands)
        if {command.phase for command in commands} != set(_RUNNABLE_PHASES):
            raise ContractValidationError(
                "dry-run plan requires rollout, training and evaluation commands"
            )
        if len(commands) != len(_RUNNABLE_PHASES):
            raise ContractValidationError("dry-run plan contains duplicate phase command")
        object.__setattr__(
            self, "commands", tuple(sorted(commands, key=lambda item: item.phase.value))
        )
        if self.schema_version != WORKFLOW_PLAN_VERSION:
            raise ContractValidationError(
                f"schema_version must be {WORKFLOW_PLAN_VERSION!r}"
            )

    def next_command(self, state: SingleGpuCycleState) -> PlannedCommand | None:
        """Return argv for the next runnable phase; never execute it."""

        next_phase = state.next_phase
        return next(
            (command for command in self.commands if command.phase is next_phase),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "commands": [command.to_dict() for command in self.commands],
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


class SingleGpuCycleStore:
    """Atomic local persistence with checksum compare-and-swap transitions."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir() or self.root.is_symlink():
            raise ContractValidationError("workflow store root must be a real directory")

    def create(self, state: SingleGpuCycleState) -> Path:
        if not isinstance(state, SingleGpuCycleState):
            raise TypeError("state must be a SingleGpuCycleState")
        with self._locked(state.cycle_id):
            path = self._state_path(state.cycle_id)
            if path.exists():
                raise ContractValidationError(f"cycle {state.cycle_id!r} already exists")
            self._atomic_write(path, state.to_dict())
            return path

    def load(self, cycle_id: str) -> SingleGpuCycleState:
        path = self._state_path(cycle_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ContractValidationError(f"cycle {cycle_id!r} does not exist") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractValidationError(f"cycle {cycle_id!r} state is unreadable") from exc
        state = SingleGpuCycleState.from_dict(payload)
        if state.cycle_id != cycle_id:
            raise ContractValidationError("stored cycle identity mismatch")
        return state

    def advance(
        self,
        cycle_id: str,
        *,
        expected_state_checksum: str,
        next_phase: CyclePhase,
        evidence_checksum: str | None = None,
        candidate_policy_fingerprint: str | None = None,
    ) -> SingleGpuCycleState:
        """Persist one transition only if the caller observed the current state."""

        with self._locked(cycle_id):
            current = self.load(cycle_id)
            if current.checksum != expected_state_checksum:
                raise ContractValidationError("stale workflow state checksum")
            updated = current.advance(
                next_phase,
                evidence_checksum=evidence_checksum,
                candidate_policy_fingerprint=candidate_policy_fingerprint,
            )
            self._atomic_write(self._state_path(cycle_id), updated.to_dict())
            return updated

    def _state_path(self, cycle_id: str) -> Path:
        if not isinstance(cycle_id, str) or not cycle_id.strip():
            raise ContractValidationError("cycle_id must be a non-empty string")
        return self.root / f"cycle-{sha256_json(cycle_id)}.json"

    @contextmanager
    def _locked(self, cycle_id: str) -> Iterator[None]:
        if not isinstance(cycle_id, str) or not cycle_id.strip():
            raise ContractValidationError("cycle_id must be a non-empty string")
        lock_path = self.root / f"cycle-{sha256_json(cycle_id)}.lock"
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _atomic_write(self, path: Path, payload: Mapping[str, Any]) -> None:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.root,
                prefix=".cycle-state-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(payload, temporary, ensure_ascii=False, sort_keys=True)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
            directory_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
