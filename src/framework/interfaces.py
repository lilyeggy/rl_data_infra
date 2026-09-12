# Minimal framework-level protocols for the harness data plane.
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from src.contracts._json import sha256_json

CONSUMERS = ("SFT", "PREFERENCE", "ON_POLICY_RL", "EVALUATION")


@dataclass(frozen=True, slots=True)
class TaskRef:
    task_id: str
    statement: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelRequest:
    run_id: str
    task: TaskRef
    messages: Sequence[Mapping[str, Any]]
    tools: Sequence[Mapping[str, Any]] = ()
    sampling: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str
    prompt_token_ids: tuple[int, ...] = ()
    response_token_ids: tuple[int, ...] = ()
    response_logprobs: tuple[float, ...] = ()
    tool_calls: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class Verdict:
    status: str
    score: float | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    episode_id: str
    task_id: str
    verdict: Verdict
    evidence_dir: Path
    model_calls: int = 0
    tool_rounds: int = 0


@runtime_checkable
class TaskSource(Protocol):
    def tasks(self) -> Sequence[TaskRef]:
        ...


@runtime_checkable
class ModelBackend(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse:
        ...


@runtime_checkable
class Verifier(Protocol):
    def verify(self, result: ExecutionResult, workspace: Path) -> Verdict:
        ...


@runtime_checkable
class HarnessAdapter(Protocol):
    def run(self, spec: RunSpec) -> ExecutionResult:
        ...


@runtime_checkable
class ConsumerCompiler(Protocol):
    name: str

    def compile(self, result: ExecutionResult, output_dir: Path) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True, slots=True)
class RunSpec:
    run_id: str
    task: TaskRef
    harness: HarnessAdapter
    model: ModelBackend
    verifier: Verifier
    consumer: str
    output_dir: Path
    group_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DataPlane:
    # Thin facade over a harness, a model, a verifier and consumer compilers.
    def __init__(self, consumers: Sequence[ConsumerCompiler] = ()):
        self._consumers = {consumer.name: consumer for consumer in consumers}

    def run(self, spec: RunSpec) -> ExecutionResult:
        if spec.consumer not in CONSUMERS:
            raise ValueError("unknown consumer profile: " + str(spec.consumer))
        result = spec.harness.run(spec)
        manifest = {
            "run_id": spec.run_id,
            "task_id": spec.task.task_id,
            "harness": type(spec.harness).__name__,
            "model": type(spec.model).__name__,
            "verifier": type(spec.verifier).__name__,
            "consumer": spec.consumer,
            "group_id": spec.group_id,
            "verdict": result.verdict.status,
            "score": result.verdict.score,
            "evidence_dir": str(result.evidence_dir),
            "model_calls": result.model_calls,
            "tool_rounds": result.tool_rounds,
        }
        manifest["checksum"] = sha256_json(manifest)
        out = Path(spec.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "run-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result

    def compile(self, result: ExecutionResult, consumer: str, output_dir: Path) -> Mapping[str, Any]:
        if consumer not in self._consumers:
            raise KeyError("no consumer compiler registered for " + str(consumer))
        return self._consumers[consumer].compile(result, output_dir)
