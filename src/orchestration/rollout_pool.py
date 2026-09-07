"""A bounded Local rollout pool with Polar-compatible stage observability."""

from __future__ import annotations

import concurrent.futures
import json
import statistics
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Generic, Sequence, TypeVar

from src.orchestration.local_execution import (
    LocalExecutionOrchestrator,
    LocalExecutionResult,
    LocalExecutionSpec,
)


class RolloutPoolStage(str, Enum):
    QUEUED = "QUEUED"
    INIT = "INIT"
    RUN = "RUN"
    POSTRUN = "POSTRUN"
    TERMINAL = "TERMINAL"


@dataclass(frozen=True, slots=True)
class RolloutPoolSnapshot:
    queued_depth: int
    init_inflight: int
    run_inflight: int
    postrun_inflight: int
    completed_total: int
    failed_total: int

    def to_dict(self) -> dict[str, int]:
        return {
            "queued_depth": self.queued_depth,
            "init_inflight": self.init_inflight,
            "run_inflight": self.run_inflight,
            "postrun_inflight": self.postrun_inflight,
            "completed_total": self.completed_total,
            "failed_total": self.failed_total,
        }


@dataclass(frozen=True, slots=True)
class RolloutPoolEpisodeTiming:
    episode_id: str
    queue_ms: float
    init_ms: float
    run_ms: float
    postrun_ms: float
    terminal_status: str

    def to_dict(self) -> dict[str, str | float]:
        return {
            "episode_id": self.episode_id,
            "queue_ms": self.queue_ms,
            "init_ms": self.init_ms,
            "run_ms": self.run_ms,
            "postrun_ms": self.postrun_ms,
            "terminal_status": self.terminal_status,
        }


class RolloutPoolMetrics:
    """Thread-safe stage gauges plus terminal timings for a bounded rollout pool."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.RLock()
        self._states: dict[str, RolloutPoolStage] = {}
        self._timestamps: dict[str, dict[RolloutPoolStage, float]] = {}
        self._terminal: list[RolloutPoolEpisodeTiming] = []

    def register(self, episode_id: str) -> None:
        with self._lock:
            if episode_id in self._states:
                raise ValueError(f"episode is already registered: {episode_id}")
            self._states[episode_id] = RolloutPoolStage.QUEUED
            self._timestamps[episode_id] = {RolloutPoolStage.QUEUED: self._clock()}

    def transition(self, episode_id: str, stage: RolloutPoolStage) -> None:
        with self._lock:
            current = self._states.get(episode_id)
            if current is None:
                raise ValueError(f"unknown episode: {episode_id}")
            if current is RolloutPoolStage.TERMINAL:
                raise ValueError(f"terminal episode cannot transition: {episode_id}")
            if stage is RolloutPoolStage.QUEUED or stage is RolloutPoolStage.TERMINAL:
                raise ValueError("use register() or finish() for QUEUED/TERMINAL")
            order = {
                RolloutPoolStage.QUEUED: 0,
                RolloutPoolStage.INIT: 1,
                RolloutPoolStage.RUN: 2,
                RolloutPoolStage.POSTRUN: 3,
            }
            if order[stage] < order[current]:
                raise ValueError(f"stage regression: {current.value} -> {stage.value}")
            if stage is current:
                return
            self._states[episode_id] = stage
            self._timestamps[episode_id][stage] = self._clock()

    def finish(self, episode_id: str, *, status: str) -> None:
        with self._lock:
            current = self._states.get(episode_id)
            if current is None or current is RolloutPoolStage.TERMINAL:
                raise ValueError(f"episode is not active: {episode_id}")
            now = self._clock()
            times = self._timestamps[episode_id]
            queued = times[RolloutPoolStage.QUEUED]
            init = times.get(RolloutPoolStage.INIT, now)
            run = times.get(RolloutPoolStage.RUN, now)
            postrun = times.get(RolloutPoolStage.POSTRUN, now)
            self._terminal.append(RolloutPoolEpisodeTiming(
                episode_id=episode_id,
                queue_ms=max(0.0, (init - queued) * 1000),
                init_ms=max(0.0, (run - init) * 1000),
                run_ms=max(0.0, (postrun - run) * 1000),
                postrun_ms=max(0.0, (now - postrun) * 1000),
                terminal_status=status,
            ))
            self._states[episode_id] = RolloutPoolStage.TERMINAL

    def snapshot(self) -> RolloutPoolSnapshot:
        with self._lock:
            counts = {stage: sum(value is stage for value in self._states.values()) for stage in RolloutPoolStage}
            completed = sum(item.terminal_status == "COMPLETED" for item in self._terminal)
            return RolloutPoolSnapshot(
                queued_depth=counts[RolloutPoolStage.QUEUED],
                init_inflight=counts[RolloutPoolStage.INIT],
                run_inflight=counts[RolloutPoolStage.RUN],
                postrun_inflight=counts[RolloutPoolStage.POSTRUN],
                completed_total=completed,
                failed_total=len(self._terminal) - completed,
            )

    def to_dict(self) -> dict[str, object]:
        with self._lock:
            timings = list(self._terminal)
        fields = ("queue_ms", "init_ms", "run_ms", "postrun_ms")
        summary = {
            field: {
                "mean": statistics.mean(getattr(item, field) for item in timings) if timings else 0.0,
                "p95": sorted(getattr(item, field) for item in timings)[max(0, int(len(timings) * .95) - 1)] if timings else 0.0,
            }
            for field in fields
        }
        return {
            "schema_version": "local-rollout-pool-metrics/v1",
            "snapshot": self.snapshot().to_dict(),
            "summary_ms": summary,
            "episodes": [item.to_dict() for item in timings],
        }


ResultT = TypeVar("ResultT")


class LocalRolloutPool(Generic[ResultT]):
    """Runs fixed-spec Local executions with bounded concurrency and stage gauges."""

    def __init__(
        self,
        *,
        max_concurrency: int,
        execute: Callable[[LocalExecutionSpec, Path, Callable[[str], None]], ResultT] | None = None,
        metrics: RolloutPoolMetrics | None = None,
    ) -> None:
        if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        self.max_concurrency = max_concurrency
        self.metrics = metrics or RolloutPoolMetrics()
        self._execute = execute or self._default_execute

    @staticmethod
    def _default_execute(
        spec: LocalExecutionSpec,
        output_dir: Path,
        observer: Callable[[str], None],
    ) -> LocalExecutionResult:
        return LocalExecutionOrchestrator().run(spec, output_dir=output_dir, stage_observer=observer)

    def run(self, specs: Sequence[LocalExecutionSpec], *, output_root: str | Path) -> list[ResultT]:
        root = Path(output_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            self.metrics.register(spec.identity.episode_id)

        def one(spec: LocalExecutionSpec) -> ResultT:
            episode_id = spec.identity.episode_id
            self.metrics.transition(episode_id, RolloutPoolStage.INIT)

            def observer(stage: str) -> None:
                self.metrics.transition(episode_id, RolloutPoolStage(stage))

            try:
                result = self._execute(spec, root / spec.identity.checksum[:16], observer)
            except Exception:
                self.metrics.finish(episode_id, status="FAILED")
                raise
            producer_status = getattr(getattr(result, "producer_artifact", None), "status", None)
            self.metrics.finish(
                episode_id,
                status=(
                    "COMPLETED"
                    if getattr(producer_status, "value", None) == "COMPLETED"
                    else "FAILED"
                ),
            )
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            results = list(executor.map(one, specs))
        (root / "pool-metrics.json").write_text(json.dumps(self.metrics.to_dict(), indent=2) + "\n", encoding="utf-8")
        return results
