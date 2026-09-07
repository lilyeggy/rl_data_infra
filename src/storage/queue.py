"""Bounded ingestion queue and worker boundary for V2.3."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from src.contracts.trace_event import TraceEvent
from src.storage.event_store import IngestionBatchResult, PartitionedEventStore


class BackpressureError(RuntimeError):
    """The bounded queue cannot accept more events without blocking."""


@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    max_events: int
    pending_events: int
    pending_batches: int
    enqueued_events: int
    dequeued_events: int
    rejected_batches: int


class BoundedEventQueue:
    """Thread-safe bounded queue measured in events, not batch count."""

    def __init__(self, *, max_events: int) -> None:
        if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events <= 0:
            raise ValueError("max_events must be a positive integer")
        self.max_events = max_events
        self._condition = threading.Condition()
        self._batches: deque[tuple[TraceEvent, ...]] = deque()
        self._pending_events = 0
        self._enqueued_events = 0
        self._dequeued_events = 0
        self._rejected_batches = 0

    def put(
        self,
        events: Iterable[TraceEvent],
        *,
        block: bool = False,
        timeout_seconds: float | None = None,
    ) -> None:
        batch = tuple(events)
        if not batch:
            return
        if any(not isinstance(event, TraceEvent) for event in batch):
            raise TypeError("BoundedEventQueue accepts TraceEvent values")
        if len(batch) > self.max_events:
            raise BackpressureError("batch is larger than queue capacity")
        deadline = None
        if timeout_seconds is not None:
            if timeout_seconds < 0:
                raise ValueError("timeout_seconds must be non-negative")
            deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._pending_events + len(batch) > self.max_events:
                if not block:
                    self._rejected_batches += 1
                    raise BackpressureError("event queue is full")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    self._rejected_batches += 1
                    raise BackpressureError("event queue put timed out")
                self._condition.wait(remaining)
            self._batches.append(batch)
            self._pending_events += len(batch)
            self._enqueued_events += len(batch)
            self._condition.notify_all()

    def get(
        self, *, block: bool = False, timeout_seconds: float | None = None
    ) -> tuple[TraceEvent, ...]:
        deadline = None
        if timeout_seconds is not None:
            if timeout_seconds < 0:
                raise ValueError("timeout_seconds must be non-negative")
            deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while not self._batches:
                if not block:
                    raise BackpressureError("event queue is empty")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise BackpressureError("event queue get timed out")
                self._condition.wait(remaining)
            batch = self._batches.popleft()
            self._pending_events -= len(batch)
            self._dequeued_events += len(batch)
            self._condition.notify_all()
            return batch

    def snapshot(self) -> QueueSnapshot:
        with self._condition:
            return QueueSnapshot(
                max_events=self.max_events,
                pending_events=self._pending_events,
                pending_batches=len(self._batches),
                enqueued_events=self._enqueued_events,
                dequeued_events=self._dequeued_events,
                rejected_batches=self._rejected_batches,
            )


@dataclass(frozen=True, slots=True)
class WorkerDrainResult:
    batches: int
    events: int
    ingestion_results: tuple[IngestionBatchResult, ...]


class IngestionWorker:
    """Drain bounded batches into a PartitionedEventStore."""

    def __init__(self, queue: BoundedEventQueue, store: PartitionedEventStore) -> None:
        self.queue = queue
        self.store = store

    def drain(self, *, max_batches: int | None = None) -> WorkerDrainResult:
        if max_batches is not None and (
            isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches <= 0
        ):
            raise ValueError("max_batches must be a positive integer when provided")
        results: list[IngestionBatchResult] = []
        events = 0
        while max_batches is None or len(results) < max_batches:
            try:
                batch = self.queue.get()
            except BackpressureError:
                break
            results.append(self.store.ingest(batch))
            events += len(batch)
        return WorkerDrainResult(
            batches=len(results),
            events=events,
            ingestion_results=tuple(results),
        )
