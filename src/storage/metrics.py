"""Deterministic ingestion latency summaries for V2.3 load evidence."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Iterable


INGESTION_METRICS_VERSION = "ingestion-metrics/v1"


@dataclass(frozen=True, slots=True)
class IngestionLatencySummary:
    count: int
    mean_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    max_ms: float | None
    metrics_version: str = INGESTION_METRICS_VERSION

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "metrics_version": self.metrics_version,
            "count": self.count,
            "mean_ms": self.mean_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "max_ms": self.max_ms,
        }


def summarize_ingestion_latencies(values: Iterable[float]) -> IngestionLatencySummary:
    observations = tuple(float(value) for value in values)
    if any(value < 0 for value in observations):
        raise ValueError("latencies must be non-negative")
    if not observations:
        return IngestionLatencySummary(
            count=0,
            mean_ms=None,
            p95_ms=None,
            p99_ms=None,
            max_ms=None,
        )
    ordered = tuple(sorted(observations))
    return IngestionLatencySummary(
        count=len(ordered),
        mean_ms=fmean(ordered),
        p95_ms=_nearest_rank(ordered, 0.95),
        p99_ms=_nearest_rank(ordered, 0.99),
        max_ms=ordered[-1],
    )


def _nearest_rank(values: tuple[float, ...], percentile: float) -> float:
    index = max(0, int((len(values) * percentile + 0.999999999) - 1))
    return values[min(index, len(values) - 1)]
