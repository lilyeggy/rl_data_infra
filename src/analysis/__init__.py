"""Versioned derived observability records; canonical facts remain unchanged."""

from src.analysis.attribution import (
    ATTRIBUTION_RULE_VERSION,
    AttributionEngine,
    AttributionReport,
    Diagnosis,
    FailureLayer,
)
from src.analysis.metrics import METRIC_VERSION, EpisodeMetrics, compute_episode_metrics
from src.analysis.compare import (
    COMPARISON_VERSION,
    AggregateComparison,
    ComparisonReport,
    CompatibilityMismatch,
    EpisodePairDelta,
    compare_runs,
)
from src.analysis.regression_gate import (
    GATE_VERSION,
    GateCheck,
    GateConfig,
    GateDecision,
    GateResult,
    evaluate_gate,
)

__all__ = [
    "ATTRIBUTION_RULE_VERSION",
    "AttributionEngine",
    "AttributionReport",
    "AggregateComparison",
    "COMPARISON_VERSION",
    "ComparisonReport",
    "CompatibilityMismatch",
    "Diagnosis",
    "EpisodeMetrics",
    "EpisodePairDelta",
    "FailureLayer",
    "METRIC_VERSION",
    "GATE_VERSION",
    "GateCheck",
    "GateConfig",
    "GateDecision",
    "GateResult",
    "compare_runs",
    "compute_episode_metrics",
    "evaluate_gate",
]
