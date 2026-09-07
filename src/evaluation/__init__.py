"""Layered canonical-action evaluation (format / selection / replay)."""

from src.evaluation.data_quality import DataQualityReport, build_data_quality_report
from src.evaluation.layered_eval import (
    EVAL_VERSION,
    FormatEvalResult,
    FormatFailure,
    ReplayResult,
    SelectionResult,
    evaluate_action_selection,
    evaluate_format,
    evaluate_trajectory_replay,
)

__all__ = [
    "EVAL_VERSION",
    "DataQualityReport",
    "FormatEvalResult",
    "FormatFailure",
    "ReplayResult",
    "SelectionResult",
    "build_data_quality_report",
    "evaluate_action_selection",
    "evaluate_format",
    "evaluate_trajectory_replay",
]