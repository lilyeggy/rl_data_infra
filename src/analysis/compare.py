"""Controlled paired comparison over canonical AgentEpisode records."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Mapping

from src.analysis.metrics import EpisodeMetrics
from src.contracts._json import sha256_json
from src.contracts.agent_episode import AgentEpisode, ExecutionValidity, IntegrityState, TaskStatus
from src.contracts.experiment import ExperimentManifest


COMPARISON_VERSION = "paired-comparison/v1"


@dataclass(frozen=True, slots=True)
class CompatibilityMismatch:
    pair_key: str
    field: str
    control_value: Any
    candidate_value: Any
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_key": self.pair_key,
            "field": self.field,
            "control_value": self.control_value,
            "candidate_value": self.candidate_value,
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class EpisodePairDelta:
    pair_key: str
    control_episode_id: str
    candidate_episode_id: str
    outcome_transition: str
    control_validity: str
    candidate_validity: str
    metric_deltas: Mapping[str, float | None]
    control_reason_codes: tuple[str, ...]
    candidate_reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_key": self.pair_key,
            "control_episode_id": self.control_episode_id,
            "candidate_episode_id": self.candidate_episode_id,
            "outcome_transition": self.outcome_transition,
            "control_validity": self.control_validity,
            "candidate_validity": self.candidate_validity,
            "metric_deltas": dict(self.metric_deltas),
            "control_reason_codes": list(self.control_reason_codes),
            "candidate_reason_codes": list(self.candidate_reason_codes),
        }


@dataclass(frozen=True, slots=True)
class AggregateComparison:
    control_success_rate: float | None
    candidate_success_rate: float | None
    success_rate_delta: float | None
    control_infra_invalid_rate: float
    candidate_infra_invalid_rate: float
    infra_invalid_rate_delta: float
    control_mean_tokens: float | None
    candidate_mean_tokens: float | None
    token_increase_ratio: float | None
    control_mean_duration_ms: float | None
    candidate_mean_duration_ms: float | None
    latency_increase_ratio: float | None
    control_target_slice_count: int
    candidate_target_slice_count: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy() if hasattr(self, "__dict__") else {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    experiment_id: str
    experiment_checksum: str
    pairs: tuple[EpisodePairDelta, ...]
    unmatched_control_episode_ids: tuple[str, ...]
    unmatched_candidate_episode_ids: tuple[str, ...]
    compatibility_mismatches: tuple[CompatibilityMismatch, ...]
    paired_coverage: float
    aggregate: AggregateComparison
    input_episode_checksums: tuple[str, ...]
    comparison_version: str = COMPARISON_VERSION

    @property
    def comparable(self) -> bool:
        return not self.compatibility_mismatches

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_version": self.comparison_version,
            "experiment_id": self.experiment_id,
            "experiment_checksum": self.experiment_checksum,
            "comparable": self.comparable,
            "pairs": [pair.to_dict() for pair in self.pairs],
            "unmatched_control_episode_ids": list(self.unmatched_control_episode_ids),
            "unmatched_candidate_episode_ids": list(self.unmatched_candidate_episode_ids),
            "compatibility_mismatches": [item.to_dict() for item in self.compatibility_mismatches],
            "paired_coverage": self.paired_coverage,
            "aggregate": self.aggregate.to_dict(),
            "input_episode_checksums": list(self.input_episode_checksums),
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


def _seed(episode: AgentEpisode) -> Any:
    return episode.model_manifest.sampling_config.get("seed")


def _pair_key(episode: AgentEpisode) -> tuple[str, str, int]:
    return (episode.task_id, str(_seed(episode)), episode.attempt)


def _pair_key_text(key: tuple[str, str, int]) -> str:
    return f"{key[0]}|seed={key[1]}|attempt={key[2]}"


def _index(episodes: Iterable[AgentEpisode]) -> dict[tuple[str, str, int], AgentEpisode]:
    result: dict[tuple[str, str, int], AgentEpisode] = {}
    for episode in episodes:
        key = _pair_key(episode)
        if key in result:
            raise ValueError(f"duplicate comparison pair key: {_pair_key_text(key)}")
        result[key] = episode
    return result


def _difference(candidate: float | int | None, control: float | int | None) -> float | None:
    if candidate is None or control is None:
        return None
    return float(candidate) - float(control)


def _mean(values: Iterable[float | int | None]) -> float | None:
    observed = [float(value) for value in values if value is not None]
    return fmean(observed) if observed else None


def _ratio(candidate: float | None, control: float | None) -> float | None:
    if candidate is None or control is None or control == 0:
        return None
    return (candidate - control) / control


def _success_rate(episodes: Iterable[AgentEpisode]) -> float | None:
    valid = [
        episode
        for episode in episodes
        if episode.outcome.execution_validity is ExecutionValidity.VALID
        and episode.integrity.state is IntegrityState.COMPLETE
    ]
    if not valid:
        return None
    return sum(item.outcome.task_status is TaskStatus.SUCCESS for item in valid) / len(valid)


def compare_runs(
    control_episodes: Iterable[AgentEpisode],
    candidate_episodes: Iterable[AgentEpisode],
    *,
    control_metrics: Iterable[EpisodeMetrics],
    candidate_metrics: Iterable[EpisodeMetrics],
    experiment: ExperimentManifest,
    control_reason_codes: Mapping[str, tuple[str, ...]] | None = None,
    candidate_reason_codes: Mapping[str, tuple[str, ...]] | None = None,
) -> ComparisonReport:
    controls = tuple(control_episodes)
    candidates = tuple(candidate_episodes)
    control_index = _index(controls)
    candidate_index = _index(candidates)
    control_metric_index = {item.episode_id: item for item in control_metrics}
    candidate_metric_index = {item.episode_id: item for item in candidate_metrics}
    control_reasons = control_reason_codes or {}
    candidate_reasons = candidate_reason_codes or {}
    common_keys = sorted(set(control_index) & set(candidate_index))
    mismatches: list[CompatibilityMismatch] = []
    pairs: list[EpisodePairDelta] = []

    for key in common_keys:
        control = control_index[key]
        candidate = candidate_index[key]
        key_text = _pair_key_text(key)
        comparisons = (
            ("model_manifest", control.model_manifest.to_dict(), candidate.model_manifest.to_dict()),
            ("environment_manifest", control.environment_manifest.to_dict(), candidate.environment_manifest.to_dict()),
            ("evaluator_manifest", control.evaluator_manifest.to_dict(), candidate.evaluator_manifest.to_dict()),
            ("experiment_manifest_ref", control.experiment_manifest_ref, candidate.experiment_manifest_ref),
        )
        for field, control_value, candidate_value in comparisons:
            if control_value != candidate_value:
                mismatches.append(
                    CompatibilityMismatch(
                        pair_key=key_text,
                        field=field,
                        control_value=control_value,
                        candidate_value=candidate_value,
                        explanation=f"{field} must be identical; only Harness policy may vary",
                    )
                )
        run_expectations = (
            ("control_run_id", control.run_id, experiment.control_run_id),
            ("candidate_run_id", candidate.run_id, experiment.candidate_run_id),
            (
                "task_dataset_revision",
                control.environment_manifest.task_snapshot,
                experiment.task_dataset_revision,
            ),
            (
                "task_dataset_revision",
                candidate.environment_manifest.task_snapshot,
                experiment.task_dataset_revision,
            ),
        )
        for field, actual, expected in run_expectations:
            if actual != expected:
                mismatches.append(
                    CompatibilityMismatch(
                        pair_key=key_text,
                        field=field,
                        control_value=actual,
                        candidate_value=expected,
                        explanation=f"observed {field} does not match ExperimentManifest",
                    )
                )
        cm = control_metric_index.get(control.episode_id)
        tm = candidate_metric_index.get(candidate.episode_id)
        if cm is None or tm is None:
            raise ValueError(f"metrics missing for pair {key_text}")
        pairs.append(
            EpisodePairDelta(
                pair_key=key_text,
                control_episode_id=control.episode_id,
                candidate_episode_id=candidate.episode_id,
                outcome_transition=(
                    f"{control.outcome.task_status.value}->{candidate.outcome.task_status.value}"
                ),
                control_validity=control.outcome.execution_validity.value,
                candidate_validity=candidate.outcome.execution_validity.value,
                metric_deltas={
                    "duration_ms": _difference(tm.duration_ms, cm.duration_ms),
                    "turn_count": _difference(tm.turn_count, cm.turn_count),
                    "tool_call_count": _difference(tm.tool_call_count, cm.tool_call_count),
                    "duplicate_action_count": _difference(
                        tm.duplicate_action_count, cm.duplicate_action_count
                    ),
                    "input_tokens": _difference(tm.input_tokens, cm.input_tokens),
                    "output_tokens": _difference(tm.output_tokens, cm.output_tokens),
                },
                control_reason_codes=tuple(control_reasons.get(control.episode_id, ())),
                candidate_reason_codes=tuple(candidate_reasons.get(candidate.episode_id, ())),
            )
        )

    maximum = max(len(controls), len(candidates), 1)
    coverage = len(common_keys) / maximum
    control_success = _success_rate(controls)
    candidate_success = _success_rate(candidates)
    control_infra = sum(
        item.outcome.execution_validity is ExecutionValidity.INFRA_INVALID for item in controls
    ) / max(len(controls), 1)
    candidate_infra = sum(
        item.outcome.execution_validity is ExecutionValidity.INFRA_INVALID for item in candidates
    ) / max(len(candidates), 1)
    control_paired_metrics = [control_metric_index[control_index[key].episode_id] for key in common_keys]
    candidate_paired_metrics = [candidate_metric_index[candidate_index[key].episode_id] for key in common_keys]
    control_tokens = _mean(
        None if item.input_tokens is None or item.output_tokens is None else item.input_tokens + item.output_tokens
        for item in control_paired_metrics
    )
    candidate_tokens = _mean(
        None if item.input_tokens is None or item.output_tokens is None else item.input_tokens + item.output_tokens
        for item in candidate_paired_metrics
    )
    control_duration = _mean(item.duration_ms for item in control_paired_metrics)
    candidate_duration = _mean(item.duration_ms for item in candidate_paired_metrics)
    target = experiment.target_policy_flag
    aggregate = AggregateComparison(
        control_success_rate=control_success,
        candidate_success_rate=candidate_success,
        success_rate_delta=_difference(candidate_success, control_success),
        control_infra_invalid_rate=control_infra,
        candidate_infra_invalid_rate=candidate_infra,
        infra_invalid_rate_delta=candidate_infra - control_infra,
        control_mean_tokens=control_tokens,
        candidate_mean_tokens=candidate_tokens,
        token_increase_ratio=_ratio(candidate_tokens, control_tokens),
        control_mean_duration_ms=control_duration,
        candidate_mean_duration_ms=candidate_duration,
        latency_increase_ratio=_ratio(candidate_duration, control_duration),
        control_target_slice_count=sum(
            target in control_reasons.get(item.episode_id, ()) for item in controls
        ),
        candidate_target_slice_count=sum(
            target in candidate_reasons.get(item.episode_id, ()) for item in candidates
        ),
    )
    unmatched_control = tuple(
        control_index[key].episode_id for key in sorted(set(control_index) - set(candidate_index))
    )
    unmatched_candidate = tuple(
        candidate_index[key].episode_id for key in sorted(set(candidate_index) - set(control_index))
    )
    return ComparisonReport(
        experiment_id=experiment.experiment_id,
        experiment_checksum=experiment.checksum,
        pairs=tuple(pairs),
        unmatched_control_episode_ids=unmatched_control,
        unmatched_candidate_episode_ids=unmatched_candidate,
        compatibility_mismatches=tuple(mismatches),
        paired_coverage=coverage,
        aggregate=aggregate,
        input_episode_checksums=tuple(sorted(item.checksum for item in controls + candidates)),
    )
