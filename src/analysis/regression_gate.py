"""Three-state regression decision with explicit evidence for every rule."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.analysis.compare import ComparisonReport
from src.contracts._json import sha256_json


GATE_VERSION = "regression-gate/v1"


class GateDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True, slots=True, kw_only=True)
class GateConfig:
    minimum_paired_coverage: float = 1.0
    minimum_episode_pairs: int = 3
    success_rate_drop_tolerance: float = 0.0
    require_target_slice_improvement: bool = True
    max_token_increase_ratio: float = 0.20
    max_latency_increase_ratio: float = 0.25
    max_infra_invalid_increase: float = 0.0
    reject_new_severe_regression: bool = True
    schema_version: str = "gate-config/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class GateCheck:
    rule: str
    actual: Any
    threshold: Any
    passed: bool | None
    evidence: tuple[str, ...]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "actual": self.actual,
            "threshold": self.threshold,
            "passed": self.passed,
            "evidence": list(self.evidence),
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class GateResult:
    decision: GateDecision
    checks: tuple[GateCheck, ...]
    comparison_checksum: str
    config_checksum: str
    gate_version: str = GATE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_version": self.gate_version,
            "decision": self.decision.value,
            "checks": [check.to_dict() for check in self.checks],
            "comparison_checksum": self.comparison_checksum,
            "config_checksum": self.config_checksum,
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


def evaluate_gate(report: ComparisonReport, config: GateConfig) -> GateResult:
    evidence = tuple(pair.pair_key for pair in report.pairs)
    checks: list[GateCheck] = []
    checks.append(
        GateCheck(
            "compatibility",
            len(report.compatibility_mismatches),
            0,
            not report.compatibility_mismatches,
            tuple(item.pair_key for item in report.compatibility_mismatches),
            "non-Harness confounders must be absent",
        )
    )
    checks.append(
        GateCheck(
            "minimum_episode_pairs",
            len(report.pairs),
            config.minimum_episode_pairs,
            len(report.pairs) >= config.minimum_episode_pairs,
            evidence,
            "paired sample floor",
        )
    )
    checks.append(
        GateCheck(
            "minimum_paired_coverage",
            report.paired_coverage,
            config.minimum_paired_coverage,
            report.paired_coverage >= config.minimum_paired_coverage,
            evidence,
            "unmatched executions reduce causal coverage",
        )
    )

    blocking = any(check.passed is False for check in checks)
    if blocking:
        return GateResult(
            decision=GateDecision.INSUFFICIENT_EVIDENCE,
            checks=tuple(checks),
            comparison_checksum=report.checksum,
            config_checksum=config.checksum,
        )

    aggregate = report.aggregate

    def bounded(rule: str, actual: float | None, threshold: float, explanation: str) -> None:
        checks.append(
            GateCheck(
                rule,
                actual,
                threshold,
                None if actual is None else actual <= threshold,
                evidence,
                explanation,
            )
        )

    success_delta = aggregate.success_rate_delta
    checks.append(
        GateCheck(
            "success_rate_drop_tolerance",
            success_delta,
            -config.success_rate_drop_tolerance,
            None if success_delta is None else success_delta >= -config.success_rate_drop_tolerance,
            evidence,
            "candidate success rate may not regress beyond tolerance",
        )
    )
    bounded(
        "max_token_increase_ratio",
        aggregate.token_increase_ratio,
        config.max_token_increase_ratio,
        "candidate token growth is bounded",
    )
    bounded(
        "max_latency_increase_ratio",
        aggregate.latency_increase_ratio,
        config.max_latency_increase_ratio,
        "candidate wall-time growth is bounded",
    )
    bounded(
        "max_infra_invalid_increase",
        aggregate.infra_invalid_rate_delta,
        config.max_infra_invalid_increase,
        "candidate may not introduce infrastructure invalidity",
    )
    if config.require_target_slice_improvement:
        checks.append(
            GateCheck(
                "target_slice_improvement",
                {
                    "control": aggregate.control_target_slice_count,
                    "candidate": aggregate.candidate_target_slice_count,
                },
                "candidate < control",
                aggregate.candidate_target_slice_count < aggregate.control_target_slice_count,
                evidence,
                "the declared Harness failure slice must improve",
            )
        )
    if config.reject_new_severe_regression:
        regressions = tuple(
            pair.pair_key
            for pair in report.pairs
            if pair.outcome_transition == "SUCCESS->FAILURE"
            and pair.candidate_validity == "VALID"
        )
        checks.append(
            GateCheck(
                "reject_new_severe_regression",
                len(regressions),
                0,
                not regressions,
                regressions,
                "a previously successful valid task may not become a valid failure",
            )
        )
    if any(check.passed is None for check in checks):
        decision = GateDecision.INSUFFICIENT_EVIDENCE
    elif any(check.passed is False for check in checks):
        decision = GateDecision.REJECT
    else:
        decision = GateDecision.ACCEPT
    return GateResult(
        decision=decision,
        checks=tuple(checks),
        comparison_checksum=report.checksum,
        config_checksum=config.checksum,
    )
