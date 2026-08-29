"""Evidence-linked deterministic rules for the first failure-attribution slice."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.contracts._json import canonical_json_bytes, sha256_json
from src.contracts.agent_episode import (
    AgentEpisode,
    CaptureCapability,
    ExecutionValidity,
    IntegrityState,
    TaskStatus,
)
from src.contracts.trace_event import EventComponent, EventStatus, EventType, TraceEvent


ATTRIBUTION_RULE_VERSION = "failure-attribution/v1"


class FailureLayer(str, Enum):
    MODEL = "MODEL"
    HARNESS = "HARNESS"
    SANDBOX = "SANDBOX"
    MODEL_BACKEND = "MODEL_BACKEND"
    EVALUATOR = "EVALUATOR"
    EXTERNAL_SERVICE = "EXTERNAL_SERVICE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class Diagnosis:
    diagnosis_id: str
    episode_id: str
    layer: FailureLayer
    reason_code: str
    evidence_event_ids: tuple[str, ...]
    evidence_artifact_ids: tuple[str, ...]
    confidence: float
    explanation: str
    input_episode_checksum: str
    rule_version: str = ATTRIBUTION_RULE_VERSION
    schema_version: str = "diagnosis/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "diagnosis_id": self.diagnosis_id,
            "episode_id": self.episode_id,
            "layer": self.layer.value,
            "reason_code": self.reason_code,
            "evidence_event_ids": list(self.evidence_event_ids),
            "evidence_artifact_ids": list(self.evidence_artifact_ids),
            "confidence": self.confidence,
            "rule_version": self.rule_version,
            "explanation": self.explanation,
            "input_episode_checksum": self.input_episode_checksum,
        }


@dataclass(frozen=True, slots=True)
class AttributionReport:
    episode_id: str
    diagnoses: tuple[Diagnosis, ...]
    insufficient_evidence_rules: tuple[str, ...]
    input_episode_checksum: str
    rule_version: str = ATTRIBUTION_RULE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "diagnoses": [item.to_dict() for item in self.diagnoses],
            "insufficient_evidence_rules": list(self.insufficient_evidence_rules),
            "input_episode_checksum": self.input_episode_checksum,
            "rule_version": self.rule_version,
        }


class AttributionEngine:
    def analyze(self, episode: AgentEpisode) -> AttributionReport:
        diagnoses: list[Diagnosis] = []
        insufficient: list[str] = []

        if episode.integrity.state is not IntegrityState.COMPLETE:
            evidence = tuple(event.event_id for event in episode.events)
            diagnoses.append(
                self._diagnosis(
                    episode,
                    FailureLayer.UNKNOWN,
                    "TRACE_INTEGRITY_FAILURE",
                    evidence,
                    1.0,
                    "capture integrity is not COMPLETE; behavioral conclusions are bounded",
                )
            )

        if episode.outcome.execution_validity is ExecutionValidity.INFRA_INVALID:
            diagnoses.append(self._infra_diagnosis(episode))
        elif episode.outcome.task_status is TaskStatus.FAILURE:
            repeated = self._repeated_action_after_tool_error(episode.events)
            if repeated is not None:
                error_event, repeated_call, decision_event = repeated
                if (
                    CaptureCapability.HARNESS_DECISIONS in episode.capabilities
                    and decision_event is not None
                ):
                    layer = FailureLayer.HARNESS
                    reason = "TOOL_ERROR_FEEDBACK_LOSS"
                    confidence = 0.8
                    explanation = (
                        "an observed tool error was followed by an identical action while "
                        "Harness decision capture was active"
                    )
                else:
                    layer = FailureLayer.UNKNOWN
                    reason = "OBSERVED_TOOL_ERROR_LOOP"
                    confidence = 0.5
                    explanation = (
                        "an observed tool error was followed by an identical action, but "
                        "the responsible Harness decision was NOT_OBSERVABLE"
                    )
                    insufficient.append(
                        "TOOL_ERROR_FEEDBACK_LOSS requires an observed HARNESS_DECISION"
                    )
                evidence = [error_event.event_id]
                if decision_event is not None:
                    evidence.append(decision_event.event_id)
                evidence.append(repeated_call.event_id)
                diagnoses.append(
                    self._diagnosis(
                        episode,
                        layer,
                        reason,
                        tuple(evidence),
                        confidence,
                        explanation,
                    )
                )
            else:
                diagnoses.append(
                    self._diagnosis(
                        episode,
                        FailureLayer.UNKNOWN,
                        "UNATTRIBUTED_TASK_FAILURE",
                        episode.outcome.evidence_event_ids,
                        0.0,
                        "the verifier observed a valid task failure, but no enabled rule identifies its cause",
                    )
                )

        return AttributionReport(
            episode_id=episode.episode_id,
            diagnoses=tuple(diagnoses),
            insufficient_evidence_rules=tuple(insufficient),
            input_episode_checksum=episode.checksum,
        )

    def _infra_diagnosis(self, episode: AgentEpisode) -> Diagnosis:
        failures = [
            event
            for event in episode.events
            if event.status in {EventStatus.ERROR, EventStatus.TIMEOUT}
        ]
        event = failures[0] if failures else None
        if event is None:
            return self._diagnosis(
                episode,
                FailureLayer.UNKNOWN,
                "INFRA_FAILURE_UNATTRIBUTED",
                episode.outcome.evidence_event_ids,
                0.0,
                "execution was marked infra-invalid without an observable failing component event",
            )
        layer_by_component = {
            EventComponent.SANDBOX: FailureLayer.SANDBOX,
            EventComponent.MODEL_BACKEND: FailureLayer.MODEL_BACKEND,
            EventComponent.EVALUATOR: FailureLayer.EVALUATOR,
            EventComponent.EXTERNAL_SERVICE: FailureLayer.EXTERNAL_SERVICE,
        }
        layer = layer_by_component.get(event.component, FailureLayer.UNKNOWN)
        return self._diagnosis(
            episode,
            layer,
            f"{layer.value}_OPERATION_{event.status.value}",
            (event.event_id,),
            1.0,
            "an observable component operation failed before a trustworthy task result was produced",
        )

    @staticmethod
    def _repeated_action_after_tool_error(
        events: tuple[TraceEvent, ...],
    ) -> tuple[TraceEvent, TraceEvent, TraceEvent | None] | None:
        last_call_key: bytes | None = None
        last_error: TraceEvent | None = None
        decision_after_error: TraceEvent | None = None
        for event in events:
            if event.event_type is EventType.TOOL_CALL:
                key = canonical_json_bytes(
                    {
                        "tool_name": event.attributes.get("tool_name"),
                        "arguments": event.attributes.get("arguments"),
                    }
                )
                if last_error is not None and key == last_call_key:
                    return last_error, event, decision_after_error
                last_call_key = key
            elif event.event_type is EventType.HARNESS_DECISION and last_error is not None:
                decision_after_error = event
            elif event.event_type is EventType.TOOL_RESULT and event.status in {
                EventStatus.FAILED,
                EventStatus.ERROR,
                EventStatus.TIMEOUT,
            }:
                last_error = event
                decision_after_error = None
            elif event.event_type is EventType.TOOL_RESULT and event.status is EventStatus.SUCCEEDED:
                last_error = None
                decision_after_error = None
        return None

    @staticmethod
    def _diagnosis(
        episode: AgentEpisode,
        layer: FailureLayer,
        reason_code: str,
        evidence_event_ids: tuple[str, ...],
        confidence: float,
        explanation: str,
    ) -> Diagnosis:
        artifact_ids = tuple(
            sorted(
                {
                    artifact_id
                    for event in episode.events
                    if event.event_id in evidence_event_ids
                    for artifact_id in event.artifact_refs
                }
            )
        )
        identity = sha256_json(
            {
                "episode_id": episode.episode_id,
                "layer": layer.value,
                "reason_code": reason_code,
                "evidence_event_ids": evidence_event_ids,
                "rule_version": ATTRIBUTION_RULE_VERSION,
            }
        )
        return Diagnosis(
            diagnosis_id=f"diagnosis-{identity[:20]}",
            episode_id=episode.episode_id,
            layer=layer,
            reason_code=reason_code,
            evidence_event_ids=evidence_event_ids,
            evidence_artifact_ids=artifact_ids,
            confidence=confidence,
            explanation=explanation,
            input_episode_checksum=episode.checksum,
        )
