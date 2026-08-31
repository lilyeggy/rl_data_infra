"""Deterministically assemble interleaved, duplicated or partial event streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from src.contracts._json import sha256_json
from src.contracts.agent_episode import (
    AgentEpisode,
    CaptureCapability,
    EpisodeOutcome,
    EpisodeTermination,
    EpisodeVerifierStatus,
    ExecutionValidity,
    IntegrityReport,
    IntegrityState,
    TaskStatus,
)
from src.contracts.artifacts import ArtifactRef
from src.contracts.manifests import (
    EnvironmentManifest,
    EvaluatorManifest,
    HarnessManifest,
    ModelManifest,
)
from src.contracts.trace_event import EventType, TraceEvent
from src.errors import ContractValidationError

ASSEMBLER_VERSION = "episode-assembler/v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class EpisodeContext:
    """Run metadata supplied out-of-band rather than guessed from event text."""

    task_id: str
    attempt: int
    harness_manifest: HarnessManifest
    model_manifest: ModelManifest
    environment_manifest: EnvironmentManifest
    evaluator_manifest: EvaluatorManifest
    experiment_manifest_ref: str
    capabilities: frozenset[CaptureCapability]


@dataclass(frozen=True, slots=True)
class QuarantinedEvent:
    event_id: str
    episode_id: str
    reason_code: str
    explanation: str
    event_checksum: str


@dataclass(frozen=True, slots=True)
class AssemblyResult:
    episodes: tuple[AgentEpisode, ...]
    quarantined_events: tuple[QuarantinedEvent, ...]
    warnings: tuple[str, ...]
    input_checksum: str
    output_checksum: str


class EpisodeAssembler:
    def assemble(
        self,
        events: Iterable[TraceEvent],
        *,
        contexts: Mapping[str, EpisodeContext],
        artifacts: Iterable[ArtifactRef] = (),
    ) -> AssemblyResult:
        raw_events = tuple(events)
        if any(not isinstance(event, TraceEvent) for event in raw_events):
            raise TypeError("EpisodeAssembler accepts only validated TraceEvent values")
        input_checksum = sha256_json([event.to_dict() for event in raw_events])
        artifact_index = self._artifact_index(artifacts)
        accepted: list[TraceEvent] = []
        quarantine: list[QuarantinedEvent] = []
        duplicate_counts: dict[str, int] = {}
        conflicting_episode_ids: set[str] = set()
        seen: dict[str, TraceEvent] = {}

        for event in raw_events:
            prior = seen.get(event.event_id)
            if prior is not None:
                duplicate_counts[event.episode_id] = duplicate_counts.get(event.episode_id, 0) + 1
                if prior.checksum != event.checksum:
                    conflicting_episode_ids.add(prior.episode_id)
                    conflicting_episode_ids.add(event.episode_id)
                    quarantine.append(
                        QuarantinedEvent(
                            event.event_id,
                            event.episode_id,
                            "CONFLICTING_EVENT_ID",
                            "the same event_id carried different immutable content",
                            event.checksum,
                        )
                    )
                continue
            seen[event.event_id] = event
            if event.episode_id not in contexts:
                quarantine.append(
                    QuarantinedEvent(
                        event.event_id,
                        event.episode_id,
                        "EPISODE_CONTEXT_MISSING",
                        "task and manifest context was not supplied; metadata was not inferred",
                        event.checksum,
                    )
                )
                continue
            accepted.append(event)

        grouped: dict[str, list[TraceEvent]] = {}
        for event in accepted:
            grouped.setdefault(event.episode_id, []).append(event)

        episodes: list[AgentEpisode] = []
        warnings: list[str] = []
        for episode_id in sorted(grouped):
            try:
                episode = self._assemble_episode(
                    grouped[episode_id],
                    contexts[episode_id],
                    artifact_index,
                    duplicate_event_count=duplicate_counts.get(episode_id, 0),
                    conflicting_event_id=episode_id in conflicting_episode_ids,
                )
            except ContractValidationError as exc:
                warnings.append(f"episode {episode_id} could not be assembled: {exc}")
                for event in grouped[episode_id]:
                    quarantine.append(
                        QuarantinedEvent(
                            event.event_id,
                            event.episode_id,
                            "EPISODE_CONTRACT_INVALID",
                            str(exc),
                            event.checksum,
                        )
                    )
            else:
                episodes.append(episode)

        output_checksum = sha256_json([episode.to_dict() for episode in episodes])
        return AssemblyResult(
            episodes=tuple(episodes),
            quarantined_events=tuple(quarantine),
            warnings=tuple(warnings),
            input_checksum=input_checksum,
            output_checksum=output_checksum,
        )

    @staticmethod
    def _artifact_index(artifacts: Iterable[ArtifactRef]) -> dict[str, ArtifactRef]:
        index: dict[str, ArtifactRef] = {}
        for artifact in artifacts:
            if not isinstance(artifact, ArtifactRef):
                raise TypeError("artifacts must contain ArtifactRef values")
            prior = index.get(artifact.artifact_id)
            if prior is not None and prior != artifact:
                raise ContractValidationError(
                    f"conflicting ArtifactRef for {artifact.artifact_id!r}"
                )
            index[artifact.artifact_id] = artifact
        return index

    def _assemble_episode(
        self,
        events: list[TraceEvent],
        context: EpisodeContext,
        artifact_index: Mapping[str, ArtifactRef],
        *,
        duplicate_event_count: int,
        conflicting_event_id: bool,
    ) -> AgentEpisode:
        ordered = tuple(
            sorted(events, key=lambda item: (item.sequence, item.timestamp, item.event_id))
        )
        run_ids = {event.run_id for event in ordered}
        if len(run_ids) != 1:
            raise ContractValidationError("an episode cannot contain multiple run_id values")

        warnings: list[str] = []
        sequence_counts: dict[int, int] = {}
        for event in ordered:
            sequence_counts[event.sequence] = sequence_counts.get(event.sequence, 0) + 1
        sequence_collisions = sorted(
            sequence for sequence, count in sequence_counts.items() if count > 1
        )
        if sequence_collisions:
            warnings.append(f"sequence collisions: {sequence_collisions}")
        maximum_sequence = max(sequence_counts)
        sequence_gaps = tuple(
            sequence for sequence in range(maximum_sequence + 1) if sequence not in sequence_counts
        )

        span_ids = {event.span_id for event in ordered}
        orphan_span_ids = tuple(
            sorted(
                {
                    event.span_id
                    for event in ordered
                    if event.parent_span_id is not None
                    and event.parent_span_id not in span_ids
                }
            )
        )
        missing_types = self._missing_expected_types(ordered)

        requested_artifact_ids = {
            artifact_id for event in ordered for artifact_id in event.artifact_refs
        }
        missing_artifacts = sorted(requested_artifact_ids - set(artifact_index))
        if missing_artifacts:
            warnings.append(f"missing artifact refs: {missing_artifacts}")
        resolved_artifacts = tuple(
            artifact_index[artifact_id]
            for artifact_id in sorted(requested_artifact_ids & set(artifact_index))
        )

        terminal_events = [
            event for event in ordered if event.event_type is EventType.EPISODE_FINISHED
        ]
        terminal = terminal_events[-1] if terminal_events else None
        outcome, termination, terminal_warnings = self._derive_terminal(terminal)
        warnings.extend(terminal_warnings)

        if conflicting_event_id or sequence_collisions or len(terminal_events) > 1:
            state = IntegrityState.CORRUPT
        elif sequence_gaps or orphan_span_ids or missing_types or missing_artifacts:
            state = IntegrityState.PARTIAL
        else:
            state = IntegrityState.COMPLETE

        event_digest = sha256_json(
            {
                "context": {
                    "task_id": context.task_id,
                    "attempt": context.attempt,
                    "harness": context.harness_manifest.to_dict(),
                    "model": context.model_manifest.to_dict(),
                    "environment": context.environment_manifest.to_dict(),
                    "evaluator": context.evaluator_manifest.to_dict(),
                    "experiment_manifest_ref": context.experiment_manifest_ref,
                    "capabilities": sorted(item.value for item in context.capabilities),
                },
                "events": [event.to_dict() for event in ordered],
                "artifacts": [artifact.to_dict() for artifact in resolved_artifacts],
            }
        )
        integrity = IntegrityReport(
            state=state,
            duplicate_event_count=duplicate_event_count,
            sequence_gaps=sequence_gaps,
            orphan_span_ids=orphan_span_ids,
            missing_expected_event_types=missing_types,
            assembler_version=ASSEMBLER_VERSION,
            input_checksum=sha256_json([event.to_dict() for event in events]),
            output_checksum=event_digest,
            warnings=tuple(warnings),
        )
        return AgentEpisode(
            episode_id=ordered[0].episode_id,
            run_id=ordered[0].run_id,
            task_id=context.task_id,
            attempt=context.attempt,
            harness_manifest=context.harness_manifest,
            model_manifest=context.model_manifest,
            environment_manifest=context.environment_manifest,
            evaluator_manifest=context.evaluator_manifest,
            experiment_manifest_ref=context.experiment_manifest_ref,
            capabilities=context.capabilities,
            events=ordered,
            artifact_refs=resolved_artifacts,
            started_at=min(event.timestamp for event in ordered),
            ended_at=max(event.timestamp for event in ordered),
            outcome=outcome,
            termination=termination,
            integrity=integrity,
            source_lineage={
                "assembler_version": ASSEMBLER_VERSION,
                "raw_event_checksums": [event.checksum for event in events],
                "raw_arrival_event_ids": [event.event_id for event in events],
            },
        )

    @staticmethod
    def _missing_expected_types(events: tuple[TraceEvent, ...]) -> tuple[EventType, ...]:
        present = {event.event_type for event in events}
        missing: set[EventType] = set()
        if EventType.EPISODE_FINISHED not in present:
            missing.add(EventType.EPISODE_FINISHED)
        pairs = (
            (EventType.MODEL_REQUEST, EventType.MODEL_RESPONSE),
            (EventType.TOOL_CALL, EventType.TOOL_RESULT),
            (EventType.SANDBOX_STARTED, EventType.SANDBOX_FINISHED),
            (EventType.VERIFICATION_STARTED, EventType.VERIFICATION_FINISHED),
        )
        for opened, closed in pairs:
            opened_spans = {event.span_id for event in events if event.event_type is opened}
            closed_spans = {event.span_id for event in events if event.event_type is closed}
            if opened_spans - closed_spans:
                missing.add(closed)
        return tuple(sorted(missing, key=lambda item: item.value))

    @staticmethod
    def _derive_terminal(
        terminal: TraceEvent | None,
    ) -> tuple[EpisodeOutcome, EpisodeTermination, tuple[str, ...]]:
        if terminal is None:
            return (
                EpisodeOutcome(
                    task_status=TaskStatus.UNKNOWN,
                    execution_validity=ExecutionValidity.UNKNOWN,
                    verifier_status=EpisodeVerifierStatus.UNKNOWN,
                ),
                EpisodeTermination(reason="CAPTURE_INTERRUPTED", observable=False),
                (),
            )
        attributes = terminal.attributes
        warnings: list[str] = []
        try:
            task_status = TaskStatus(attributes.get("task_status", "UNKNOWN"))
            validity = ExecutionValidity(
                attributes.get("execution_validity", "UNKNOWN")
            )
            verifier_status = EpisodeVerifierStatus(
                attributes.get("verifier_status", "UNKNOWN")
            )
            score = attributes.get("score")
            evidence = tuple(attributes.get("evidence_event_ids", ()))
            if not evidence:
                evidence = (terminal.event_id,)
            outcome = EpisodeOutcome(
                task_status=task_status,
                execution_validity=validity,
                verifier_status=verifier_status,
                score=score,
                evidence_event_ids=evidence,
            )
        except (ValueError, TypeError, ContractValidationError) as exc:
            warnings.append(f"invalid terminal outcome declaration: {exc}")
            outcome = EpisodeOutcome(
                task_status=TaskStatus.UNKNOWN,
                execution_validity=ExecutionValidity.UNKNOWN,
                verifier_status=EpisodeVerifierStatus.UNKNOWN,
                evidence_event_ids=(terminal.event_id,),
            )
        reason = attributes.get("termination_reason", "UNSPECIFIED")
        if not isinstance(reason, str) or not reason.strip():
            warnings.append("terminal event did not declare a valid termination_reason")
            reason = "UNSPECIFIED"
        termination = EpisodeTermination(
            reason=reason,
            observable=True,
            source_event_id=terminal.event_id,
        )
        return outcome, termination, tuple(warnings)
