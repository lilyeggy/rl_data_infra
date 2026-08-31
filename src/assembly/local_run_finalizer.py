"""Join a persisted Local Launcher run into one canonical evidence bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.assembly.episode_assembler import EpisodeAssembler, EpisodeContext
from src.assembly.execution_bundle_assembler import assemble_execution_bundle
from src.capture.event_writer import EventJsonlReader
from src.capture.model_proxy import ModelCallEvidence
from src.contracts._json import sha256_bytes
from src.contracts.agent_episode import AgentEpisode
from src.contracts.artifacts import ArtifactRef
from src.contracts.execution_bundle import ExecutionBundle
from src.contracts.run_manifest import ExecutionRunManifest
from src.contracts.verifier_report import LocalVerifierReport
from src.errors import ContractValidationError
from src.producers import ProducerArtifact


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalRunFinalization:
    episode: AgentEpisode
    execution_bundle: ExecutionBundle
    model_evidence: tuple[ModelCallEvidence, ...]
    warnings: tuple[str, ...]


def finalize_local_run(
    *,
    manifest: ExecutionRunManifest,
    producer_artifact: ProducerArtifact,
    events_path: str | Path,
    model_evidence_path: str | Path,
    artifacts_path: str | Path | None = None,
) -> LocalRunFinalization:
    if producer_artifact.identity != manifest.identity:
        raise ContractValidationError("producer artifact/run manifest identity mismatch")
    if producer_artifact.payload.get("launch_plan_checksum") != (manifest.launcher_plan_checksum):
        raise ContractValidationError("producer artifact/run manifest launch plan mismatch")
    event_result = EventJsonlReader.read(events_path)
    if event_result.issues:
        raise ContractValidationError("raw event log contains malformed records")
    evidence = _read_model_evidence(model_evidence_path)
    if any(item.request.identity != manifest.identity for item in evidence):
        raise ContractValidationError("model evidence/run manifest identity mismatch")
    artifacts = _read_artifacts(artifacts_path)
    verifier_report_checksum = producer_artifact.payload.get("verifier_report_checksum")
    verifier_report = _read_verifier_report(
        checksum=verifier_report_checksum,
        artifacts=artifacts,
        manifest=manifest,
        producer_artifact=producer_artifact,
    )
    assembly = EpisodeAssembler().assemble(
        event_result.events,
        contexts={
            manifest.identity.episode_id: EpisodeContext(
                task_id=manifest.identity.task_id,
                attempt=manifest.identity.attempt_id,
                harness_manifest=manifest.harness_manifest,
                model_manifest=manifest.model_manifest,
                environment_manifest=manifest.environment_manifest,
                evaluator_manifest=manifest.evaluator_manifest,
                experiment_manifest_ref=manifest.experiment_manifest_ref,
                capabilities=manifest.capture_capabilities,
            )
        },
        artifacts=artifacts,
    )
    if assembly.quarantined_events or len(assembly.episodes) != 1:
        raise ContractValidationError("local run did not assemble into exactly one Episode")
    episode = assembly.episodes[0]
    if (
        verifier_report is not None
        and episode.outcome.verifier_status is not verifier_report.verifier_status
    ):
        raise ContractValidationError("verifier report/terminal outcome mismatch")
    bundle = assemble_execution_bundle(
        identity=manifest.identity,
        episode=episode,
        producer_artifacts=(producer_artifact,),
        source_artifact_checksums=(
            *(item.checksum for item in evidence),
            *(item.sha256 for item in artifacts),
        ),
        verifier_report_checksum=verifier_report_checksum,
    )
    warnings = tuple(assembly.warnings)
    if not evidence:
        warnings += ("run contains no model-call evidence",)
    return LocalRunFinalization(
        episode=episode,
        execution_bundle=bundle,
        model_evidence=evidence,
        warnings=warnings,
    )


def _read_model_evidence(path: str | Path) -> tuple[ModelCallEvidence, ...]:
    source = Path(path)
    if not source.exists():
        return ()
    values = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                values.append(ModelCallEvidence.from_dict(json.loads(line)))
            except Exception as exc:
                raise ContractValidationError(
                    f"invalid model evidence at line {line_number}: {exc}"
                ) from exc
    return tuple(values)


def _read_artifacts(path: str | Path | None) -> tuple[ArtifactRef, ...]:
    if path is None:
        return ()
    source = Path(path)
    if not source.exists():
        return ()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractValidationError(f"invalid artifact manifest: {exc}") from exc
    if not isinstance(payload, list):
        raise ContractValidationError("artifact manifest must be an array")
    try:
        artifacts = tuple(ArtifactRef.from_dict(item) for item in payload)
    except Exception as exc:
        raise ContractValidationError(f"invalid artifact reference: {exc}") from exc
    if len({item.artifact_id for item in artifacts}) != len(artifacts):
        raise ContractValidationError("artifact manifest contains duplicate IDs")
    for artifact in artifacts:
        if "://" in artifact.uri:
            continue
        object_path = Path(artifact.uri)
        if not object_path.is_absolute() or not object_path.is_file():
            raise ContractValidationError(
                f"local artifact object is missing: {artifact.artifact_id}"
            )
        content = object_path.read_bytes()
        if len(content) != artifact.size_bytes or sha256_bytes(content) != artifact.sha256:
            raise ContractValidationError(
                f"local artifact object failed checksum: {artifact.artifact_id}"
            )
    return artifacts


def _read_verifier_report(
    *,
    checksum: object,
    artifacts: tuple[ArtifactRef, ...],
    manifest: ExecutionRunManifest,
    producer_artifact: ProducerArtifact,
) -> LocalVerifierReport | None:
    if checksum is None:
        return None
    matches = [
        item for item in artifacts if item.sha256 == checksum and item.kind == "verifier-report"
    ]
    if len(matches) != 1:
        raise ContractValidationError(
            "producer verifier report checksum must resolve to one verifier-report artifact"
        )
    reference = matches[0]
    if "://" in reference.uri:
        raise ContractValidationError(
            "local finalization requires a locally resolvable verifier report"
        )
    try:
        report = LocalVerifierReport.from_dict(
            json.loads(Path(reference.uri).read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise ContractValidationError(f"invalid verifier report: {exc}") from exc
    if report.checksum != reference.sha256:
        raise ContractValidationError("verifier report semantic checksum mismatch")
    if report.identity_checksum != manifest.identity.checksum:
        raise ContractValidationError("verifier report/run identity mismatch")
    if report.evaluator != manifest.evaluator_manifest:
        raise ContractValidationError("verifier report/evaluator manifest mismatch")
    if report.producer_artifact_checksum != producer_artifact.payload.get(
        "verifier_artifact_checksum"
    ):
        raise ContractValidationError("verifier report/producer artifact mismatch")
    missing_outputs = set(report.output_artifact_checksums) - {item.checksum for item in artifacts}
    if missing_outputs:
        raise ContractValidationError("verifier report references unknown output artifacts")
    return report
