"""Strongly join one canonical Episode with producer-side policy evidence."""

from __future__ import annotations

from collections.abc import Iterable

from src.contracts._json import validate_sha256
from src.contracts.agent_episode import AgentEpisode
from src.contracts.execution_bundle import ExecutionBundle
from src.contracts.execution_identity import ExecutionIdentity
from src.errors import ContractValidationError
from src.producers.base import ProducerArtifact, ProducerCapability


def assemble_execution_bundle(
    *,
    identity: ExecutionIdentity,
    episode: AgentEpisode,
    producer_artifacts: Iterable[ProducerArtifact],
    source_artifact_checksums: Iterable[str] = (),
    verifier_report_checksum: str | None = None,
) -> ExecutionBundle:
    """Build an evidence-only bundle with no certification checksum cycle.

    Every producer artifact must carry the exact same immutable identity. Policy
    trace checksums are selected only from artifacts that actually expose token
    ids. Certification happens after this bundle exists and references its
    checksum in one direction.
    """

    if not isinstance(identity, ExecutionIdentity):
        raise TypeError("identity must be an ExecutionIdentity")
    if not isinstance(episode, AgentEpisode):
        raise TypeError("episode must be an AgentEpisode")
    _validate_episode_identity(identity, episode)

    artifacts = tuple(producer_artifacts)
    if not artifacts:
        raise ContractValidationError("execution bundle requires a producer artifact")
    artifact_checksums: list[str] = []
    policy_checksums: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, ProducerArtifact):
            raise TypeError("producer_artifacts must contain ProducerArtifact values")
        if artifact.identity != identity:
            raise ContractValidationError("producer artifact identity mismatch")
        artifact_checksums.append(artifact.checksum)
        if ProducerCapability.TOKEN_IDS in artifact.capabilities:
            policy_checksums.append(artifact.checksum)

    if len(artifact_checksums) != len(set(artifact_checksums)):
        raise ContractValidationError("duplicate producer artifact checksum")
    supplied_sources = tuple(source_artifact_checksums)
    for index, checksum in enumerate(supplied_sources):
        validate_sha256(checksum, f"source_artifact_checksums[{index}]")
    validate_sha256(verifier_report_checksum, "verifier_report_checksum")

    claims_verifier = any(
        ProducerCapability.VERIFIER_EVIDENCE in artifact.capabilities
        for artifact in artifacts
    )
    if claims_verifier and verifier_report_checksum is None:
        raise ContractValidationError(
            "verifier-capable producer artifact requires verifier report checksum"
        )

    all_sources = tuple(sorted(set(supplied_sources) | set(artifact_checksums)))
    return ExecutionBundle(
        identity=identity,
        episode_checksum=episode.checksum,
        source_artifact_checksums=all_sources,
        policy_trace_checksums=tuple(sorted(policy_checksums)),
        verifier_report_checksum=verifier_report_checksum,
    )


def _validate_episode_identity(identity: ExecutionIdentity, episode: AgentEpisode) -> None:
    expected = (
        identity.run_id,
        identity.task_id,
        identity.episode_id,
        identity.attempt_id,
    )
    observed = (episode.run_id, episode.task_id, episode.episode_id, episode.attempt)
    if observed != expected:
        raise ContractValidationError("AgentEpisode identity mismatch")
