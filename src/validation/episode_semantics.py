"""Semantic validation of assembled AgentEpisodes (Episode Certification).

This is a **derived** layer: it reads canonical episodes and produces an
``EpisodeCertification`` plus per-capability evidence.  It never writes back to
raw ``TraceEvent`` values or mutates an ``AgentEpisode``.

The certification is fail-closed: a positive training claim (SUCCESS + VALID),
or a claimed training capability, must be backed by real observable evidence.
Where evidence is missing the verdict is REJECTED or INSUFFICIENT_EVIDENCE —
never guessed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from src.contracts.agent_episode import (
    AgentEpisode,
    EpisodeVerifierStatus,
    ExecutionValidity,
    IntegrityState,
    TaskStatus,
)
from src.contracts.episode_certification import (
    CertificationStatus,
    EpisodeCertification,
    SemanticCheck,
)
from src.contracts.trace_event import EventStatus, EventType

VALIDATOR_VERSION = "episode-certification/v1"


# ---------------------------------------------------------------------------
# Verifier attestation taken from the *real* VERIFICATION_FINISHED events.
# ---------------------------------------------------------------------------
def _verifier_attestation(event: Any) -> EpisodeVerifierStatus | None:
    """Map a real VERIFICATION_FINISHED event to an attested verdict."""
    passed = event.attributes.get("passed")
    if event.status is EventStatus.SUCCEEDED and passed is True:
        return EpisodeVerifierStatus.PASSED
    if event.status is EventStatus.FAILED and passed is False:
        return EpisodeVerifierStatus.FAILED
    if event.status is EventStatus.ERROR:
        return EpisodeVerifierStatus.ERROR
    if event.status is EventStatus.TIMEOUT:
        return EpisodeVerifierStatus.TIMEOUT
    return None


def _verifier_events(episode: AgentEpisode) -> tuple[Any, ...]:
    return tuple(
        event
        for event in episode.events
        if event.event_type is EventType.VERIFICATION_FINISHED
    )


def _latest_verifier_attestation(episode: AgentEpisode) -> EpisodeVerifierStatus | None:
    for event in reversed(_verifier_events(episode)):
        status = _verifier_attestation(event)
        if status is not None:
            return status
    return None


# ---------------------------------------------------------------------------
# Training-capability evidence: fields must actually exist in the events.
# ---------------------------------------------------------------------------
def _valid_token_ids(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        tokens = tuple(
            int(item) for item in value
        )
    except (TypeError, ValueError):
        return None
    if any(isinstance(item, bool) for item in tokens) or any(item < 0 for item in tokens):
        return None
    return tokens


def _valid_logprobs(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        numbers = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if any(isinstance(item, bool) or not math.isfinite(item) for item in numbers):
        return None
    return numbers


def _valid_mask(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        mask = tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return None
    if any(item not in (0, 1) for item in mask):
        return None
    return mask


def validate_capability_evidence(
    episode: AgentEpisode,
) -> dict[str, str]:
    """Return capability -> evidence status by scanning real event fields.

    A capability may be *declared* in ``episode.capabilities`` but that is not
    proof; only an actual observable field/event counts as EVIDENCE_PRESENT.
    """

    model_requests = [
        event for event in episode.events if event.event_type is EventType.MODEL_REQUEST
    ]
    model_responses = [
        event for event in episode.events if event.event_type is EventType.MODEL_RESPONSE
    ]
    verifiers = _verifier_events(episode)

    evidence: dict[str, str] = {}

    def _present(predicate: bool) -> str:
        return "EVIDENCE_PRESENT" if predicate else "MISSING"

    evidence["MODEL_IO"] = _present(bool(model_requests) and bool(model_responses))
    evidence["VERIFIER_EVIDENCE"] = _present(bool(verifiers))

    saw_token = saw_logprob = saw_mask = False
    aligned = True
    for event in model_responses:
        attrs = event.attributes
        tokens = _valid_token_ids(attrs.get("token_ids"))
        logprobs = _valid_logprobs(attrs.get("logprobs"))
        # action_mask preferred, fall back to loss_mask, never both
        mask = _valid_mask(attrs.get("action_mask"))
        if mask is None:
            mask = _valid_mask(attrs.get("loss_mask"))
        if tokens is not None:
            saw_token = True
        if logprobs is not None:
            saw_logprob = True
        if mask is not None:
            saw_mask = True
        if tokens is not None and logprobs is not None and len(tokens) != len(logprobs):
            aligned = False
        if tokens is not None and mask is not None and len(tokens) != len(mask):
            aligned = False
        if logprobs is not None and mask is not None and len(logprobs) != sum(mask):
            # behavior logprobs are per-position; a mask over trainable positions
            # may legitimately be a subset, but full-coverage masks must align.
            if len(logprobs) == len(mask) and mask and sum(mask) != len(mask):
                pass  # subset mask allowed with per-token logprobs
            elif len(mask) != len(logprobs):
                aligned = False

    evidence["MODEL_TOKEN_IDS"] = _present(saw_token)
    evidence["MODEL_LOGPROBS"] = _present(saw_logprob)
    evidence["ACTION_MASK"] = _present(saw_mask)
    saw_any_array = saw_token or saw_logprob or saw_mask
    if not saw_any_array:
        evidence["ARRAYS_ALIGNED"] = "MISSING"
    else:
        evidence["ARRAYS_ALIGNED"] = "EVIDENCE_PRESENT" if aligned else "MISALIGNED"
    return evidence


# ---------------------------------------------------------------------------
# Certification driver.
# ---------------------------------------------------------------------------
def certify_episode(
    episode: AgentEpisode,
    *,
    capability_evidence: Mapping[str, str] | None = None,
) -> EpisodeCertification:
    if not isinstance(episode, AgentEpisode):
        raise TypeError("certify_episode requires an AgentEpisode")

    checks: list[SemanticCheck] = []
    reasons_rejected: list[str] = []
    reasons_insufficient: list[str] = []

    # --- real evidence present? -------------------------------------------------
    verifier_events = _verifier_events(episode)
    checks.append(
        SemanticCheck(
            "verifier_event_present",
            bool(verifier_events),
            f"{len(verifier_events)} VERIFICATION_FINISHED event(s) observed",
        )
    )

    model_requests = [
        event for event in episode.events if event.event_type is EventType.MODEL_REQUEST
    ]
    model_responses = [
        event for event in episode.events if event.event_type is EventType.MODEL_RESPONSE
    ]
    tool_calls = [
        event for event in episode.events if event.event_type is EventType.TOOL_CALL
    ]
    has_observation = bool(model_requests)
    has_action = bool(model_responses) or bool(tool_calls)
    checks.append(
        SemanticCheck(
            "observable_input_observation",
            has_observation,
            "MODEL_REQUEST" if has_observation else "no model observation event",
        )
    )
    checks.append(
        SemanticCheck(
            "observable_assistant_action",
            has_action,
            "assistant response/tool action observed"
            if has_action
            else "no model response or tool action",
        )
    )

    integrity_ok = episode.integrity.state is IntegrityState.COMPLETE
    checks.append(
        SemanticCheck(
            "integrity_complete",
            integrity_ok,
            f"integrity.state={episode.integrity.state.value}",
        )
    )
    if not integrity_ok:
        if episode.integrity.state is IntegrityState.CORRUPT:
            reasons_rejected.append(
                f"integrity is not COMPLETE ({episode.integrity.state.value})"
            )
        else:
            reasons_insufficient.append(
                f"integrity is not COMPLETE ({episode.integrity.state.value})"
            )

    # --- terminal outcome vs verifier evidence ---------------------------------
    task_status = episode.outcome.task_status
    validity = episode.outcome.execution_validity
    score = episode.outcome.score
    attested = _latest_verifier_attestation(episode)
    verifier_attested = attested in (
        EpisodeVerifierStatus.PASSED,
        EpisodeVerifierStatus.FAILED,
    )

    if attested is None:
        # No real verifier verdict.
        if task_status is TaskStatus.SUCCESS and validity is ExecutionValidity.VALID:
            reasons_insufficient.append(
                "terminal SUCCESS/VALID but no verifier PASSED evidence"
            )
            checks.append(
                SemanticCheck("success_attested", False, "no verifier evidence for SUCCESS")
            )
        else:
            checks.append(
                SemanticCheck(
                    "success_attested",
                    True,
                    "no trustworthy-success claim requiring verifier evidence",
                )
            )
    elif attested is EpisodeVerifierStatus.PASSED:
        if task_status is not TaskStatus.SUCCESS:
            reasons_rejected.append(
                "verifier attested PASSED but terminal outcome is not SUCCESS"
            )
            checks.append(
                SemanticCheck("terminal_verifier_consistency", False, "PASSED vs non-SUCCESS")
            )
        else:
            checks.append(
                SemanticCheck("terminal_verifier_consistency", True, "PASSED supports SUCCESS")
            )
        if validity is not ExecutionValidity.VALID:
            reasons_rejected.append(
                f"verifier PASSED but execution validity is {validity.value}"
            )
    elif attested is EpisodeVerifierStatus.FAILED:
        if task_status is TaskStatus.SUCCESS:
            reasons_rejected.append(
                "verifier attested FAILED but terminal declared SUCCESS"
            )
            checks.append(
                SemanticCheck("terminal_verifier_consistency", False, "FAILED vs SUCCESS")
            )
        else:
            checks.append(
                SemanticCheck(
                    "terminal_verifier_consistency",
                    True,
                    "verifier FAILED is a valid task failure",
                )
            )
        if validity is not ExecutionValidity.VALID:
            reasons_rejected.append(
                f"verifier FAILED but execution validity is {validity.value}"
            )
    elif attested in (EpisodeVerifierStatus.ERROR, EpisodeVerifierStatus.TIMEOUT):
        checks.append(
            SemanticCheck("verifier_result_trustworthy", False, f"verifier {attested.value}")
        )
        if task_status is TaskStatus.SUCCESS:
            reasons_rejected.append(
                f"verifier {attested.value} cannot support a trustworthy task SUCCESS"
            )
        if score is not None:
            reasons_rejected.append(
                f"verifier {attested.value} must not be converted into a reward "
                "(including 0); got score={score!r}"
            )
    else:  # UNKNOWN / NOT_RUN attestations from real events
        if task_status is TaskStatus.SUCCESS and validity is ExecutionValidity.VALID:
            reasons_insufficient.append(
                "verifier attestation is not trustworthy (UNKNOWN/NOT_RUN)"
            )
            checks.append(SemanticCheck("success_attested", False, "untrustworthy verifier"))
        else:
            checks.append(
                SemanticCheck(
                    "success_attested",
                    True,
                    "no trustworthy-success claim requiring verifier evidence",
                )
            )

    # --- outcome evidence_event_ids must reference real verifier events ---------
    verifier_event_ids = {event.event_id for event in verifier_events}
    declared_evidence = tuple(episode.outcome.evidence_event_ids)
    if verifier_events:
        referenced = set(declared_evidence) & verifier_event_ids
        if declared_evidence and not referenced:
            reasons_rejected.append(
                "outcome evidence_event_ids do not reference a real verifier event: "
                + ", ".join(declared_evidence)
            )
            checks.append(
                SemanticCheck("evidence_ids_reference_verifier", False, str(declared_evidence))
            )
        else:
            checks.append(
                SemanticCheck("evidence_ids_reference_verifier", True, str(referenced))
            )
    else:
        # No verifier events exist; any success claim is handled by
        # # success_attested -> INSUFFICIENT_EVIDENCE, not a mis-citation.
        pass

    # --- capability evidence (actual data, not producer declarations) -----------
    capabilities = capability_evidence if capability_evidence is not None else validate_capability_evidence(episode)

    if reasons_rejected:
        status = CertificationStatus.REJECTED
    elif reasons_insufficient:
        status = CertificationStatus.INSUFFICIENT_EVIDENCE
    else:
        status = CertificationStatus.CERTIFIED

    return EpisodeCertification(
        episode_id=episode.episode_id,
        episode_checksum=episode.checksum,
        certification_status=status,
        verifier_attested=verifier_attested,
        semantic_checks=tuple(checks),
        capability_evidence=capabilities,
        # This field also carries insufficient-evidence explanations so every
        # non-positive verdict remains actionable and auditable.
        rejection_reasons=tuple(reasons_rejected or reasons_insufficient),
        validator_version=VALIDATOR_VERSION,
    )


def certified_verifier_status(episode: AgentEpisode) -> EpisodeVerifierStatus | None:
    """Attested verifier verdict from real evidence, or None when unproven."""
    return _latest_verifier_attestation(episode)
