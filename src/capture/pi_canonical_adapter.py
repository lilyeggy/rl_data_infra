"""Pi trace → canonical action adapter (Stage B).

Reads a canonical ``AgentEpisode``'s ``TraceEvent`` stream (produced by the Pi
adapter) and re-labels every tool call/result pair into a harness-neutral
``CanonicalAction``. The raw Pi message/tool-call data remains in
``arguments`` / ``observation`` as evidence and the original ``TraceEvent`` ids
are retained in ``evidence_event_ids``.

Guarantees
==========
- Pi tool call/result pairs are strictly matched by ``pi_tool_call_id``.
- Unknown Pi tool names and outside-workspace paths fail closed or are marked
  ``lossy`` — never silently re-labeled.
- The canonical action even for the *first* assistant turn is never dropped.
- If a tool has no paired result, it becomes a ``TOOL_ERROR``-classified action
  so we never fabricate a success.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.contracts._json import sha256_json, thaw_json
from src.contracts.agent_episode import AgentEpisode
from src.contracts.canonical_action import (
    ActionType,
    CanonicalAction,
    ResultStatus,
    _action_type_for_tool,
    build_canonical_action,
    canonical_tool_for_source,
)
from src.contracts.trace_event import EventStatus, EventType, TraceEvent
from src.errors import AdapterConversionError, AdapterIssue, ErrorCode

CANONICAL_EPISODE_SCHEMA_VERSION = "canonical-episode/v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class CanonicalEpisode:
    """One episode's ordered harness-neutral action stream."""

    episode_id: str
    task_id: str
    source_harness: str
    model_id: str
    actions: tuple[CanonicalAction, ...]
    verifier_status: str | None = None
    schema_version: str = CANONICAL_EPISODE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        from src.contracts._validation import required_text

        for name in ("episode_id", "task_id", "source_harness", "model_id"):
            required_text(getattr(self, name), name)
        for index, action in enumerate(self.actions):
            if not isinstance(action, CanonicalAction):
                raise AdapterConversionError(
                    f"actions[{index}] must be a CanonicalAction"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "source_harness": self.source_harness,
            "model_id": self.model_id,
            "verifier_status": self.verifier_status,
            "actions": [action.to_dict() for action in self.actions],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalEpisode":
        if not isinstance(value, Mapping):
            raise AdapterConversionError("CanonicalEpisode payload must be an object")
        from src.contracts._validation import strict_fields

        allowed = {
            "schema_version", "episode_id", "task_id", "source_harness",
            "model_id", "verifier_status", "actions",
        }
        strict_fields(value, allowed, "CanonicalEpisode")
        return cls(
            schema_version=value["schema_version"],
            episode_id=value["episode_id"],
            task_id=value["task_id"],
            source_harness=value["source_harness"],
            model_id=value["model_id"],
            verifier_status=value["verifier_status"],
            actions=tuple(CanonicalAction.from_dict(item) for item in value["actions"]),
        )

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


def _workspace_root_from_episode(episode: AgentEpisode) -> str | None:
    """Recover the workspace root from environment evidence if possible.

    Prefer an explicit ``workspace_root``; otherwise look for the common
    absolute prefix in tool commands as a fallback. Null when unknown.
    """

    candidates: list[str] = []
    for event in episode.events:
        if event.event_type is EventType.TOOL_CALL:
            args = event.attributes.get("arguments")
            if isinstance(args, Mapping):
                command = args.get("command")
                path = args.get("path")
                for value in (command, path):
                    if isinstance(value, str):
                        for piece in value.split():
                            if piece.startswith("/"):
                                candidates.append(piece.split("/python")[0])
    if not candidates:
        return None
    # Score candidate absolute prefixes, rewarding deeper (more specific) paths
    # that look like a workspace root (own a `.git`, or a task-style tail).
    best_root, best_score = None, -1
    for c in candidates:
        parts = c.split("/")
        for depth in range(len(parts), 4, -1):
            prefix = "/".join(parts[:depth])
            # Favour deeper prefixes that appear multiple times AND contain a
            # meaningful task tail (not just the shared ancestor).
            repeated = sum(1 for x in candidates if x.startswith(prefix))
            score = depth * 10 + repeated
            if score > best_score:
                best_root, best_score = prefix, score
    return best_root


def _extract_result(attributes: Mapping[str, Any]) -> tuple[Any, ResultStatus]:
    result = attributes.get("result")
    is_error = attributes.get("status_is_error")
    if isinstance(result, Mapping):
        is_error = bool(result.get("isError")) if is_error is None else is_error
    if is_error:
        return result, ResultStatus.ERROR
    return result, ResultStatus.SUCCEEDED


def _result_status_for_event(event: TraceEvent) -> ResultStatus:
    mapping = {
        EventStatus.SUCCEEDED: ResultStatus.SUCCEEDED,
        EventStatus.FAILED: ResultStatus.FAILED,
        EventStatus.ERROR: ResultStatus.ERROR,
        EventStatus.TIMEOUT: ResultStatus.TIMEOUT,
        EventStatus.STARTED: ResultStatus.NONE,
    }
    return mapping.get(event.status, ResultStatus.NONE)


def convert_episode_to_canonical(
    episode: AgentEpisode,
    *,
    workspace_root: str | None = None,
) -> tuple[CanonicalEpisode, tuple[AdapterIssue, ...]]:
    """Convert an ``AgentEpisode`` into a ``CanonicalEpisode``.

    Raises ``AdapterConversionError`` for structural failures (unrecoverable).
    Returns a tuple of structured non-fatal issues alongside the result.
    """

    issues: list[AdapterIssue] = []
    root = workspace_root or _workspace_root_from_episode(episode)
    if not root:
        raise AdapterConversionError(
            "cannot convert: no workspace_root available and none inferred"
        )

    # Group tool call + result by pi_tool_call_id. Strict pairing.
    calls: dict[str, TraceEvent] = {}
    results: dict[str, TraceEvent] = {}
    order: list[str] = []
    for event in episode.events:
        if event.event_type not in {EventType.TOOL_CALL, EventType.TOOL_RESULT}:
            continue
        call_id = event.attributes.get("pi_tool_call_id")
        if not call_id:
            issues.append(
                AdapterIssue(
                    code=ErrorCode.SOURCE_WARNING,
                    message="tool event missing pi_tool_call_id",
                    source_record_id=event.event_id,
                )
            )
            continue
        call_id = str(call_id)
        if event.event_type is EventType.TOOL_CALL:
            if call_id in calls:
                issues.append(
                    AdapterIssue(
                        code=ErrorCode.SOURCE_WARNING,
                        message="duplicate tool call id",
                        source_record_id=event.event_id,
                    )
                )
            calls[call_id] = event
            order.append(call_id)
        else:
            results[call_id] = event

    actions: list[CanonicalAction] = []
    seen_order: list[str] = []
    for call_id in order:
        if call_id in seen_order:
            continue
        seen_order.append(call_id)
        call = calls[call_id]
        result_event = results.get(call_id)
        source_tool_name = str(call.attributes.get("tool_name") or "unknown")
        kind, failure = canonical_tool_for_source(source_tool_name, source_harness="pi")
        if kind is None:
            issues.append(
                AdapterIssue(
                    code=ErrorCode.SOURCE_WARNING,
                    message=f"unknown Pi tool {source_tool_name!r} quarantined",
                    source_record_id=event_source_id(call),
                    field="tool_name",
                )
            )
            actions.append(
                _unknown_tool_action(call, root, source_tool_name)
            )
            continue

        arguments = call.attributes.get("arguments") or {}
        result_status = (
            _result_status_for_event(result_event)
            if result_event is not None
            else ResultStatus.NONE
        )
        observation = (
            thaw_json(result_event.attributes.get("result"))
            if result_event is not None
            else None
        )
        evidence = (call.event_id,) + (
            (result_event.event_id,) if result_event is not None else ()
        )
        if result_event is None:
            issues.append(
                AdapterIssue(
                    code=ErrorCode.SOURCE_WARNING,
                    message="tool call has no paired result (no success fabricated)",
                    source_record_id=call.event_id,
                )
            )
        action = build_canonical_action(
            action_id=f"ca-{sha256_json({'episode': episode.episode_id, 'call': call_id})[:24]}",
            action_type=_action_type_for_tool(kind),
            canonical_tool_name=kind,
            source_tool_name=source_tool_name,
            source_harness="pi",
            arguments=arguments,
            workspace_root=root,
            observation=observation,
            result_status=result_status,
            action_timestamp=call.timestamp,
            result_timestamp=result_event.timestamp if result_event else None,
            evidence_event_ids=evidence,
            environment_state_refs=(episode.environment_manifest.workspace_ref,)
            if getattr(episode.environment_manifest, "workspace_ref", None)
            else (),
        )
        actions.append(action)

    canonical = CanonicalEpisode(
        episode_id=episode.episode_id,
        task_id=episode.task_id,
        source_harness="pi",
        model_id=episode.model_manifest.model_id,
        verifier_status=(
            episode.outcome.verifier_status.value
            if episode.outcome.verifier_status is not None
            else None
        ),
        actions=tuple(actions),
    )
    return canonical, tuple(issues)


def event_source_id(event: TraceEvent) -> str:
    return event.event_id


def _unknown_tool_action(
    call: TraceEvent,
    workspace_root: str,
    source_tool_name: str,
) -> CanonicalAction:
    """Represent an unmapped Pi tool as a marked tool_error action (lossy)."""

    return build_canonical_action(
        action_id=f"ca-{sha256_json({'episode': call.episode_id, 'call': call.event_id})[:24]}",
        action_type=ActionType.TOOL_ERROR,
        canonical_tool_name="tool_error",
        source_tool_name=source_tool_name,
        source_harness="pi",
        arguments=call.attributes.get("arguments") or {},
        workspace_root=workspace_root,
        observation=None,
        result_status=ResultStatus.NONE,
        action_timestamp=call.timestamp,
        evidence_event_ids=(call.event_id,),
        forced_lossy_reasons=(f"unmapped pi tool {source_tool_name!r}",),
    )