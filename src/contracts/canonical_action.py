"""Harness-neutral canonical agent Action/Observation contract.

This is the single cross-harness representation of an agent action and its
result. Harness- or model-specific tool names, output grammars, and absolute
machine paths must NOT appear here; they belong in the per-harness adapter and
in ``raw_elements`` evidence. The canonical view uses stable semantic
``action_type`` values and normalized (workspace-relative) arguments so that the
same episode can be exported to many training views and compared across
harnesses without leaking implementation details.

Design rules
============
- ``action_type`` is drawn from a small stable vocabulary (Stage A contract).
- ``canonical_tool_name`` is the semantic tool name used in training views.
- ``source_tool_name`` / ``source_harness`` record provenance only.
- Absolute machine paths are normalized to workspace-relative paths in
  ``normalized_arguments``; the original values stay in ``arguments`` as
  evidence but consumers that need a leakage-free view use normalized values.
- ``lossy`` is ``True`` when a normalization lost information or a mapping was
  irreversible. A lossy action must never be treated as a clean training
  example without an explicit policy decision.
- Unknown tool mappings and irreversible normalizations fail closed (raise)
  or are marked ``lossy``; they are never silently re-labeled.
"""

from __future__ import annotations

import os
import posixpath
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from src.contracts._json import freeze_json, sha256_json, thaw_json
from src.contracts._validation import required_text, strict_fields, utc_instant
from src.errors import ContractValidationError

SCHEMA_VERSION = "canonical-action/v1"

# Stable, harness-neutral action vocabulary (Stage A).
CANONICAL_ACTION_TYPES = (
    "read_file",
    "search_code",
    "list_directory",
    "run_command",
    "edit_file",
    "write_file",
    "finish",
    "tool_error",
    "environment_observation",
)

# Fixed argument keys whose path semantics we know how to normalize per action.
# Values under these keys are treated as workspace-relative path candidates.
_PATH_ARGUMENT_KEYS: dict[str, tuple[str, ...]] = {
    "read_file": ("path",),
    "search_code": ("path", "pattern"),
    "list_directory": ("path",),
    "edit_file": ("path",),
    "write_file": ("path",),
    "run_command": ("command",),
    "environment_observation": ("cwd",),
}


class ActionType(str, Enum):
    READ_FILE = "read_file"
    SEARCH_CODE = "search_code"
    LIST_DIRECTORY = "list_directory"
    RUN_COMMAND = "run_command"
    EDIT_FILE = "edit_file"
    WRITE_FILE = "write_file"
    FINISH = "finish"
    TOOL_ERROR = "tool_error"
    ENVIRONMENT_OBSERVATION = "environment_observation"


class ResultStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    NONE = "NONE"


# ---------------------------------------------------------------------------
# Path normalization helpers (workspace-relative)
# ---------------------------------------------------------------------------

# Sentinel to place where no meaningful normalization applied.
_NO_NORMALIZATION = "__raw__"


def _is_absolute_path(value: str) -> bool:
    return value.startswith("/") or value.startswith("~/") or value == "~"


def normalize_path_argument(
    value: str,
    *,
    workspace_root: str,
    source_harness: str = "pi",
) -> tuple[str, bool]:
    """Return ``(normalized_value, lossy)``.

    - A path equal to or under ``workspace_root`` is rewritten relative to it.
    - A relative path is kept as-is (already workspace-relative).
    - A path outside the workspace is left unchanged but marked ``lossy`` so
      downstream consumers know it may leak host layout.
    """

    root = workspace_root.rstrip("/")
    if not value:
        return value, False
    if not _is_absolute_path(value):
        # Relative path: already workspace-relative in Pi; keep, not lossy.
        return _normalize_rel(value), False

    expanded = os.path.expanduser(value)
    # Normalize away ".." and "." lexically on the workspace prefix.
    norm = posixpath.normpath(expanded)
    if norm == root:
        return ".", False
    if norm.startswith(root + "/"):
        rel = norm[len(root) + 1 :]
        if not rel:
            rel = "."
        # Keep ".."  that were part of the *usage* outside the workspace marked lossy.
        inside = not _escapes_workspace(rel)
        return _normalize_rel(rel), not inside
    if norm.startswith(root) and norm != root:
        # e.g. root is a prefix of a sibling path (/a/bc vs /a/b) — treat as outside
        return expanded, True
    # Path entirely outside workspace.
    return expanded, True


def _normalize_rel(value: str) -> str:
    cleaned = posixpath.normpath(value)
    if cleaned == "." and value not in (".", ""):
        # "x/../." collapses to "."; acceptable lexical normalization.
        return "."
    return cleaned


def _escapes_workspace(relative: str) -> bool:
    if relative.startswith(".."):
        return True
    return False


def normalize_command_for_workspace(
    command: str,
    *,
    workspace_root: str,
) -> tuple[str, bool]:
    """Rewrite the workspace root inside a shell command to a neutral token.

    This is deliberately conservative: only literal occurrences of the
    workspace root are replaced, so information about the command's intent is
    retained while the absolute host path is removed. ``lossy`` reflects
    whether the command still references other absolute paths or a `cd` to a
    non-workspace location.
    """

    if not command:
        return command, False
    root = workspace_root.rstrip("/")
    if root in command:
        rest = command.split(root, 1)[-1]
        # If the character right after the root is a path separator, keep the
        # tail as a relative path; otherwise the root ends a filename/`cd` token
        # (followed by whitespace or a shell operator), so emit bare $WORKSPACE.
        if rest.startswith("/"):
            remaining_norm = rest.lstrip("/")
            neutral = (
                f"$WORKSPACE/{remaining_norm}" if remaining_norm else "$WORKSPACE"
            )
        else:
            neutral = "$WORKSPACE" + rest
        # Additional absolute references elsewhere ⇒ lossy.
        leftovers = _other_absolute_refs(command, root)
        return neutral, bool(leftovers)
    return command, _has_absolute_refs(command)


def _has_absolute_refs(command: str) -> bool:
    for token in command.replace("=", " ").replace("\n", " ").split():
        if token.startswith("/") and not token.startswith("/dev/"):
            return True
    return False


def _other_absolute_refs(command: str, root: str) -> list[str]:
    refs: list[str] = []
    for token in command.replace("=", " ").replace("\n", " ").split():
        if token.startswith("/") and not token.startswith(root):
            refs.append(token)
    return sorted(set(refs))


# ---------------------------------------------------------------------------
# CanonicalAction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class CanonicalAction:
    action_id: str
    action_type: ActionType
    canonical_tool_name: str
    arguments: Mapping[str, Any]
    normalized_arguments: Mapping[str, Any]
    observation: Any
    result_status: ResultStatus
    source_tool_name: str
    source_harness: str
    parent_action_id: str | None = None
    action_timestamp: str | None = None
    result_timestamp: str | None = None
    evidence_event_ids: tuple[str, ...] = ()
    environment_state_refs: tuple[str, ...] = ()
    lossy: bool = False
    lossy_reasons: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        required_text(self.action_id, "action_id")
        required_text(self.canonical_tool_name, "canonical_tool_name")
        required_text(self.source_tool_name, "source_tool_name")
        required_text(self.source_harness, "source_harness")
        if self.action_type not in ActionType:
            raise ContractValidationError("action_type must be a canonical ActionType")
        if self.action_type not in {item for item in ActionType}:
            raise ContractValidationError("action_type must be a known canonical type")
        if self.canonical_tool_name not in CANONICAL_ACTION_TYPES:
            raise ContractValidationError(
                f"unknown canonical_tool_name {self.canonical_tool_name!r}"
            )
        if self.parent_action_id is not None:
            required_text(self.parent_action_id, "parent_action_id")
        if self.parent_action_id == self.action_id:
            raise ContractValidationError("parent_action_id cannot equal action_id")

        object.__setattr__(self, "arguments", freeze_json(self.arguments, "$.arguments"))
        object.__setattr__(
            self, "normalized_arguments",
            freeze_json(self.normalized_arguments, "$.normalized_arguments"),
        )
        object.__setattr__(self, "observation", freeze_json(self.observation, "$.observation"))
        for name in ("action_timestamp", "result_timestamp"):
            value = getattr(self, name)
            if value is not None:
                utc_instant(value, name)
        for name in ("evidence_event_ids", "environment_state_refs"):
            items = tuple(getattr(self, name))
            for index, item in enumerate(items):
                required_text(item, f"{name}[{index}]")
            object.__setattr__(self, name, items)
        if not self.lossy and self.lossy_reasons:
            raise ContractValidationError("lossy_reasons require lossy=True")
        if self.lossy and not self.lossy_reasons:
            raise ContractValidationError("lossy actions must carry lossy_reasons")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "canonical_tool_name": self.canonical_tool_name,
            "arguments": thaw_json(self.arguments),
            "normalized_arguments": thaw_json(self.normalized_arguments),
            "observation": thaw_json(self.observation),
            "result_status": self.result_status.value,
            "source_tool_name": self.source_tool_name,
            "source_harness": self.source_harness,
            "parent_action_id": self.parent_action_id,
            "action_timestamp": self.action_timestamp,
            "result_timestamp": self.result_timestamp,
            "evidence_event_ids": list(self.evidence_event_ids),
            "environment_state_refs": list(self.environment_state_refs),
            "lossy": self.lossy,
            "lossy_reasons": list(self.lossy_reasons),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalAction":
        if not isinstance(value, Mapping):
            raise ContractValidationError("CanonicalAction payload must be an object")
        allowed = {
            "schema_version", "action_id", "action_type", "canonical_tool_name",
            "arguments", "normalized_arguments", "observation", "result_status",
            "source_tool_name", "source_harness", "parent_action_id",
            "action_timestamp", "result_timestamp", "evidence_event_ids",
            "environment_state_refs", "lossy", "lossy_reasons",
        }
        strict_fields(value, allowed, "CanonicalAction")
        return cls(
            schema_version=value["schema_version"],
            action_id=value["action_id"],
            action_type=ActionType(value["action_type"]),
            canonical_tool_name=value["canonical_tool_name"],
            arguments=value["arguments"],
            normalized_arguments=value["normalized_arguments"],
            observation=value["observation"],
            result_status=ResultStatus(value["result_status"]),
            source_tool_name=value["source_tool_name"],
            source_harness=value["source_harness"],
            parent_action_id=value["parent_action_id"],
            action_timestamp=value["action_timestamp"],
            result_timestamp=value["result_timestamp"],
            evidence_event_ids=tuple(value["evidence_event_ids"]),
            environment_state_refs=tuple(value["environment_state_refs"]),
            lossy=value["lossy"],
            lossy_reasons=tuple(value["lossy_reasons"]),
        )

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


# ---------------------------------------------------------------------------
# Normalizer: built from a deterministic mapping of Pi tool names → canonical.
# ---------------------------------------------------------------------------

# Pi tool name → canonical semantic tool name. Only these appear here; every
# other Pi tool must be rejected or explicitly classified as lossy.
PI_TO_CANONICAL: dict[str, str] = {
    "read": "read_file",
    "grep": "search_code",
    "find": "search_code",
    "ls": "list_directory",
    "bash": "run_command",
    "edit": "edit_file",
    "write": "write_file",
    "glob": "search_code",
    "finish": "finish",
}


def canonical_tool_for_source(
    source_tool_name: str,
    *,
    source_harness: str = "pi",
) -> tuple[str, str | None]:
    """Map a source tool name into (canonical_tool_name, failure_reason).

    For Pi, unknown tools are rejected (fail closed). Returns ``(kind, None)``
    on success and ``(None, reason)`` on failure.
    """

    if source_harness != "pi":
        return None, f"unsupported source_harness {source_harness!r}"
    kind = PI_TO_CANONICAL.get(source_tool_name)
    if kind is None:
        return None, f"unknown pi tool name {source_tool_name!r}"
    return kind, None


def _action_type_for_tool(kind: str) -> ActionType:
    if kind == "tool_error":
        return ActionType.TOOL_ERROR
    return ActionType(kind)


def build_normalized_args(
    *,
    source_tool_name: str,
    kind: str,
    arguments: Mapping[str, Any],
    workspace_root: str,
) -> tuple[Mapping[str, Any], list[str]]:
    """Normalize path-valued argument keys to workspace-relative form.

    Returns ``(normalized_arguments, lossy_reasons)``. Runs/commands are handled
    specially by rewriting the workspace root token. Unknown argument keys are
    passed through untouched (they hold no path semantics we can prove).
    """

    norm: dict[str, Any] = {}
    lossy_reasons: list[str] = []
    for key, value in arguments.items():
        if kind == "run_command" and key == "command" and isinstance(value, str):
            cmd_norm, cmd_lossy = normalize_command_for_workspace(
                value, workspace_root=workspace_root
            )
            norm[key] = cmd_norm
            if cmd_lossy:
                lossy_reasons.append(
                    f"command still references absolute path(s) outside workspace {key}"
                )
            continue
        if key in _PATH_ARGUMENT_KEYS.get(kind, ()) and isinstance(value, str):
            path_norm, path_lossy = normalize_path_argument(
                value, workspace_root=workspace_root
            )
            norm[key] = path_norm
            if path_lossy:
                lossy_reasons.append(
                    f"path argument {key!r} is outside the workspace (host-path leak potential)"
                )
            continue
        norm[key] = value
    return norm, lossy_reasons


def build_canonical_action(
    *,
    action_id: str,
    action_type: ActionType,
    canonical_tool_name: str,
    source_tool_name: str,
    source_harness: str,
    arguments: Mapping[str, Any],
    workspace_root: str,
    observation: Any = None,
    result_status: ResultStatus = ResultStatus.NONE,
    parent_action_id: str | None = None,
    action_timestamp: str | None = None,
    result_timestamp: str | None = None,
    evidence_event_ids: tuple[str, ...] = (),
    environment_state_refs: tuple[str, ...] = (),
    forced_lossy_reasons: tuple[str, ...] = (),
) -> CanonicalAction:
    normalized, lossy_reasons = build_normalized_args(
        source_tool_name=source_tool_name,
        kind=canonical_tool_name,
        arguments=arguments,
        workspace_root=workspace_root,
    )
    reasons = list(lossy_reasons) + list(forced_lossy_reasons)
    return CanonicalAction(
        action_id=action_id,
        action_type=action_type,
        canonical_tool_name=canonical_tool_name,
        arguments=arguments,
        normalized_arguments=normalized,
        observation=observation,
        result_status=result_status,
        source_tool_name=source_tool_name,
        source_harness=source_harness,
        parent_action_id=parent_action_id,
        action_timestamp=action_timestamp,
        result_timestamp=result_timestamp,
        evidence_event_ids=evidence_event_ids,
        environment_state_refs=environment_state_refs,
        lossy=bool(reasons),
        lossy_reasons=tuple(dict.fromkeys(reasons)),
    )