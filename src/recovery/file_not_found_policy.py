"""Learner-owned interface for bounded FILE_NOT_FOUND recovery.

This module intentionally does not implement the recovery state machine.  It
freezes the boundary between a Harness policy and the capture/data plane so
that the policy can be implemented and tested without changing canonical
``TraceEvent`` or ``AgentEpisode`` semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Protocol

import posixpath

from src.contracts._json import freeze_json, sha256_json, thaw_json
from src.errors import ContractValidationError


FILE_NOT_FOUND_POLICY_VERSION = "file-not-found-recovery/v2.1"


class RecoveryAction(str, Enum):
    """The only externally visible actions a bounded recovery policy may emit."""

    DISCOVER = "DISCOVER"
    READ = "READ"
    TERMINATE = "TERMINATE"
    NO_ACTION = "NO_ACTION"


@dataclass(frozen=True, slots=True, kw_only=True)
class RecoveryInput:
    """Facts available to a recovery policy at one decision point.

    ``discovery_cache`` is supplied as an immutable snapshot.  The policy must
    return cache/budget effects in its decision; mutation of this input is not
    part of the interface.
    """

    error_code: str
    error_path: str | None
    scope_key: str
    discovery_cache: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    remaining_discovery_budget: int
    trigger_event_ids: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.error_code, str) or not self.error_code.strip():
            raise ContractValidationError("error_code must be a non-empty string")
        if self.error_path is not None and (
            not isinstance(self.error_path, str) or not self.error_path.strip()
        ):
            raise ContractValidationError("error_path must be non-empty when provided")
        if not isinstance(self.scope_key, str) or not self.scope_key.strip():
            raise ContractValidationError("scope_key must be a non-empty string")
        if (
            isinstance(self.remaining_discovery_budget, bool)
            or not isinstance(self.remaining_discovery_budget, int)
            or self.remaining_discovery_budget < 0
        ):
            raise ContractValidationError(
                "remaining_discovery_budget must be a non-negative integer"
            )
        trigger_event_ids = tuple(self.trigger_event_ids)
        if any(not isinstance(item, str) or not item.strip() for item in trigger_event_ids):
            raise ContractValidationError("trigger_event_ids must contain non-empty strings")
        object.__setattr__(self, "trigger_event_ids", trigger_event_ids)
        if not isinstance(self.discovery_cache, Mapping):
            raise ContractValidationError("discovery_cache must be an object")
        cache: dict[str, tuple[str, ...]] = {}
        for scope, candidates in self.discovery_cache.items():
            if not isinstance(scope, str) or not scope.strip():
                raise ContractValidationError(
                    "discovery_cache scope keys must be non-empty strings"
                )
            if not isinstance(candidates, (tuple, list)):
                raise ContractValidationError("discovery_cache values must be arrays")
            normalized = tuple(candidates)
            if any(
                not isinstance(candidate, str) or not candidate.strip()
                for candidate in normalized
            ):
                raise ContractValidationError(
                    "discovery_cache candidates must be non-empty strings"
                )
            cache[scope] = normalized
        if not isinstance(self.attributes, Mapping):
            raise ContractValidationError("attributes must be an object")
        object.__setattr__(self, "discovery_cache", freeze_json(cache, "$.discovery_cache"))
        object.__setattr__(self, "attributes", freeze_json(self.attributes, "$.attributes"))


@dataclass(frozen=True, slots=True, kw_only=True)
class RecoveryDecision:
    """A deterministic policy decision that can be emitted as a Harness fact."""

    action: RecoveryAction
    reason_code: str
    trigger_event_ids: tuple[str, ...] = ()
    scope_key: str | None = None
    tool_name: str | None = None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    selected_candidates: tuple[str, ...] = ()
    budget_before: int | None = None
    budget_after: int | None = None
    cache_hit: bool | None = None
    policy_version: str = FILE_NOT_FOUND_POLICY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.action, RecoveryAction):
            raise ContractValidationError("action must be a RecoveryAction")
        if not isinstance(self.reason_code, str) or not self.reason_code.strip():
            raise ContractValidationError("reason_code must be a non-empty string")
        trigger_event_ids = tuple(self.trigger_event_ids)
        if any(not isinstance(item, str) or not item.strip() for item in trigger_event_ids):
            raise ContractValidationError("trigger_event_ids must contain non-empty strings")
        object.__setattr__(self, "trigger_event_ids", trigger_event_ids)
        if self.scope_key is not None and (
            not isinstance(self.scope_key, str) or not self.scope_key.strip()
        ):
            raise ContractValidationError("scope_key must be a non-empty string when provided")
        if self.tool_name is not None and (
            not isinstance(self.tool_name, str) or not self.tool_name.strip()
        ):
            raise ContractValidationError("tool_name must be a non-empty string when provided")
        if not isinstance(self.arguments, Mapping):
            raise ContractValidationError("arguments must be an object")
        object.__setattr__(self, "arguments", freeze_json(self.arguments, "$.arguments"))
        candidates = tuple(self.selected_candidates)
        if any(not isinstance(item, str) or not item.strip() for item in candidates):
            raise ContractValidationError(
                "selected_candidates must contain non-empty strings"
            )
        object.__setattr__(self, "selected_candidates", candidates)
        for name in ("budget_before", "budget_after"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ContractValidationError(f"{name} must be a non-negative integer")
        if self.budget_before is not None and self.budget_after is not None:
            if self.budget_after > self.budget_before:
                raise ContractValidationError("budget_after cannot exceed budget_before")
        if self.cache_hit is not None and not isinstance(self.cache_hit, bool):
            raise ContractValidationError("cache_hit must be boolean when provided")
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise ContractValidationError("policy_version must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "action": self.action.value,
            "reason_code": self.reason_code,
            "trigger_event_ids": list(self.trigger_event_ids),
            "scope_key": self.scope_key,
            "tool_name": self.tool_name,
            "arguments": thaw_json(self.arguments),
            "selected_candidates": list(self.selected_candidates),
            "budget_before": self.budget_before,
            "budget_after": self.budget_after,
            "cache_hit": self.cache_hit,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecoveryDecision":
        if not isinstance(value, Mapping):
            raise ContractValidationError("RecoveryDecision payload must be an object")
        required = {"action", "reason_code"}
        missing = sorted(required - set(value))
        allowed = {
            "policy_version",
            "action",
            "reason_code",
            "trigger_event_ids",
            "scope_key",
            "tool_name",
            "arguments",
            "selected_candidates",
            "budget_before",
            "budget_after",
            "cache_hit",
        }
        unknown = sorted(set(value) - allowed)
        if missing:
            raise ContractValidationError(f"missing RecoveryDecision fields: {missing}")
        if unknown:
            raise ContractValidationError(f"unknown RecoveryDecision fields: {unknown}")
        try:
            action = RecoveryAction(value["action"])
        except (TypeError, ValueError) as exc:
            raise ContractValidationError("action must be a valid RecoveryAction") from exc
        return cls(
            policy_version=value.get("policy_version", FILE_NOT_FOUND_POLICY_VERSION),
            action=action,
            reason_code=value["reason_code"],
            trigger_event_ids=tuple(value.get("trigger_event_ids", ())),
            scope_key=value.get("scope_key"),
            tool_name=value.get("tool_name"),
            arguments=value.get("arguments", {}),
            selected_candidates=tuple(value.get("selected_candidates", ())),
            budget_before=value.get("budget_before"),
            budget_after=value.get("budget_after"),
            cache_hit=value.get("cache_hit"),
        )

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


class FileNotFoundRecoveryPolicy(Protocol):
    """Learner-owned policy boundary; no transition logic is supplied here."""

    def decide(self, input: RecoveryInput) -> RecoveryDecision:
        """Return one bounded, deterministic decision for the supplied facts."""
        ...


class BoundedFileNotFoundRecoveryPolicy:
    """Deterministic, bounded recovery policy for structured file-not-found errors.

    The policy is intentionally stateless.  Discovery results and remaining
    budget are supplied by the caller in ``RecoveryInput`` and returned effects
    are represented by ``RecoveryDecision``.  This makes the policy safe to
    replay and prevents cache leakage across Episodes.
    """

    discovery_limit = 32
    read_offset = 1
    read_limit = 2000

    def decide(self, input: RecoveryInput) -> RecoveryDecision:
        if input.error_code != "FILE_NOT_FOUND":
            return self._decision(
                RecoveryAction.NO_ACTION,
                "UNSUPPORTED_ERROR_CODE",
                input,
            )
        if input.error_path is None:
            return self._decision(
                RecoveryAction.TERMINATE,
                "MISSING_ERROR_PATH",
                input,
            )

        cached = input.discovery_cache.get(input.scope_key)
        if cached is None:
            if input.remaining_discovery_budget == 0:
                return self._decision(
                    RecoveryAction.TERMINATE,
                    "RECOVERY_BUDGET_EXHAUSTED",
                    input,
                    budget_before=input.remaining_discovery_budget,
                    budget_after=0,
                )
            pattern = self.extract_pattern(input.error_path)
            return self._decision(
                RecoveryAction.DISCOVER,
                "FILE_NOT_FOUND_DISCOVERY",
                input,
                tool_name="find",
                arguments={
                    "path": self._normalize_scope(input.scope_key),
                    "pattern": pattern,
                    "limit": self.discovery_limit,
                },
                budget_before=input.remaining_discovery_budget,
                budget_after=input.remaining_discovery_budget - 1,
                cache_hit=False,
            )

        candidates = self._safe_candidates(input.scope_key, cached)
        if len(candidates) == 1:
            return self._decision(
                RecoveryAction.READ,
                "UNIQUE_DISCOVERY_CANDIDATE",
                input,
                tool_name="read",
                arguments={
                    "path": candidates[0],
                    "offset": self.read_offset,
                    "limit": self.read_limit,
                },
                selected_candidates=candidates,
                budget_before=input.remaining_discovery_budget,
                budget_after=input.remaining_discovery_budget,
                cache_hit=True,
            )
        if not candidates:
            return self._decision(
                RecoveryAction.TERMINATE,
                "NO_DISCOVERY_CANDIDATE",
                input,
                selected_candidates=(),
                budget_before=input.remaining_discovery_budget,
                budget_after=input.remaining_discovery_budget,
                cache_hit=True,
            )
        return self._decision(
            RecoveryAction.TERMINATE,
            "AMBIGUOUS_DISCOVERY_CANDIDATES",
            input,
            selected_candidates=candidates,
            budget_before=input.remaining_discovery_budget,
            budget_after=input.remaining_discovery_budget,
            cache_hit=True,
        )

    @classmethod
    def extract_pattern(cls, error_path: str) -> str:
        """Extract a bounded basename pattern without consulting model text."""

        normalized = error_path.replace("\\", "/")
        basename = PurePosixPath(normalized).name
        if not basename:
            return "*"
        if any(marker in basename for marker in ("*", "?", "[")):
            return basename
        stem, suffix = posixpath.splitext(basename)
        segments = [segment for segment in stem.split("-") if segment]
        if len(segments) >= 2:
            return f"*-{segments[-1]}{suffix}"
        return basename

    @staticmethod
    def _normalize_scope(scope_key: str) -> str:
        normalized = posixpath.normpath(scope_key.replace("\\", "/"))
        return "." if normalized == "" else normalized

    @classmethod
    def _safe_candidates(
        cls,
        scope_key: str,
        candidates: tuple[str, ...],
    ) -> tuple[str, ...]:
        scope = cls._normalize_scope(scope_key)
        normalized: set[str] = set()
        for candidate in candidates:
            value = candidate.replace("\\", "/")
            if not value.strip():
                continue
            candidate_path = posixpath.normpath(value)
            if not posixpath.isabs(candidate_path):
                if candidate_path == scope or candidate_path.startswith(scope + "/"):
                    resolved = candidate_path
                else:
                    resolved = posixpath.normpath(posixpath.join(scope, candidate_path))
            else:
                resolved = candidate_path
            if cls._within_scope(scope, resolved):
                normalized.add(resolved)
        return tuple(sorted(normalized))

    @staticmethod
    def _within_scope(scope: str, candidate: str) -> bool:
        if scope == ".":
            return (
                not posixpath.isabs(candidate)
                and candidate not in {".."}
                and not candidate.startswith("../")
            )
        if posixpath.isabs(scope) != posixpath.isabs(candidate):
            return False
        try:
            PurePosixPath(candidate).relative_to(PurePosixPath(scope))
        except ValueError:
            return False
        return True

    @staticmethod
    def _decision(
        action: RecoveryAction,
        reason_code: str,
        input: RecoveryInput,
        *,
        tool_name: str | None = None,
        arguments: Mapping[str, Any] | None = None,
        selected_candidates: tuple[str, ...] = (),
        budget_before: int | None = None,
        budget_after: int | None = None,
        cache_hit: bool | None = None,
    ) -> RecoveryDecision:
        return RecoveryDecision(
            action=action,
            reason_code=reason_code,
            trigger_event_ids=input.trigger_event_ids,
            scope_key=input.scope_key,
            tool_name=tool_name,
            arguments=arguments or {},
            selected_candidates=selected_candidates,
            budget_before=budget_before,
            budget_after=budget_after,
            cache_hit=cache_hit,
        )
