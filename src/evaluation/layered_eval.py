"""Layered evaluation over canonical actions (Stage E / Gate 1).

Separates FORMAT ability from AGENT decision ability so a model that can emit a
legal action is not mistaken for one that picks the right action.

Views
=====
1. FormatEval       — legal canonical action, schema, parseable args, adapter-consumable.
2. ActionSelectionEval — given task + action history, compare the model's chosen
                         canonical action to the reference (next action in a
                         certified trajectory). Supports canonical-semantic,
                         tool-name, argument-normalization and
                         verifier-assisted validity, not just exact string match.
3. TrajectoryReplayEval — fixed history checks model recovers after failure,
                          adjusts after a test error, avoids repeating invalid
                          actions, tends toward shorter/efficient trajectories.

Every evaluation result is forkable (deterministic from inputs + submission),
serializable, and carries a checksum.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from src.capture.pi_canonical_adapter import CanonicalEpisode
from src.contracts._json import sha256_json, thaw_json
from src.contracts.canonical_action import (
    CANONICAL_ACTION_TYPES,
    ActionType,
)

EVAL_VERSION = "canonical-layered-eval/v1"

# A submitted action is structurally "formally valid" only if it is a parsed
# mapping with a known action_type and at least one argument key present.
VALID_PARSED_ACTIONS = ("json", "openai", "qwen")


@dataclass(frozen=True, slots=True, kw_only=True)
class FormatFailure:
    reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "detail": self.detail}


@dataclass(frozen=True, slots=True, kw_only=True)
class FormatEvalResult:
    episode_id: str
    submission_count: int
    legal_count: int
    schema_valid_count: int
    args_parseable_count: int
    adapter_consumable_count: int
    failures: tuple[FormatFailure, ...]
    version: str = EVAL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "episode_id": self.episode_id,
            "submission_count": self.submission_count,
            "legal_count": self.legal_count,
            "schema_valid_count": self.schema_valid_count,
            "args_parseable_count": self.args_parseable_count,
            "adapter_consumable_count": self.adapter_consumable_count,
            "failures": [f.to_dict() for f in self.failures],
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


def _structured_or_text(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {"raw_text": value}
    return value


def _normalize_submission_action(action: Any) -> tuple[bool, str, Any]:
    """Return (is_legal, failure_reason, parsed). Accepts JSON string,
    OpenAI function-call, Qwen toolCall-name/arguments, or plain mapping."""
    obj = _structured_or_text(action)
    if not isinstance(obj, Mapping):
        return False, "submission is not an object", None

    if "function" in obj and isinstance(obj["function"], Mapping):
        fn = obj["function"]
        obj = {"name": fn.get("name"), "arguments": fn.get("arguments")}
    elif "name" in obj and "arguments" in obj:
        # Qwen-style or plain
        pass
    elif "action_type" in obj:
        obj = {"name": obj["action_type"], "arguments": obj.get("arguments", {})}
    elif "tool" in obj:
        obj = {"name": obj["tool"], "arguments": obj.get("args", obj.get("arguments", {}))}

    name = obj.get("name")
    if not isinstance(name, str) or not name.strip():
        return False, "missing action name", obj
    if name not in CANONICAL_ACTION_TYPES:
        return False, f"unknown action name {name!r}", obj
    args = obj.get("arguments", obj.get("args", {}))
    if args is None:
        args = {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return False, "arguments JSON unparseable", obj
    if not isinstance(args, Mapping):
        return False, "arguments must be an object", obj
    return True, "", {"name": name, "arguments": args}


def evaluate_format(submitted: tuple[Any, ...], episode: CanonicalEpisode) -> FormatEvalResult:
    """Check only whether actions are legal/schema-valid/parseable/consumable.

    A format failure is explicitly labelled FORMAT_FAILURE (via FormatFailure);
    it is distinct from a wrong-but-legal action (that belongs to selection).
    """

    failures: list[FormatFailure] = []
    legal = schema_valid = args_ok = consumable = 0
    for idx, action in enumerate(submitted):
        is_legal, reason, parsed = _normalize_submission_action(action)
        if not is_legal:
            failures.append(FormatFailure(reason="FORMAT_FAILURE", detail=f"#{idx}: {reason}"))
            continue
        legal += 1
        name = parsed["name"]
        args = parsed["arguments"]
        # schema-valid: name maps to ActionType and canonical field present
        if name not in {item.value for item in ActionType}:
            failures.append(FormatFailure(reason="FORMAT_FAILURE", detail=f"#{idx}: bad type"))
            continue
        schema_valid += 1
        if not isinstance(args, Mapping):
            failures.append(
                FormatFailure(
                    reason="FORMAT_FAILURE",
                    detail=f"#{idx}: args not object",
                )
            )
            continue
        args_ok += 1
        # adapter-consumable: at least one known arg key OR a command/path present
        if args or name in {"finish", "tool_error"}:
            consumable += 1
        else:
            failures.append(
                FormatFailure(reason="FORMAT_FAILURE", detail=f"#{idx}: empty args, non-terminal")
            )
    return FormatEvalResult(
        episode_id=episode.episode_id,
        submission_count=len(submitted),
        legal_count=legal,
        schema_valid_count=schema_valid,
        args_parseable_count=args_ok,
        adapter_consumable_count=consumable,
        failures=tuple(failures),
    )


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class SelectionResult:
    episode_id: str
    correct_tool: int
    exact_match: int
    semantic_match: int
    arg_normalization_match: int
    verifier_assisted_valid: int
    attempted: int
    invalid_chosen: int
    repeated_choice: int
    version: str = EVAL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "episode_id": self.episode_id,
            "attempted": self.attempted,
            "correct_tool": self.correct_tool,
            "exact_match": self.exact_match,
            "semantic_match": self.semantic_match,
            "arg_normalization_match": self.arg_normalization_match,
            "verifier_assisted_valid": self.verifier_assisted_valid,
            "invalid_chosen": self.invalid_chosen,
            "repeated_choice": self.repeated_choice,
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


def _tool_name_of(submission: Mapping[str, Any]) -> str | None:
    return submission.get("name") or submission.get("tool") or submission.get("action_type")


def evaluate_action_selection(
    *,
    episode: CanonicalEpisode,
    history: tuple[Mapping[str, Any], ...],
    submissions: tuple[Mapping[str, Any], ...],
) -> SelectionResult:
    """Compare model-chosen actions to reference next actions.

    A reference is the real next canonical action in a certified trajectory.
    ``pending_failure`` reflects whether the prior reference action failed (used
    for verifier-assisted validity & recovery hints).
    """

    attempted = len(submissions)
    correct_tool = semantic = exact = arg_norm = verifier_valid = invalid = repeated = 0
    for idx, submission in enumerate(submissions):
        chosen = _tool_name_of(submission)
        if idx >= len(episode.actions):
            break
        reference = episode.actions[idx]
        ref_tool = reference.canonical_tool_name
        # tool-name correctness
        if chosen == ref_tool:
            correct_tool += 1
        else:
            # invalid chosen action name
            if chosen not in CANONICAL_ACTION_TYPES:
                invalid += 1
            continue
        # exact match = same tool + same normalized args
        sub_args = submission.get("arguments", submission.get("args", {}))
        if isinstance(sub_args, str):
            try:
                sub_args = json.loads(sub_args)
            except json.JSONDecodeError:
                sub_args = {}
        norm_ref = thaw_json(reference.normalized_arguments)
        if sub_args == norm_ref:
            exact += 1
        # semantic match = same tool + at least same path (normalized) if present
        if "path" in sub_args and "path" in norm_ref:
            if sub_args["path"] == norm_ref["path"]:
                semantic += 1
            elif "pattern" in sub_args or "pattern" in norm_ref:
                semantic += 1
        # argument normalization match
        if _args_overlap(sub_args, norm_ref):
            arg_norm += 1
        # verifier-assisted validity: if a prior action failed, choosing a
        # different, valid tool to inspect is good; repeating the failed call is bad.
        prev = episode.actions[idx - 1] if idx > 0 else None
        if prev is not None and prev.result_status.value in {"FAILED", "ERROR", "TIMEOUT"}:
            if chosen != prev.canonical_tool_name:
                verifier_valid += 1
            elif chosen == ref_tool and idx == len(episode.actions) - 1:
                # last action producing a result after retry
                verifier_valid += 1
            else:
                repeated += 1

    return SelectionResult(
        episode_id=episode.episode_id,
        attempted=attempted,
        correct_tool=correct_tool,
        exact_match=exact,
        semantic_match=semantic,
        arg_normalization_match=arg_norm,
        verifier_assisted_valid=verifier_valid,
        invalid_chosen=invalid,
        repeated_choice=repeated,
    )


def _args_overlap(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    keys = set(a) | set(b)
    return any(
        k for k in keys if a.get(k) is not None and a.get(k) == b.get(k)
    ) if keys else True


# ---------------------------------------------------------------------------
# Trajectory replay
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayResult:
    episode_id: str
    steps_evaluated: int
    recovered_after_failure: int
    failure_contexts: int
    avoided_repeat_invalid: int
    adjusted_after_test_error: int
    shorter_or_equal_steps: bool
    version: str = EVAL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "episode_id": self.episode_id,
            "steps_evaluated": self.steps_evaluated,
            "recovered_after_failure": self.recovered_after_failure,
            "failure_contexts": self.failure_contexts,
            "avoided_repeat_invalid": self.avoided_repeat_invalid,
            "adjusted_after_test_error": self.adjusted_after_test_error,
            "shorter_or_equal_steps": self.shorter_or_equal_steps,
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


def evaluate_trajectory_replay(
    *,
    episode: CanonicalEpisode,
    rewound_states: tuple[tuple[int, tuple[Mapping[str, Any], ...]], ...],
) -> ReplayResult:
    """Given fixed histories (rewound states), check the model's continuation.

    Each element is ``(after_index, continuation)`` where ``after_index`` is the
    index of the last already-executed reference action (``-1`` = no actions yet)
    and ``continuation`` is the sequence of actions a model would take from that
    state. We compare recovery and avoidance of repeated invalid calls with the
    reference trajectory.
    """

    recovered = failure_ctx = avoided = adjusted = 0
    total_steps = 0
    reference_total = len(episode.actions)
    ref_tools = [a.canonical_tool_name for a in episode.actions]

    for after_index, continuation in rewound_states:
        steps = len(continuation)
        total_steps += steps
        prior_index = after_index
        prior_failed = (
            episode.actions[prior_index].result_status.value
            in {"FAILED", "ERROR", "TIMEOUT"}
            if 0 <= prior_index < len(episode.actions)
            else False
        )
        prior_tool = (
            episode.actions[prior_index].canonical_tool_name
            if 0 <= prior_index < len(episode.actions)
            else None
        )
        if prior_failed:
            failure_ctx += 1
            first_chosen = _tool_name_of(continuation[0]) if continuation else None
            if (
                first_chosen
                and first_chosen != prior_tool
                and first_chosen in CANONICAL_ACTION_TYPES
            ):
                recovered += 1
            if not any(_tool_name_of(s) == prior_tool for s in continuation):
                avoided += 1
        # A model that continues from state `after_index` should pick the next
        # reference action next (or a reasonable alternative after a failure).
        next_ref = after_index + 1
        if steps > 0 and 0 <= next_ref < len(episode.actions):
            ref_next = ref_tools[next_ref]
            if _tool_name_of(continuation[0]) != ref_next:
                adjusted += 1

    return ReplayResult(
        episode_id=episode.episode_id,
        steps_evaluated=total_steps,
        recovered_after_failure=recovered,
        failure_contexts=failure_ctx,
        avoided_repeat_invalid=avoided,
        adjusted_after_test_error=adjusted,
        shorter_or_equal_steps=total_steps <= reference_total,
    )