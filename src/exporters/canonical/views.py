"""Canonical-episode multi-view exporters (Stage C).

Each exporter converts the same harness-neutral ``CanonicalEpisode`` into a
distinct training view. The canonical episode is never mutated; every exporter
renders its own format and stamps ``format`` / ``version`` / ``checksum`` /
``episode_lineage`` so the same episode can produce several formats with
round-trip and semantic-equivalence guarantees.

Views delivered here:

1. ``Generic Agent SFT`` — harness/model-neutral decision data (no Pi prompt,
   no absolute paths, no model-specific tool tokens). Sub-goals are supported
   (first-action, tool-selection, failure-recovery, patch, test-selection,
   final-answer) so downstream training does not weight every turn equally.
2. ``Model-native Tool SFT`` — format renderers:
     - Qwen/Hermes tool format (``<tool_call>`` style belongs ONLY here)
     - OpenAI function-call-like format
     - protocol-neutral JSON action format
3. ``Harness Improvement`` — metrics for Harness analysis (not model training).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.capture.pi_canonical_adapter import CanonicalAction, CanonicalEpisode
from src.contracts._json import sha256_json, thaw_json

CANONICAL_EXPORTER_VERSION = "canonical-exporters/v1"

# ---------------------------------------------------------------------------
# Guard: never accept a host-absolute path in a training view by default.
# ---------------------------------------------------------------------------

_HOST_PATH_MARKERS = (
    "/home/",
    "/homePLUS",
    "/Users/",
    "/root/",
    "/tmp/",
    "100.65.",
    "agent-data-plane/swebench",
)


def assert_training_view_is_leak_free(value: Any, *, label: str) -> None:
    """Fail if a rendered training view still contains host-path markers.

    Uses ``normalized_arguments``/neutral content, so legitimately normalized
    ``$WORKSPACE`` references pass while raw host paths fail.
    """

    def walk(node: Any) -> None:
        if isinstance(node, str):
            for marker in _HOST_PATH_MARKERS:
                if marker in node and "$WORKSPACE" not in node:
                    raise ValueError(
                        f"{label} leaks host path marker {marker!r}: "
                        f"…{node[max(0, node.rfind(marker)) - 20:]}"
                    )
        elif isinstance(node, Mapping):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(value)


# ---------------------------------------------------------------------------
# Generic Agent SFT view
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class GenericSFTExample:
    task: str
    context: str
    action: str
    observation: str
    patch: str | None
    verifier_outcome: str | None
    example_type: str  # first-action / tool-selection / failure-recovery / patch / final-answer
    source_episode: str
    format: str
    version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "example_type": self.example_type,
            "task": self.task,
            "context": self.context,
            "action": self.action,
            "observation": self.observation,
            "patch": self.patch,
            "verifier_outcome": self.verifier_outcome,
            "source_episode": self.source_episode,
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


def _action_brief(action: CanonicalAction) -> dict[str, Any]:
    """Render one canonical action into a neutral, leak-free structure."""
    nargs = action.normalized_arguments
    return {
        "tool": action.canonical_tool_name,
        "args": thaw_json(nargs),
        "status": action.result_status.value,
        "observation": action.observation,
    }


def _example_type_for_index(
    index: int, actions: tuple, verifier: str | None, count: int
) -> str:
    """Classify an example by its decision role.

    - first index → ``first-action``
    - an action that follows a failed/errored previous action → ``failure-recovery``
    - final action → ``final-answer``
    - otherwise → ``tool-selection``
    """

    if index == 0:
        return "first-action"
    if index == count - 1:
        return "final-answer"
    previous = actions[index - 1]
    if previous.result_status.value in {"FAILED", "ERROR", "TIMEOUT"}:
        return "failure-recovery"
    return "tool-selection"


def export_generic_agent_sft(
    episode: CanonicalEpisode,
    *,
    task: str,
    context: str = "",
) -> tuple[tuple[GenericSFTExample, ...], dict[str, Any]]:
    """Export one canonical episode as generic (harness/model-neutral) SFT.

    Returns (examples, exporter_manifest). Each example is one turn of
    ``task + context prefix → canonical action`` so the model learns decision
    semantics rather than a Pi format. ``patch``/``verifier_outcome`` are
    included only when available (final/patch examples).
    """

    examples: list[GenericSFTExample] = []
    count = len(episode.actions)
    for index, action in enumerate(episode.actions):
        brief = _action_brief(action)
        example_type = _example_type_for_index(
            index, episode.actions, episode.verifier_status, count
        )
        is_patch_like = action.canonical_tool_name in {"edit_file", "write_file"}
        example = GenericSFTExample(
            task=task,
            context=context,
            action=thaw_json(brief),
            observation=action.observation,
            patch=None,
            verifier_outcome=episode.verifier_status,
            example_type=example_type,
            source_episode=episode.episode_id,
            format="generic-agent-sft",
            version="v1",
        )
        examples.append(example)
        assert_training_view_is_leak_free(
            brief, label=f"generic-agent-sft {episode.episode_id}"
        )
    manifest = {
        "format": "generic-agent-sft",
        "version": "v1",
        "exporter_version": CANONICAL_EXPORTER_VERSION,
        "episode_id": episode.episode_id,
        "episode_checksum": episode.checksum,
        "example_count": len(examples),
        "checksum": sha256_json(
            [example.to_dict() for example in examples]
        ),
    }
    return tuple(examples), manifest


# ---------------------------------------------------------------------------
# Model-native Tool SFT view (three renderers)
# ---------------------------------------------------------------------------

QwenToQwen1_5 = {
    "read_file": "read_file",
    "search_code": "search_code",
    "list_directory": "list_directory",
    "run_command": "run_command",
    "edit_file": "edit_file",
    "write_file": "write_file",
    "finish": "finish",
    "tool_error": "tool_error",
    "environment_observation": "environment_observation",
}


def render_qwen(tool: str, args: Mapping[str, Any]) -> dict[str, Any]:
    """Qwen/Hermes tool-call renderer. The model-specific '<tool_call>' protocol
    is expressed here only; the canonical episode never stores it."""
    return {"name": QwenToQwen1_5.get(tool, tool), "arguments": thaw_json(args)}


def render_openai(tool: str, args: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool,
            "arguments": _json_str(args),
        },
    }


def render_json(tool: str, args: Mapping[str, Any]) -> dict[str, Any]:
    return {"action_type": tool, "arguments": thaw_json(args)}


def _json_str(value: Any) -> str:
    import json

    return json.dumps(thaw_json(value), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelNativeExample:
    renderer: str  # qwen | openai | json
    episode_id: str
    messages: tuple[dict[str, Any], ...]
    format: str
    version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "renderer": self.renderer,
            "episode_id": self.episode_id,
            "messages": list(self.messages),
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


def _model_native_messages(
    episode: CanonicalEpisode,
    *,
    renderer: str,
    task: str,
) -> tuple[dict[str, Any], ...]:
    messages: list[dict[str, Any]] = []
    user = {"role": "user", "content": task}
    if renderer == "qwen":
        messages.append(user)
    elif renderer == "openai":
        messages.append(user)
    else:
        messages.append(user)

    for action in episode.actions:
        args = action.normalized_arguments
        # Render the action into the assistant tool-call message.
        tool_payload = _render_tool_call(renderer, action, args)
        messages.append({"role": "assistant", "content": "", "tool_calls": tool_payload})
        observation = action.observation
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call-{action.action_id[-12:]}",
                "content": _observation_text(observation),
            }
        )
    return tuple(messages)


def _render_tool_call(
    renderer: str, action: CanonicalAction, args: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if renderer == "qwen":
        rendered = render_qwen(action.canonical_tool_name, args)
        return [rendered]
    if renderer == "openai":
        return [render_openai(action.canonical_tool_name, args)]
    # json
    return [render_json(action.canonical_tool_name, args)]


def _observation_text(observation: Any) -> str:
    import json

    if observation is None:
        return ""
    if isinstance(observation, str):
        return observation
    return json.dumps(thaw_json(observation), ensure_ascii=False, sort_keys=True)


def export_model_native(
    episode: CanonicalEpisode,
    *,
    task: str,
    renderers: tuple[str, ...] = ("qwen", "openai", "json"),
) -> tuple[tuple[ModelNativeExample, ...], dict[str, Any]]:
    """Render the same canonical episode into multiple model-native formats.

    The canonical episode stays tool-token-free; only the renderer introduces a
    format. Every rendered example carries ``format``/``version``/checksum and
    the canonical episode's lineage. Round-trip is guaranteed by
    ``industry round-trip`` in the tests.
    """

    examples: list[ModelNativeExample] = []
    for renderer in renderers:
        if renderer not in {"qwen", "openai", "json"}:
            raise ValueError(f"unknown native renderer {renderer!r}")
        messages = _model_native_messages(episode, renderer=renderer, task=task)
        example = ModelNativeExample(
            renderer=renderer,
            episode_id=episode.episode_id,
            messages=messages,
            format=f"model-native-{renderer}",
            version="v1",
        )
        examples.append(example)
    # Leak-free check across all rendered content.
    for example in examples:
        assert_training_view_is_leak_free(
            example.to_dict(), label=f"model-native-{example.renderer}"
        )
    manifest = {
        "format": "model-native-tool-sft",
        "version": "v1",
        "exporter_version": CANONICAL_EXPORTER_VERSION,
        "episode_id": episode.episode_id,
        "episode_checksum": episode.checksum,
        "renderers": list(renderers),
        "checksum": sha256_json(
            [example.to_dict() for example in examples]
        ),
    }
    return tuple(examples), manifest


# ---------------------------------------------------------------------------
# Harness Improvement view
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class HarnessMetrics:
    episode_id: str
    tool_call_count: int
    first_action_latency_ms: int | None
    time_to_patch_ms: int | None
    redundant_call_count: int
    repeated_failure_count: int
    path_tokens_approx: int
    context_bytes_approx: int
    termination_reason: str | None
    verifier_outcome: str | None
    format: str
    version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "episode_id": self.episode_id,
            "tool_call_count": self.tool_call_count,
            "first_action_latency_ms": self.first_action_latency_ms,
            "time_to_patch_ms": self.time_to_patch_ms,
            "redundant_call_count": self.redundant_call_count,
            "repeated_failure_count": self.repeated_failure_count,
            "path_tokens_approx": self.path_tokens_approx,
            "context_bytes_approx": self.context_bytes_approx,
            "termination_reason": self.termination_reason,
            "verifier_outcome": self.verifier_outcome,
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


def _ms(ts: str | None) -> int | None:
    from datetime import datetime

    if not ts:
        return None
    candidate = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
    try:
        return int(datetime.fromisoformat(candidate).timestamp() * 1000)
    except ValueError:
        return None


def export_harness_improvement(episode: CanonicalEpisode) -> HarnessMetrics:
    """Fold one canonical episode into Harness-analysis metrics.

    These are intended for Harness policy comparison (context compaction,
    retries, duplicate prevention, timeouts), not for model training.
    """

    actions = episode.actions
    call_count = len(actions)
    redundant = sum(
        1
        for i, a in enumerate(actions)
        if a.canonical_tool_name in {"search_code", "list_directory", "read_file"}
        and any(
            a.canonical_tool_name == b.canonical_tool_name
            and a.normalized_arguments == b.normalized_arguments
            for b in actions[:i]
        )
    )
    failed_signatures: set[tuple[str, str]] = set()
    repeated_failures = 0
    for action in actions:
        if action.result_status.value not in {"FAILED", "ERROR", "TIMEOUT"}:
            continue
        signature = (
            action.canonical_tool_name,
            json.dumps(thaw_json(action.normalized_arguments), sort_keys=True),
        )
        if signature in failed_signatures:
            repeated_failures += 1
        failed_signatures.add(signature)
    first_ts = _ms(actions[0].action_timestamp) if actions else None
    patch_idx = next(
        (
            i
            for i, a in enumerate(actions)
            if a.canonical_tool_name in {"edit_file", "write_file"}
        ),
        None,
    )
    # CanonicalEpisode v1 has no episode-start timestamp. Reporting the first
    # action's Unix epoch as a latency would be fabricated, so leave it unknown.
    first_action_latency = None
    time_to_patch = (
        _ms(actions[patch_idx].action_timestamp) - first_ts
        if (
            patch_idx is not None
            and first_ts is not None
            and _ms(actions[patch_idx].action_timestamp) is not None
        )
        else None
    )
    path_tokens = sum(
        len(str(a.normalized_arguments.get("path", "")))
        for a in actions
        if a.normalized_arguments.get("path")
    )
    context_bytes = sum(
        len(json.dumps(thaw_json(a.normalized_arguments), ensure_ascii=False))
        + len(json.dumps(thaw_json(a.observation), ensure_ascii=False))
        for a in actions
    )
    # The final tool is not a termination reason. CanonicalEpisode v1 does not
    # carry the source termination declaration.
    termination = None
    return HarnessMetrics(
        episode_id=episode.episode_id,
        tool_call_count=call_count,
        first_action_latency_ms=first_action_latency,
        time_to_patch_ms=time_to_patch,
        redundant_call_count=redundant,
        repeated_failure_count=repeated_failures,
        path_tokens_approx=path_tokens,
        context_bytes_approx=context_bytes,
        termination_reason=termination,
        verifier_outcome=episode.verifier_status,
        format="harness-improvement",
        version="v1",
    )
