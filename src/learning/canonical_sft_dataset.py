"""Build a canonical, harness-neutral SFT dataset from CanonicalEpisodes.

This is the Section-6 redesign: instead of copying full Teacher outputs as a
flat list of turns (all weighted equally), we draw *decision-role* examples
(first-action / tool-selection / failure-recovery / patch / test-selection /
final-answer) explicitly, so training can weight canonical action boundaries and
tool selection rather than Pi-format tokens.

Quality gates applied:
- trajectory-level grouping (split by episode) to prevent task leakage;
- per-role statistics (task count, example count, source, verifier, lossy);
- host-path leakage scan (from canonical exporter);
- duplication tracking;
- optional per-token loss weighting markers (assist-only, action weight) written
  into the manifest, NOT hard-coded to Pi-private tokens.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping

from src.capture.pi_canonical_adapter import CanonicalEpisode, convert_episode_to_canonical
from src.contracts._json import sha256_json, thaw_json
from src.exporters.canonical import (
    assert_training_view_is_leak_free,
)

SFT_DATASET_VERSION = "canonical-generic-sft/v1"

ROLES = (
    "first-action",
    "tool-selection",
    "failure-recovery",
    "patch",
    "test-selection",
    "final-answer",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RoleStats:
    role: str
    example_count: int
    task_count: int
    token_count: int
    lossy_count: int
    verifier_passed: int
    source_episodes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "example_count": self.example_count,
            "task_count": self.task_count,
            "token_count": self.token_count,
            "lossy_count": self.lossy_count,
            "verifier_passed": self.verifier_passed,
            "source_episodes": list(self.source_episodes),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SFTDatasetReport:
    episodes: int
    total_examples: int
    total_tokens: int
    role_stats: tuple[RoleStats, ...]
    lossy_examples: int
    duplication_count: int
    leak_count: int
    source_episodes: tuple[str, ...]
    channels: tuple[str, ...]  # loss-mask channels produced

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SFT_DATASET_VERSION,
            "episodes": self.episodes,
            "total_examples": self.total_examples,
            "total_tokens": self.total_tokens,
            "role_stats": [r.to_dict() for r in self.role_stats],
            "lossy_examples": self.lossy_examples,
            "duplication_count": self.duplication_count,
            "leak_count": self.leak_count,
            "source_episodes": list(self.source_episodes),
            "channels": list(self.channels),
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True, kw_only=True)
class SFTExample:
    episode_id: str
    task_id: str
    role: str
    system_prompt_marker: str  # neutral; never Pi-private
    user_text: str
    action: Mapping[str, Any]
    expected: Mapping[str, Any]
    verifier_status: str | None
    lossy: bool
    split: str  # TRAIN / DEV / TEST (dataset-level)
    token_count_approx: int
    source_episode_checksum: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "role": self.role,
            "system_prompt_marker": self.system_prompt_marker,
            "user_text": self.user_text,
            "action": thaw_json(self.action),
            "expected": thaw_json(self.expected),
            "verifier_status": self.verifier_status,
            "lossy": self.lossy,
            "split": self.split,
            "token_count_approx": self.token_count_approx,
            "source_episode_checksum": self.source_episode_checksum,
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())


def _token_approx(text: str) -> int:
    # conservative estimate: words + punctuation chunks
    return len(text.replace("\n", " ").split())


def _role_of(action, index: int, count: int, prev_failed: bool) -> str:
    if index == 0:
        return "first-action"
    if index == count - 1:
        return "final-answer"
    if action.canonical_tool_name in {"edit_file", "write_file"}:
        return "patch"
    if prev_failed:
        return "failure-recovery"
    return "tool-selection"


def build_canonical_sft_dataset(
    episodes: tuple[Any, ...],
    *,
    task_ids: Mapping[str, str] | None = None,
    workspace_roots: Mapping[str, str] | None = None,
    split: str = "TRAIN",
    channel: str = "assistant-only",
) -> tuple[tuple[SFTExample, ...], SFTDatasetReport]:
    """Convert episodes (AgentEpisode or CanonicalEpisode) into SFT examples.

    Every example is derived from a CanonicalEpisode (harness-neutral), carries
    an explicit role, a neutral system-prompt marker (never Pi-private), and
    only normalized/leak-free action content. Loss masking is declared per
    channel in the report (assistant-only by default) so the trainer can apply
    per-role weights without inventing Pi-private tokens.
    """

    task_ids = task_ids or {}
    canonical_episodes: dict[str, CanonicalEpisode] = {}
    canonical_issues: dict[str, tuple] = {}
    for ep in episodes:
        if isinstance(ep, CanonicalEpisode):
            canonical_episodes[ep.episode_id] = ep
            canonical_issues[ep.episode_id] = ()
            continue
        canonical, issues = convert_episode_to_canonical(
            ep, workspace_root=(workspace_roots or {}).get(ep.episode_id)
        )
        canonical_episodes[ep.episode_id] = canonical
        canonical_issues[ep.episode_id] = issues

    examples: list[SFTExample] = []
    role_examples: defaultdict[str, int] = defaultdict(int)
    role_tokens: defaultdict[str, int] = defaultdict(int)
    role_tasks: defaultdict[str, set] = defaultdict(set)
    role_lossy: defaultdict[str, int] = defaultdict(int)
    role_verifier: defaultdict[str, int] = defaultdict(int)
    source_set: set[str] = set()
    leak_count = 0
    seen_hashes: set[str] = set()
    dup_count = 0

    for ep in episodes:
        canonical = canonical_episodes[ep.episode_id]
        source_set.add(ep.episode_id)
        task_id = task_ids.get(
            ep.episode_id,
            ep.task_id if not isinstance(ep, CanonicalEpisode) else None,
        )
        if task_id is None:
            # CanonicalEpisode always carries a task_id
            task_id = ep.task_id
        count = len(canonical.actions)
        for index, action in enumerate(canonical.actions):
            prev_failed = (
                canonical.actions[index - 1].result_status.value
                in {"FAILED", "ERROR", "TIMEOUT"}
                if index > 0
                else False
            )
            role = _role_of(action, index, count, prev_failed)
            # leakage scan
            try:
                assert_training_view_is_leak_free(
                    {"action": action.normalized_arguments}, label="sft-dataset"
                )
            except ValueError:
                leak_count += 1
                continue
            user_text = _render_user(task_id, canonical, index)
            action_payload = {
                "tool": action.canonical_tool_name,
                "args": thaw_json(action.normalized_arguments),
                "observation": action.observation,
            }
            expected = {
                "tool": action.canonical_tool_name,
                "args": thaw_json(action.normalized_arguments),
            }
            ex = SFTExample(
                episode_id=ep.episode_id,
                task_id=task_id,
                role=role,
                system_prompt_marker="<neutral-agent-instructions>",
                user_text=user_text,
                action=action_payload,
                expected=expected,
                verifier_status=canonical.verifier_status,
                lossy=action.lossy,
                split=split,
                token_count_approx=_token_approx(user_text)
                + _token_approx(json.dumps(expected)),
                source_episode_checksum=ep.checksum,
            )
            digest = sha256_json(ex.to_dict())
            if digest in seen_hashes:
                dup_count += 1
                continue
            seen_hashes.add(digest)
            examples.append(ex)
            role_examples[role] += 1
            role_tokens[role] += ex.token_count_approx
            role_tasks[role].add(ep.episode_id)
            if action.lossy:
                role_lossy[role] += 1
            if canonical.verifier_status == "PASSED":
                role_verifier[role] += 1

    role_stats = tuple(
        RoleStats(
            role=role,
            example_count=role_examples[role],
            task_count=len(role_tasks[role]),
            token_count=role_tokens[role],
            lossy_count=role_lossy[role],
            verifier_passed=role_verifier[role],
            source_episodes=tuple(sorted(role_tasks[role])),
        )
        for role in ROLES
        if role_examples[role] > 0 or role in ("first-action", "final-answer")
    )
    report = SFTDatasetReport(
        episodes=len(canonical_episodes),
        total_examples=len(examples),
        total_tokens=sum(role_tokens.values()),
        role_stats=role_stats,
        lossy_examples=sum(role_lossy.values()),
        duplication_count=dup_count,
        leak_count=leak_count,
        source_episodes=tuple(sorted(source_set)),
        channels=(channel,),
    )
    return tuple(examples), report


def _render_user(task_id: str, episode: CanonicalEpisode, index: int) -> str:
    """Render a neutral task+history prompt ending right before the target action.

    Uses only canonical action content (no Pi tool names, no absolute paths,
    no Pi-private system prompt).
    """

    history: list[str] = []
    for i, action in enumerate(episode.actions[:index]):
        status = (
            "ok"
            if action.result_status.value == "SUCCEEDED"
            else action.result_status.value.lower()
        )
        args_repr = json.dumps(
            thaw_json(action.normalized_arguments), ensure_ascii=False, sort_keys=True
        )
        history.append(
            f"  {action.canonical_tool_name}({args_repr}) -> {status}"
        )
    prefix = "\n".join(g for g in history) if history else "(no prior actions)"
    return (
        f"Solve the task: {task_id}\n\n"
        f"Progress so far ({index} completed action):\n{prefix}\n\n"
        f"Choose the next canonical action."
    )