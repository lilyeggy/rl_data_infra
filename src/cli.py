"""Small dependency-free CLI for the V1 evidence artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.contracts.agent_episode import AgentEpisode
from src.demo_v1 import generate_v1_demo


def _read_episodes(path: Path) -> tuple[AgentEpisode, ...]:
    episodes: list[AgentEpisode] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                episodes.append(AgentEpisode.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid episode at {path}:{line_number}: {exc}") from exc
    return tuple(episodes)


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _demo(args: argparse.Namespace) -> int:
    _print(generate_v1_demo(args.output))
    return 0


def _inspect(args: argparse.Namespace) -> int:
    episodes = _read_episodes(Path(args.episodes))
    matches = [episode for episode in episodes if episode.episode_id == args.episode_id]
    if not matches:
        raise SystemExit(f"episode_id {args.episode_id!r} not found")
    episode = matches[0]
    _print(
        {
            "episode_id": episode.episode_id,
            "task_id": episode.task_id,
            "harness": episode.harness_manifest.to_dict(),
            "model": episode.model_manifest.to_dict(),
            "outcome": episode.outcome.to_dict(),
            "integrity": episode.integrity.to_dict(),
            "capabilities": sorted(item.value for item in episode.capabilities),
            "timeline": [
                {
                    "sequence": event.sequence,
                    "timestamp": event.timestamp,
                    "event_id": event.event_id,
                    "span_id": event.span_id,
                    "parent_span_id": event.parent_span_id,
                    "type": event.event_type.value,
                    "component": event.component.value,
                    "status": event.status.value,
                    "attempt": event.attempt,
                    "artifact_refs": list(event.artifact_refs),
                }
                for event in episode.events
            ],
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-infra")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo-v1", help="generate the deterministic V1 artifact")
    demo.add_argument("--output", default="artifacts/v1-observability")
    demo.set_defaults(func=_demo)
    inspect = commands.add_parser("inspect", help="inspect one canonical Episode timeline")
    inspect.add_argument("--episodes", required=True)
    inspect.add_argument("--episode-id", required=True)
    inspect.set_defaults(func=_inspect)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
