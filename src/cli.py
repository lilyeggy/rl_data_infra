"""Dependency-free CLI for V1/V2 Agent Infra evidence artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.capture.pi_adapter import dump_pi_ndjson, read_pi_ndjson
from src.contracts.agent_episode import AgentEpisode
from src.demo_v1 import generate_v1_demo
from src.demo_v2 import generate_v2_demo


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


def _demo_v2(args: argparse.Namespace) -> int:
    _print(generate_v2_demo(args.output))
    return 0


def _sanitize_pi(args: argparse.Namespace) -> int:
    source = Path(args.input)
    destination = Path(args.output)
    records, issues = read_pi_ndjson(source.read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(dump_pi_ndjson(records))
    _print(
        {
            "input": str(source),
            "output": str(destination),
            "record_count": len(records),
            "issue_count": len(issues),
        }
    )
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
    demo_v2 = commands.add_parser(
        "demo-v2", help="generate V2 from captured real Pi comparison pairs"
    )
    demo_v2.add_argument("--output", default="artifacts/v2-harness-decision")
    demo_v2.set_defaults(func=_demo_v2)
    sanitize_pi = commands.add_parser(
        "sanitize-pi", help="redact and normalize a Pi NDJSON trace for fixtures"
    )
    sanitize_pi.add_argument("--input", required=True)
    sanitize_pi.add_argument("--output", required=True)
    sanitize_pi.set_defaults(func=_sanitize_pi)
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
