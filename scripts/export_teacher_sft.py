#!/usr/bin/env python3
"""Export verifier-certified Teacher episodes as immutable SFT JSONL.

The exporter recomputes SFT eligibility from the canonical Episode and rejects
any persisted decision mismatch.  It never treats a directory name, model
claim, or boolean in arbitrary JSON as training authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.certification import ConsumerProfile, ConsumerVerdict, EligibilityDecision, certify_for
from src.contracts._json import canonical_json_bytes, sha256_json, thaw_json
from src.contracts.agent_episode import AgentEpisode
from src.contracts.dataset import DatasetPurpose, DatasetRole, DatasetSplit
from src.contracts.run_manifest import ExecutionRunManifest
from src.contracts.trace_event import EventType
from src.learning import CertifiedArtifact, classify_sft_example, compile_dataset


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _content_blocks(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [thaw_json(item) for item in value if isinstance(item, Mapping)]


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    parts = [
        str(item.get("text"))
        for item in _content_blocks(value)
        if item.get("type") == "text" and isinstance(item.get("text"), str)
    ]
    return "\n".join(part for part in parts if part).strip()


def _message(value: Mapping[str, Any]) -> dict[str, Any] | None:
    role = value.get("role")
    if role == "toolResult":
        return {
            "role": "tool",
            "tool_call_id": value.get("tool_call_id"),
            "name": value.get("tool_name"),
            "content": _text(value.get("content"))
            or json.dumps(thaw_json(value.get("content")), ensure_ascii=False),
        }
    if role not in {"user", "assistant", "system"}:
        return None
    blocks = _content_blocks(value.get("content"))
    normalized: dict[str, Any] = {"role": role, "content": _text(value.get("content"))}
    calls = []
    raw_calls = value.get("tool_calls")
    if isinstance(raw_calls, Sequence) and not isinstance(raw_calls, (str, bytes)):
        for call in raw_calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            if not isinstance(function, Mapping):
                continue
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments}
            calls.append({
                "id": call.get("id"), "type": "function",
                "function": {"name": function.get("name"), "arguments": arguments},
            })
    for block in blocks:
        if block.get("type") != "toolCall":
            continue
        calls.append(
            {
                "id": block.get("id"),
                "type": "function",
                "function": {
                    "name": block.get("name"),
                    "arguments": json.dumps(
                        block.get("arguments", {}), ensure_ascii=False, sort_keys=True
                    ),
                },
            }
        )
    if calls:
        normalized["tool_calls"] = calls
    return normalized


def _conversation(episode: AgentEpisode) -> tuple[list[dict[str, Any]], str]:
    requests = [event for event in episode.events if event.event_type is EventType.MODEL_REQUEST]
    responses = [event for event in episode.events if event.event_type is EventType.MODEL_RESPONSE]
    if not requests or not responses:
        return [], ""
    history = requests[-1].attributes.get("messages", ())
    messages = []
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        for item in history:
            if isinstance(item, Mapping):
                normalized = _message(item)
                if normalized is not None:
                    messages.append(normalized)
    final_native = {
        "role": "assistant",
        "content": responses[-1].attributes.get("content", ()),
    }
    final = _message(final_native)
    final_answer = _text(final_native["content"])
    if final is not None:
        messages.append(final)
    return messages, final_answer


def _steps(episode: AgentEpisode) -> list[dict[str, Any]]:
    calls: dict[str, dict[str, Any]] = {}
    ordered: list[str] = []
    for event in episode.events:
        call_id = event.attributes.get("pi_tool_call_id")
        if not isinstance(call_id, str):
            continue
        if event.event_type is EventType.TOOL_CALL:
            calls[call_id] = {
                "tool_call_id": call_id,
                "tool_name": event.attributes.get("tool_name"),
                "content": thaw_json(event.attributes.get("arguments")),
                "result": None,
            }
            ordered.append(call_id)
        elif event.event_type is EventType.TOOL_RESULT and call_id in calls:
            calls[call_id]["result"] = thaw_json(event.attributes.get("result"))
            calls[call_id]["result_status"] = event.status.value
    return [calls[item] for item in ordered]


def _example(
    *, episode: AgentEpisode, decision: EligibilityDecision, bundle_checksum: str,
    allow_verifier_finalization: bool = False,
) -> dict[str, Any]:
    messages, final_answer = _conversation(episode)
    final_answer_source = "model"
    if (allow_verifier_finalization and not final_answer.strip()
            and episode.outcome.verifier_status.value == "PASSED"):
        # A coding Agent can terminate immediately after a successful test
        # tool call. This is an explicit metadata attestation, not fabricated
        # model prose or an additional trainable action.
        final_answer = "[verifier] benchmark tests passed"
        final_answer_source = "verifier-attestation"
    return {
        "schema_version": "teacher-sft-example/v1",
        "episode_id": episode.episode_id,
        "episode_checksum": episode.checksum,
        "task_id": episode.task_id,
        "model": {
            "provider": episode.model_manifest.provider,
            "model_id": episode.model_manifest.model_id,
            "revision": episode.model_manifest.revision,
        },
        "harness": {
            "name": episode.harness_manifest.name,
            "version": episode.harness_manifest.version,
        },
        "messages": messages,
        "steps": _steps(episode),
        "final_answer": final_answer,
        "final_answer_source": final_answer_source,
        "certification_verdict": decision.verdict.value,
        "certification_checksum": decision.checksum,
        "execution_bundle_checksum": bundle_checksum,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--split", choices=[item.value for item in DatasetSplit], default="TRAIN")
    parser.add_argument(
        "--allow-verifier-finalization", action="store_true",
        help="allow successful tool-only coding episodes with verifier-attested final metadata",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    examples: list[dict[str, Any]] = []
    certified: list[CertifiedArtifact] = []
    for run in args.runs:
        episode = AgentEpisode.from_dict(json.loads((run / "finalized/episode.json").read_text()))
        manifest = ExecutionRunManifest.from_dict(json.loads((run / "execution-run-manifest.json").read_text()))
        persisted = EligibilityDecision.from_dict(
            json.loads((run / "finalized/sft-eligibility.json").read_text())
        )
        decision = certify_for(episode, ConsumerProfile.SFT)
        if decision.checksum != persisted.checksum:
            raise ValueError(f"persisted SFT decision mismatch for {episode.episode_id}")
        if decision.verdict is not ConsumerVerdict.ELIGIBLE:
            raise ValueError(f"episode {episode.episode_id} is not SFT ELIGIBLE")
        bundle = json.loads((run / "finalized/execution-bundle.json").read_text())
        bundle_checksum = sha256_json(bundle)
        example = _example(
            episode=episode, decision=decision, bundle_checksum=bundle_checksum,
            allow_verifier_finalization=args.allow_verifier_finalization,
        )
        trainable, reasons = classify_sft_example(example)
        if not trainable:
            raise ValueError(
                f"episode {episode.episode_id} failed SFT serialization gate: {', '.join(reasons)}"
            )
        artifact_checksum = sha256_json(example)
        examples.append(example)
        certified.append(
            CertifiedArtifact(
                identity=manifest.identity,
                decision=decision,
                artifact_checksum=artifact_checksum,
                split=DatasetSplit(args.split),
                role=DatasetRole.EXAMPLE,
            )
        )

    examples.sort(key=lambda item: item["episode_id"])
    jsonl = b"".join(canonical_json_bytes(item) + b"\n" for item in examples)
    (args.output / "examples.jsonl").write_bytes(jsonl)
    dataset = compile_dataset(
        dataset_id=args.dataset_id,
        revision=args.revision,
        purpose=DatasetPurpose.SFT,
        selection_policy_version="teacher-sft-selection/v1",
        artifacts=tuple(certified),
    )
    _write_json(args.output / "dataset-manifest.json", dataset.to_dict())
    _write_json(
        args.output / "export-summary.json",
        {
            "dataset_id": args.dataset_id,
            "revision": args.revision,
            "split": args.split,
            "example_count": len(examples),
            "examples_sha256": hashlib.sha256(jsonl).hexdigest(),
            "examples_semantic_checksum": sha256_json(examples),
            "dataset_manifest_checksum": dataset.checksum,
            "source_episode_checksums": [item["episode_checksum"] for item in examples],
        },
    )
    print((args.output / "export-summary.json").read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
