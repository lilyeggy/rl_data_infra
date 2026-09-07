"""HISTORICAL EXAMPLE: synthetic multi-agent causal/training evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.assembly.episode_assembler import EpisodeAssembler, EpisodeContext
from src.contracts._json import sha256_json
from src.contracts.agent_episode import (
    CaptureCapability,
    EpisodeVerifierStatus,
    ExecutionValidity,
    TaskStatus,
)
from src.contracts.manifests import (
    EnvironmentManifest,
    EvaluatorManifest,
    HarnessManifest,
    ModelManifest,
)
from src.contracts.trace_event import EventComponent, EventStatus, EventType, TraceEvent
from src.multi_agent.graph import build_multi_agent_execution_graph
from src.multi_agent.training import build_multi_agent_training_views


EPISODE_ID = "episode-v22-synthetic-multi-agent"
RUN_ID = "run-v22-synthetic-multi-agent"
TRACE_ID = "trace-v22-synthetic-multi-agent"


def _event(
    *,
    event_id: str,
    sequence: int,
    event_type: EventType,
    component: EventComponent,
    status: EventStatus,
    span_id: str,
    parent_span_id: str | None,
    attributes: dict[str, Any],
) -> TraceEvent:
    return TraceEvent(
        event_id=event_id,
        run_id=RUN_ID,
        episode_id=EPISODE_ID,
        trace_id=TRACE_ID,
        span_id=span_id,
        parent_span_id=parent_span_id,
        sequence=sequence,
        timestamp=f"2026-08-18T01:00:{sequence:02d}Z",
        event_type=event_type,
        component=component,
        status=status,
        attempt=1,
        attributes=attributes,
        artifact_refs=(),
    )


def build_synthetic_multi_agent_episode():
    """Build an explicit, non-production fixture with two Agent identities."""

    orchestrator = {
        "agent_id": "agent-orchestrator",
        "parent_agent_id": None,
        "agent_role": "ORCHESTRATOR",
        "policy_id": "policy-orchestrator-v1",
        "policy_revision": "r1",
        "branch_id": "branch-main",
        "parallel_group_id": "parallel-group-1",
    }
    researcher = {
        "agent_id": "agent-researcher",
        "parent_agent_id": "agent-orchestrator",
        "agent_role": "RESEARCHER",
        "policy_id": "policy-researcher-v1",
        "policy_revision": "r1",
        "branch_id": "branch-research",
        "parallel_group_id": "parallel-group-1",
    }
    events = (
        _event(
            event_id="evt-v22-orchestrator-request",
            sequence=0,
            event_type=EventType.MODEL_REQUEST,
            component=EventComponent.MODEL,
            status=EventStatus.STARTED,
            span_id="span-v22-orchestrator-model",
            parent_span_id=None,
            attributes={**orchestrator, "parallel_group_id": "parallel-group-1"},
        ),
        _event(
            event_id="evt-v22-orchestrator-response",
            sequence=1,
            event_type=EventType.MODEL_RESPONSE,
            component=EventComponent.MODEL_BACKEND,
            status=EventStatus.SUCCEEDED,
            span_id="span-v22-orchestrator-model",
            parent_span_id=None,
            attributes={**orchestrator, "parallel_group_id": "parallel-group-1"},
        ),
        _event(
            event_id="evt-v22-researcher-request",
            sequence=2,
            event_type=EventType.MODEL_REQUEST,
            component=EventComponent.MODEL,
            status=EventStatus.STARTED,
            span_id="span-v22-researcher-model",
            parent_span_id="span-v22-orchestrator-model",
            attributes={**researcher, "parallel_group_id": "parallel-group-1"},
        ),
        _event(
            event_id="evt-v22-researcher-response",
            sequence=3,
            event_type=EventType.MODEL_RESPONSE,
            component=EventComponent.MODEL_BACKEND,
            status=EventStatus.SUCCEEDED,
            span_id="span-v22-researcher-model",
            parent_span_id=None,
            attributes={**researcher, "parallel_group_id": "parallel-group-1"},
        ),
        _event(
            event_id="evt-v22-tool-call",
            sequence=4,
            event_type=EventType.TOOL_CALL,
            component=EventComponent.TOOL,
            status=EventStatus.STARTED,
            span_id="span-v22-tool",
            parent_span_id="span-v22-researcher-model",
            attributes={**researcher, "tool_name": "read", "arguments": {"path": "data.json"}},
        ),
        _event(
            event_id="evt-v22-tool-result",
            sequence=5,
            event_type=EventType.TOOL_RESULT,
            component=EventComponent.TOOL,
            status=EventStatus.SUCCEEDED,
            span_id="span-v22-tool",
            parent_span_id="span-v22-researcher-model",
            attributes={**researcher, "tool_name": "read", "result": {"content": []}},
        ),
        _event(
            event_id="evt-v22-message-route",
            sequence=6,
            event_type=EventType.HARNESS_DECISION,
            component=EventComponent.HARNESS,
            status=EventStatus.SUCCEEDED,
            span_id="span-v22-message",
            parent_span_id="span-v22-orchestrator-model",
            attributes={
                **orchestrator,
                "decision": "DELEGATE_RESULT",
                "message_id": "message-v22-1",
                "message_to_agent_id": "agent-researcher",
            },
        ),
        _event(
            event_id="evt-v22-verification-start",
            sequence=7,
            event_type=EventType.VERIFICATION_STARTED,
            component=EventComponent.EVALUATOR,
            status=EventStatus.STARTED,
            span_id="span-v22-verifier",
            parent_span_id=None,
            attributes={
                **orchestrator,
                "reward_owner_agent_id": "agent-orchestrator",
                "reward_scope": "episode",
            },
        ),
        _event(
            event_id="evt-v22-verification-finish",
            sequence=8,
            event_type=EventType.VERIFICATION_FINISHED,
            component=EventComponent.EVALUATOR,
            status=EventStatus.SUCCEEDED,
            span_id="span-v22-verifier",
            parent_span_id=None,
            attributes={
                **orchestrator,
                "passed": True,
                "score": 1.0,
                "reward_owner_agent_id": "agent-orchestrator",
                "reward_scope": "episode",
            },
        ),
        _event(
            event_id="evt-v22-finished",
            sequence=9,
            event_type=EventType.EPISODE_FINISHED,
            component=EventComponent.HARNESS,
            status=EventStatus.SUCCEEDED,
            span_id="span-v22-episode",
            parent_span_id=None,
            attributes={
                **orchestrator,
                "task_status": TaskStatus.SUCCESS.value,
                "execution_validity": ExecutionValidity.VALID.value,
                "verifier_status": EpisodeVerifierStatus.PASSED.value,
                "score": 1.0,
                "termination_reason": "MULTI_AGENT_VERIFIED",
                "evidence_event_ids": ["evt-v22-verification-finish"],
                "reward_owner_agent_id": "agent-orchestrator",
                "reward_scope": "episode",
            },
        ),
    )
    digest = sha256_json({"fixture": "synthetic-multi-agent-v1"})
    context = EpisodeContext(
        task_id="v22-synthetic/task-1",
        attempt=1,
        harness_manifest=HarnessManifest(
            name="synthetic-multi-agent-harness",
            version="v2.2-fixture",
            revision="synthetic-multi-agent-v1",
            config_digest=digest,
            policy_flags={"explicit_agent_metadata": True},
            hook_version="multi-agent-hook/v1",
        ),
        model_manifest=ModelManifest(
            provider="synthetic",
            model_id="multi-agent-fixture-model",
            revision="r1",
            sampling_config={"temperature": 0},
            tokenizer_revision=None,
        ),
        environment_manifest=EnvironmentManifest(
            runtime_type="synthetic-process",
            revision="multi-agent-fixture-env/r1",
            image=None,
            resource_limits={"timeout_seconds": 30},
            network_policy="disabled",
            task_snapshot="multi-agent-fixture/r1",
        ),
        evaluator_manifest=EvaluatorManifest(
            name="synthetic-multi-agent-verifier",
            revision="r1",
            config_digest=digest,
        ),
        experiment_manifest_ref="experiment-v22-synthetic",
        capabilities=frozenset(
            {
                CaptureCapability.MODEL_IO,
                CaptureCapability.TOOL_IO,
                CaptureCapability.VERIFIER_EVIDENCE,
                CaptureCapability.HARNESS_DECISIONS,
            }
        ),
    )
    result = EpisodeAssembler().assemble(events, contexts={EPISODE_ID: context})
    if result.warnings or result.quarantined_events or len(result.episodes) != 1:
        raise RuntimeError(
            f"synthetic multi-agent fixture failed assembly: {result.warnings}, "
            f"quarantined={len(result.quarantined_events)}"
        )
    return result.episodes[0]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def generate_v22_multi_agent_evidence(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    episode = build_synthetic_multi_agent_episode()
    graph = build_multi_agent_execution_graph(episode)
    views = build_multi_agent_training_views(
        episode,
        graph,
        target_policy_by_agent={
            "agent-orchestrator": "policy-orchestrator-v1",
            "agent-researcher": "policy-researcher-v1",
        },
    )
    _write_json(output / "episode.json", episode.to_dict())
    _write_json(output / "multi-agent-graph.json", graph.to_dict())
    _write_json(output / "training-views.json", [view.to_dict() for view in views])
    summary = {
        "release": "v2.2",
        "scope": "synthetic explicit multi-agent causal and training semantics",
        "synthetic_fixture": True,
        "episode_id": episode.episode_id,
        "graph_status": graph.graph_status.value,
        "agent_count": len(graph.agents),
        "relation_count": len(graph.relations),
        "field_status": {key: value.value for key, value in graph.field_status.items()},
        "sft_candidates": sum(view.sft_candidate for view in views),
        "on_policy_rl_candidates": sum(view.on_policy_rl_candidate for view in views),
        "claim_boundary": (
            "explicit synthetic metadata validates the V2.2 contract; it does not "
            "establish general multi-agent benchmark or training results"
        ),
        "evidence_checksums": {
            "episode": episode.checksum,
            "graph": graph.checksum,
            "training_views": sha256_json([view.to_dict() for view in views]),
        },
    }
    _write_json(output / "summary.json", summary)
    return summary
