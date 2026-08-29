"""Small deterministic fixtures for execution-contract tests."""

from __future__ import annotations

from typing import Any

from src.contracts.trace_event import (
    EventComponent,
    EventStatus,
    EventType,
    TraceEvent,
)
from src.assembly.episode_assembler import EpisodeContext
from src.contracts._json import sha256_json
from src.contracts.agent_episode import CaptureCapability
from src.contracts.manifests import (
    EnvironmentManifest,
    EvaluatorManifest,
    HarnessManifest,
    ModelManifest,
)


def make_trace_event(**overrides: Any) -> TraceEvent:
    values: dict[str, Any] = {
        "event_id": "evt-001",
        "run_id": "run-001",
        "episode_id": "episode-001",
        "trace_id": "trace-001",
        "span_id": "span-model-001",
        "parent_span_id": "span-root-001",
        "sequence": 1,
        "timestamp": "2026-08-14T00:00:01Z",
        "event_type": EventType.MODEL_REQUEST,
        "component": EventComponent.MODEL,
        "status": EventStatus.STARTED,
        "attempt": 1,
        "attributes": {
            "model": "fixture-model",
            "producer": {"name": "fixture-harness", "version": "v1"},
        },
        "artifact_refs": ("artifact-request-001",),
    }
    values.update(overrides)
    return TraceEvent(**values)


def make_episode_context(**overrides: Any) -> EpisodeContext:
    digest = sha256_json({})
    values: dict[str, Any] = {
        "task_id": "task-001",
        "attempt": 1,
        "harness_manifest": HarnessManifest(
            name="fixture-harness",
            version="v1",
            revision="fixture-revision",
            config_digest=digest,
            policy_flags={"structured_tool_errors": False},
            hook_version="fixture-hook/v1",
        ),
        "model_manifest": ModelManifest(
            provider="fixture-provider",
            model_id="fixture-model",
            revision="fixture-model-r1",
            sampling_config={"temperature": 0, "seed": 7},
            tokenizer_revision="fixture-tokenizer-r1",
        ),
        "environment_manifest": EnvironmentManifest(
            runtime_type="process",
            revision="fixture-env-r1",
            image=None,
            resource_limits={"timeout_seconds": 30},
            network_policy="disabled",
            task_snapshot="fixture-task-r1",
        ),
        "evaluator_manifest": EvaluatorManifest(
            name="fixture-verifier",
            revision="fixture-verifier-r1",
            config_digest=digest,
        ),
        "experiment_manifest_ref": "experiment-fixture-001",
        "capabilities": frozenset(
            {
                CaptureCapability.MODEL_IO,
                CaptureCapability.MODEL_TOKEN_USAGE,
                CaptureCapability.TOOL_IO,
                CaptureCapability.VERIFIER_EVIDENCE,
                CaptureCapability.HARNESS_DECISIONS,
                CaptureCapability.TERMINATION_DECISIONS,
            }
        ),
    }
    values.update(overrides)
    return EpisodeContext(**values)


def make_complete_event_stream(**identity_overrides: Any) -> tuple[TraceEvent, ...]:
    identity = {
        "run_id": identity_overrides.get("run_id", "run-001"),
        "episode_id": identity_overrides.get("episode_id", "episode-001"),
        "trace_id": identity_overrides.get("trace_id", "trace-001"),
    }
    values = (
        {
            "event_id": "evt-request",
            "sequence": 0,
            "timestamp": "2026-08-14T00:00:00Z",
            "span_id": "span-model",
            "parent_span_id": None,
            "event_type": EventType.MODEL_REQUEST,
            "component": EventComponent.MODEL,
            "status": EventStatus.STARTED,
            "attributes": {"input_tokens": 10},
            "artifact_refs": (),
        },
        {
            "event_id": "evt-response",
            "sequence": 1,
            "timestamp": "2026-08-14T00:00:01Z",
            "span_id": "span-model",
            "parent_span_id": None,
            "event_type": EventType.MODEL_RESPONSE,
            "component": EventComponent.MODEL_BACKEND,
            "status": EventStatus.SUCCEEDED,
            "attributes": {
                "usage": {"input_tokens": 10, "output_tokens": 4},
                "latency_ms": 100,
            },
            "artifact_refs": (),
        },
        {
            "event_id": "evt-tool-call",
            "sequence": 2,
            "timestamp": "2026-08-14T00:00:02Z",
            "span_id": "span-tool",
            "parent_span_id": "span-model",
            "event_type": EventType.TOOL_CALL,
            "component": EventComponent.TOOL,
            "status": EventStatus.STARTED,
            "attributes": {"tool_name": "shell", "arguments": {"cmd": "true"}},
            "artifact_refs": (),
        },
        {
            "event_id": "evt-tool-result",
            "sequence": 3,
            "timestamp": "2026-08-14T00:00:03Z",
            "span_id": "span-tool",
            "parent_span_id": "span-model",
            "event_type": EventType.TOOL_RESULT,
            "component": EventComponent.TOOL,
            "status": EventStatus.SUCCEEDED,
            "attributes": {"tool_name": "shell", "latency_ms": 25, "exit_code": 0},
            "artifact_refs": (),
        },
        {
            "event_id": "evt-verification-started",
            "sequence": 4,
            "timestamp": "2026-08-14T00:00:04Z",
            "span_id": "span-verifier",
            "parent_span_id": None,
            "event_type": EventType.VERIFICATION_STARTED,
            "component": EventComponent.EVALUATOR,
            "status": EventStatus.STARTED,
            "attributes": {},
            "artifact_refs": (),
        },
        {
            "event_id": "evt-verification-finished",
            "sequence": 5,
            "timestamp": "2026-08-14T00:00:05Z",
            "span_id": "span-verifier",
            "parent_span_id": None,
            "event_type": EventType.VERIFICATION_FINISHED,
            "component": EventComponent.EVALUATOR,
            "status": EventStatus.SUCCEEDED,
            "attributes": {"passed": True, "latency_ms": 50},
            "artifact_refs": (),
        },
        {
            "event_id": "evt-finished",
            "sequence": 6,
            "timestamp": "2026-08-14T00:00:06Z",
            "span_id": "span-episode",
            "parent_span_id": None,
            "event_type": EventType.EPISODE_FINISHED,
            "component": EventComponent.HARNESS,
            "status": EventStatus.SUCCEEDED,
            "attributes": {
                "task_status": "SUCCESS",
                "execution_validity": "VALID",
                "verifier_status": "PASSED",
                "score": 1.0,
                "termination_reason": "VERIFIER_PASSED",
                "evidence_event_ids": ["evt-verification-finished"],
            },
            "artifact_refs": (),
        },
    )
    return tuple(make_trace_event(**identity, **item) for item in values)
