"""Small canonical fixtures shared by core contract tests."""

from __future__ import annotations

from typing import Any

from src.contracts import (
    ComponentStatus,
    RolloutRecord,
    RolloutStatus,
    VerifierStatus,
)


def make_record(**overrides: Any) -> RolloutRecord:
    values: dict[str, Any] = {
        "trajectory_id": "trajectory-001",
        "task_id": "task-calculator-001",
        "group_id": "group-001",
        "policy_version": "policy-v0",
        "source_type": "fixture",
        "source_record_id": "fixture-record-001",
        "token_ids": (101, 102, 103, 104),
        "prompt_token_count": 1,
        "loss_mask": (0, 1, 1, 1),
        "old_logprobs": (-0.3, -0.2, -0.1),
        "reward": 1.0,
        "rollout_status": RolloutStatus.COMPLETED,
        "termination_reason": "agent_finished",
        "runtime_status": ComponentStatus.SUCCEEDED,
        "harness_status": ComponentStatus.SUCCEEDED,
        "model_backend_status": ComponentStatus.SUCCEEDED,
        "verifier_status": VerifierStatus.PASSED,
        "verifier_evidence_ref": "fixture://summary.json#/verifier",
        "source_payload_ref": "fixture://summary.json",
        "source_payload_sha256": "a" * 64,
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "model_revision": "b" * 40,
        "tokenizer_revision": "b" * 40,
        "started_at": "2026-08-10T00:00:00Z",
        "ended_at": "2026-08-10T00:01:00Z",
        "tool_events": ({"type": "tool_result", "name": "calculator"},),
        "opaque_metadata": {"producer": {"session_id": "session-001"}},
    }
    if overrides.get("token_ids", object()) is None and "prompt_token_count" not in overrides:
        values["prompt_token_count"] = None
    values.update(overrides)
    return RolloutRecord(**values)
