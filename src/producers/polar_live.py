"""Live Polar batch producer over the documented rollout HTTP service."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from src.integrations.polar.adapter import (
    IDENTITY_METADATA_KEY,
    PolarBatchContext,
    adapt_polar_task_result,
)
from src.integrations.polar.client import PolarHttpClient, PolarTaskRequest
from src.producers.base import ProducerArtifact, ProducerCapability


class PolarLiveBatchProducer:
    """Submit one Polar fan-out task and collect strongly bound session artifacts.

    This is intentionally a batch producer rather than pretending to implement
    ``RolloutProducer.run``: Polar allocates session ids only after submission.
    """

    producer_id = "polar"
    capabilities = frozenset(
        {
            ProducerCapability.ASYNC_SUBMISSION,
            ProducerCapability.TOKEN_IDS,
            ProducerCapability.ACTION_MASK,
            ProducerCapability.BEHAVIOR_LOGPROBS,
            ProducerCapability.POLICY_VERSION,
            ProducerCapability.VERIFIER_EVIDENCE,
        }
    )

    def __init__(self, client: PolarHttpClient, *, producer_version: str) -> None:
        if not isinstance(client, PolarHttpClient):
            raise TypeError("client must be a PolarHttpClient")
        if not isinstance(producer_version, str) or not producer_version.strip():
            raise ValueError("producer_version must be a non-empty string")
        self._client = client
        self.producer_version = producer_version

    def submit(
        self,
        task: PolarTaskRequest,
        *,
        context: PolarBatchContext,
    ) -> Mapping[str, Any]:
        self._validate_pair(task, context)
        if IDENTITY_METADATA_KEY in task.metadata:
            raise ValueError(f"metadata key {IDENTITY_METADATA_KEY!r} is reserved")
        metadata = {**dict(task.metadata), IDENTITY_METADATA_KEY: context.identity_metadata()}
        return self._client.submit_task(replace(task, metadata=metadata))

    def collect(self, *, context: PolarBatchContext) -> tuple[ProducerArtifact, ...]:
        if context.producer_version != self.producer_version:
            raise ValueError("context producer_version does not match producer")
        raw = self._client.get_task(context.polar_task_id)
        return adapt_polar_task_result(raw, context=context)

    def _validate_pair(self, task: PolarTaskRequest, context: PolarBatchContext) -> None:
        if not isinstance(task, PolarTaskRequest):
            raise TypeError("task must be a PolarTaskRequest")
        if not isinstance(context, PolarBatchContext):
            raise TypeError("context must be a PolarBatchContext")
        if task.task_id != context.polar_task_id:
            raise ValueError("task_id does not match PolarBatchContext")
        if task.num_samples != context.expected_samples:
            raise ValueError("num_samples does not match PolarBatchContext")
        if context.producer_version != self.producer_version:
            raise ValueError("context producer_version does not match producer")
