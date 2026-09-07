"""Polar rollout-service client and pinned result adapter."""

from src.integrations.polar.adapter import (
    IDENTITY_METADATA_KEY,
    POLAR_ADAPTER_VERSION,
    PolarBatchContext,
    PolarResultNotReady,
    PolarSchemaError,
    adapt_polar_task_result,
)
from src.integrations.polar.client import (
    PolarClientError,
    PolarHttpClient,
    PolarTaskRequest,
)

__all__ = [
    "IDENTITY_METADATA_KEY",
    "POLAR_ADAPTER_VERSION",
    "PolarBatchContext",
    "PolarClientError",
    "PolarHttpClient",
    "PolarResultNotReady",
    "PolarSchemaError",
    "PolarTaskRequest",
    "adapt_polar_task_result",
]
