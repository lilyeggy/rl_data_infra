"""Capture primitives for producing canonical append-only execution facts."""

from src.capture.event_writer import (
    AppendResult,
    ArtifactStore,
    EventJsonlReader,
    EventWriter,
    JsonlIssue,
    JsonlReadResult,
)
from src.capture.harness_http import (
    HARNESS_EVENT_INGRESS_VERSION,
    HarnessEventIngress,
    HarnessTraceClient,
)
from src.capture.model_proxy import (
    ModelBackendResponse,
    ModelCallEvidence,
    ModelEndpointKind,
    ModelEvidenceCapability,
    ModelEvidenceJsonlWriter,
    ModelProxyRequest,
    capture_model_call,
)
from src.capture.model_proxy_http import ModelProxyHttpServer, ModelProxyService
from src.capture.pi_adapter import (
    PI_ADAPTER_VERSION,
    PiAdapterResult,
    PiJsonAdapter,
    PiOutcomeDeclaration,
    PiRunConfig,
    dump_pi_ndjson,
    read_pi_ndjson,
)
from src.capture.pi_runner import PiProcessCapture, run_pi_process
from src.capture.recorder import (
    EnvironmentCapture,
    HarnessHook,
    TraceRecorder,
    redact_secrets,
)
from src.capture.workspace_evidence import (
    WORKSPACE_EVIDENCE_VERSION,
    WorkspaceFile,
    WorkspaceSnapshot,
    snapshot_workspace,
    store_workspace_change_evidence,
)

__all__ = [
    "AppendResult",
    "ArtifactStore",
    "EventJsonlReader",
    "EventWriter",
    "JsonlIssue",
    "JsonlReadResult",
    "HARNESS_EVENT_INGRESS_VERSION",
    "HarnessEventIngress",
    "HarnessTraceClient",
    "ModelBackendResponse",
    "ModelCallEvidence",
    "ModelEndpointKind",
    "ModelEvidenceCapability",
    "ModelEvidenceJsonlWriter",
    "ModelProxyHttpServer",
    "ModelProxyRequest",
    "ModelProxyService",
    "HarnessHook",
    "PI_ADAPTER_VERSION",
    "PiAdapterResult",
    "PiJsonAdapter",
    "PiOutcomeDeclaration",
    "PiProcessCapture",
    "PiRunConfig",
    "capture_model_call",
    "dump_pi_ndjson",
    "EnvironmentCapture",
    "TraceRecorder",
    "WORKSPACE_EVIDENCE_VERSION",
    "WorkspaceFile",
    "WorkspaceSnapshot",
    "redact_secrets",
    "read_pi_ndjson",
    "run_pi_process",
    "snapshot_workspace",
    "store_workspace_change_evidence",
]
