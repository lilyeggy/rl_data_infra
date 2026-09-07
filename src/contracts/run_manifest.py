"""Immutable manifest frozen before one Local Launcher execution starts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.contracts._json import sha256_json, validate_sha256
from src.contracts._validation import required_text, strict_fields, utc_instant
from src.contracts.agent_episode import CaptureCapability
from src.contracts.execution_identity import ExecutionIdentity
from src.contracts.manifests import (
    EnvironmentManifest,
    EvaluatorManifest,
    HarnessManifest,
    ModelManifest,
)
from src.errors import ContractValidationError

EXECUTION_RUN_MANIFEST_VERSION = "execution-run-manifest/v1"


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionRunManifest:
    identity: ExecutionIdentity
    harness_manifest: HarnessManifest
    model_manifest: ModelManifest
    environment_manifest: EnvironmentManifest
    evaluator_manifest: EvaluatorManifest
    experiment_manifest_ref: str
    capture_capabilities: frozenset[CaptureCapability]
    launcher_plan_checksum: str
    created_at: str
    schema_version: str = EXECUTION_RUN_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ExecutionIdentity):
            raise ContractValidationError("identity must be ExecutionIdentity")
        for name, expected_type in (
            ("harness_manifest", HarnessManifest),
            ("model_manifest", ModelManifest),
            ("environment_manifest", EnvironmentManifest),
            ("evaluator_manifest", EvaluatorManifest),
        ):
            if not isinstance(getattr(self, name), expected_type):
                raise ContractValidationError(f"{name} has invalid type")
        required_text(self.experiment_manifest_ref, "experiment_manifest_ref")
        capabilities = frozenset(self.capture_capabilities)
        if any(not isinstance(item, CaptureCapability) for item in capabilities):
            raise ContractValidationError("capture_capabilities contains unknown value")
        object.__setattr__(self, "capture_capabilities", capabilities)
        validate_sha256(self.launcher_plan_checksum, "launcher_plan_checksum")
        utc_instant(self.created_at, "created_at")
        if self.schema_version != EXECUTION_RUN_MANIFEST_VERSION:
            raise ContractValidationError("unsupported ExecutionRunManifest schema_version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": self.identity.to_dict(),
            "harness_manifest": self.harness_manifest.to_dict(),
            "model_manifest": self.model_manifest.to_dict(),
            "environment_manifest": self.environment_manifest.to_dict(),
            "evaluator_manifest": self.evaluator_manifest.to_dict(),
            "experiment_manifest_ref": self.experiment_manifest_ref,
            "capture_capabilities": sorted(
                item.value for item in self.capture_capabilities
            ),
            "launcher_plan_checksum": self.launcher_plan_checksum,
            "created_at": self.created_at,
        }

    @property
    def checksum(self) -> str:
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionRunManifest":
        payload = strict_fields(
            value,
            {
                "schema_version",
                "identity",
                "harness_manifest",
                "model_manifest",
                "environment_manifest",
                "evaluator_manifest",
                "experiment_manifest_ref",
                "capture_capabilities",
                "launcher_plan_checksum",
                "created_at",
            },
            "ExecutionRunManifest",
        )
        try:
            capabilities = frozenset(
                CaptureCapability(item) for item in payload["capture_capabilities"]
            )
        except (TypeError, ValueError) as exc:
            raise ContractValidationError("invalid capture capability") from exc
        return cls(
            schema_version=payload["schema_version"],
            identity=ExecutionIdentity.from_dict(payload["identity"]),
            harness_manifest=HarnessManifest.from_dict(payload["harness_manifest"]),
            model_manifest=ModelManifest.from_dict(payload["model_manifest"]),
            environment_manifest=EnvironmentManifest.from_dict(
                payload["environment_manifest"]
            ),
            evaluator_manifest=EvaluatorManifest.from_dict(
                payload["evaluator_manifest"]
            ),
            experiment_manifest_ref=payload["experiment_manifest_ref"],
            capture_capabilities=capabilities,
            launcher_plan_checksum=payload["launcher_plan_checksum"],
            created_at=payload["created_at"],
        )
