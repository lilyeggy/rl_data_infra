"""Direct Pi producer for real harness capture without RL-only claims."""

from __future__ import annotations

import subprocess
from collections.abc import Callable

from src.capture.pi_adapter import PiRunConfig
from src.capture.pi_runner import PiProcessCapture, run_pi_process
from src.errors import ContractValidationError
from src.producers.base import (
    ProducerArtifact,
    ProducerCapability,
    ProducerExecutionStatus,
    ProducerRequest,
)


class PiDirectProducer:
    """Execute Pi directly and return its raw protocol artifact.

    This producer intentionally does not claim token IDs, behavior logprobs,
    action masks, policy sync, or verifier evidence. Those capabilities require
    a gateway such as Polar plus an external verifier.
    """

    producer_id = "pi-direct"
    producer_version = "pi-direct/v1"
    capabilities = frozenset({ProducerCapability.RAW_HARNESS_TRACE})

    def __init__(
        self,
        config: PiRunConfig,
        *,
        runner: Callable[..., PiProcessCapture] = run_pi_process,
    ) -> None:
        if not isinstance(config, PiRunConfig):
            raise ContractValidationError("config must be a PiRunConfig")
        self._config = config
        self._runner = runner

    def run(self, request: ProducerRequest) -> ProducerArtifact:
        if not isinstance(request, ProducerRequest):
            raise TypeError("PiDirectProducer.run requires a ProducerRequest")
        if request.identity.producer_id != self.producer_id:
            raise ContractValidationError(
                "request identity producer_id does not match PiDirectProducer"
            )
        if request.identity.producer_version != self.producer_version:
            raise ContractValidationError(
                "request identity producer_version does not match PiDirectProducer"
            )
        try:
            capture = self._runner(
                config=self._config,
                prompt=request.instruction,
                cwd=request.workspace,
                timeout_seconds=request.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return ProducerArtifact(
                identity=request.identity,
                status=ProducerExecutionStatus.TIMEOUT,
                capabilities=self.capabilities,
                payload={"timeout_seconds": request.timeout_seconds},
                issues=(f"Pi process timed out: {exc}",),
            )
        except OSError as exc:
            return ProducerArtifact(
                identity=request.identity,
                status=ProducerExecutionStatus.INFRA_INVALID,
                capabilities=self.capabilities,
                payload={},
                issues=(f"Pi process could not start: {exc}",),
            )

        if capture.backend_error_messages:
            status = ProducerExecutionStatus.INFRA_INVALID
        elif capture.returncode != 0 or not capture.protocol_settled:
            status = ProducerExecutionStatus.FAILED
        else:
            status = ProducerExecutionStatus.COMPLETED
        return ProducerArtifact(
            identity=request.identity,
            status=status,
            capabilities=self.capabilities,
            payload={
                "command": list(capture.command),
                "cwd": capture.cwd,
                "returncode": capture.returncode,
                "stdout": capture.stdout,
                "stderr": capture.stderr,
                "record_count": capture.record_count,
                "final_stop_reason": capture.final_stop_reason,
                "backend_error_messages": list(capture.backend_error_messages),
            },
            issues=tuple(issue.message for issue in capture.issues),
        )
