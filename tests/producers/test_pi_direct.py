from __future__ import annotations

import subprocess
import unittest

from src.capture.pi_adapter import PiRunConfig
from src.capture.pi_runner import PiProcessCapture
from src.contracts.execution_identity import ExecutionIdentity
from src.producers import (
    PiDirectProducer,
    ProducerArtifact,
    ProducerCapability,
    ProducerExecutionStatus,
    ProducerRequest,
    RolloutProducer,
)


def _identity() -> ExecutionIdentity:
    return ExecutionIdentity(
        run_id="run-1",
        task_id="task-1",
        episode_id="episode-1",
        attempt_id=1,
        producer_id="pi-direct",
        producer_version="pi-direct/v1",
    )


def _request() -> ProducerRequest:
    return ProducerRequest(
        identity=_identity(),
        instruction="fix the task",
        workspace="/tmp/workspace",
        timeout_seconds=30,
    )


class PiDirectProducerTest(unittest.TestCase):
    def test_producer_is_protocol_compatible_and_claims_no_rl_fields(self) -> None:
        capture = PiProcessCapture(
            command=("pi",),
            cwd="/tmp/workspace",
            returncode=0,
            stdout='{"type":"session"}\n',
            stderr="",
            record_count=1,
            issues=(),
            final_stop_reason="stop",
            backend_error_messages=(),
        )
        producer = PiDirectProducer(PiRunConfig(), runner=lambda **_: capture)
        self.assertIsInstance(producer, RolloutProducer)
        artifact = producer.run(_request())
        self.assertIs(artifact.status, ProducerExecutionStatus.COMPLETED)
        self.assertEqual(
            artifact.capabilities,
            frozenset({ProducerCapability.RAW_HARNESS_TRACE}),
        )
        self.assertNotIn(ProducerCapability.BEHAVIOR_LOGPROBS, artifact.capabilities)
        self.assertEqual(ProducerArtifact.from_dict(artifact.to_dict()), artifact)

    def test_backend_error_is_infra_invalid(self) -> None:
        capture = PiProcessCapture(
            command=("pi",),
            cwd="/tmp/workspace",
            returncode=0,
            stdout="",
            stderr="",
            record_count=0,
            issues=(),
            final_stop_reason="error",
            backend_error_messages=("500",),
        )
        artifact = PiDirectProducer(PiRunConfig(), runner=lambda **_: capture).run(
            _request()
        )
        self.assertIs(artifact.status, ProducerExecutionStatus.INFRA_INVALID)

    def test_timeout_never_becomes_task_failure(self) -> None:
        def timeout(**_):
            raise subprocess.TimeoutExpired(cmd=("pi",), timeout=30)

        artifact = PiDirectProducer(PiRunConfig(), runner=timeout).run(_request())
        self.assertIs(artifact.status, ProducerExecutionStatus.TIMEOUT)


if __name__ == "__main__":
    unittest.main()
