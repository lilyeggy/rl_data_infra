from __future__ import annotations

import unittest

from src.capture import (
    ModelBackendResponse,
    ModelEndpointKind,
    ModelEvidenceCapability,
    ModelProxyRequest,
    capture_model_call,
)
from src.contracts.execution_identity import ExecutionIdentity
from src.errors import ContractValidationError


def _identity(*, policy: bool) -> ExecutionIdentity:
    return ExecutionIdentity(
        run_id="run-proxy",
        task_id="task-proxy",
        episode_id="episode-proxy",
        attempt_id=1,
        producer_id="local-docker",
        producer_version="local-docker-launcher/v1",
        policy_fingerprint="a" * 64 if policy else None,
        sampling_fingerprint="b" * 64,
    )


def _request(kind: ModelEndpointKind, *, policy: bool) -> ModelProxyRequest:
    return ModelProxyRequest(
        identity=_identity(policy=policy),
        request_id="request-1",
        endpoint_kind=kind,
        model_id="example-14b",
        messages=({"role": "user", "content": "fix it"},),
        sampling_config={"temperature": 0.7, "top_p": 0.95},
    )


class ModelProxyEvidenceTest(unittest.TestCase):
    def test_external_text_api_is_observable_but_not_claimed_for_rl(self) -> None:
        evidence = capture_model_call(
            _request(ModelEndpointKind.EXTERNAL_API, policy=False),
            ModelBackendResponse(
                response={"text": "done"},
                latency_ms=10,
                status_code=200,
                usage={"output_tokens": 1},
            ),
        )
        self.assertEqual(
            evidence.capabilities,
            frozenset({ModelEvidenceCapability.REQUEST_RESPONSE}),
        )
        self.assertFalse(evidence.rl_usable_call)
        self.assertTrue(evidence.issues)

    def test_controlled_backend_with_native_arrays_is_rl_usable_call_evidence(self) -> None:
        evidence = capture_model_call(
            _request(ModelEndpointKind.CONTROLLED, policy=True),
            ModelBackendResponse(
                response={"text": "done"},
                latency_ms=10,
                status_code=200,
                backend_model_revision="checkpoint-0001",
                prompt_token_ids=(10, 11),
                response_token_ids=(20, 21),
                response_logprobs=(-0.1, -0.2),
            ),
        )
        self.assertTrue(evidence.rl_usable_call)
        self.assertEqual(len(evidence.checksum), 64)

    def test_logprobs_cannot_exist_without_aligned_native_tokens(self) -> None:
        with self.assertRaises(ContractValidationError):
            ModelBackendResponse(
                response={"text": "done"},
                latency_ms=10,
                status_code=200,
                response_token_ids=(20,),
                response_logprobs=(-0.1, -0.2),
            )


if __name__ == "__main__":
    unittest.main()
