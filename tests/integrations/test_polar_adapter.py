from __future__ import annotations

import unittest

from src.contracts._json import sha256_json
from src.integrations.polar import (
    IDENTITY_METADATA_KEY,
    PolarBatchContext,
    PolarHttpClient,
    PolarResultNotReady,
    PolarSchemaError,
    PolarTaskRequest,
    adapt_polar_task_result,
)
from src.producers import ProducerCapability, ProducerExecutionStatus
from src.producers.polar_live import PolarLiveBatchProducer


def _context(*, samples: int = 2) -> PolarBatchContext:
    return PolarBatchContext(
        run_id="run-1",
        logical_task_id="swebench/example-1",
        polar_task_id="polar-task-1",
        expected_samples=samples,
        producer_version="polar-stable@abc123",
        group_id="group-1",
        policy_fingerprint=sha256_json({"policy": "checkpoint-7"}),
        sampling_fingerprint=sha256_json({"temperature": 1.0}),
        evaluator_fingerprint=sha256_json(
            {"strategy": "swebench_harness", "config": {"instance": "example-1"}}
        ),
    )


def _trace(*, logprobs=True, reward=0.0):
    return {
        "prompt_ids": [10, 11],
        "response_ids": [20, 21],
        "loss_mask": [1, 1],
        "response_logprobs": [-0.4, -0.2] if logprobs else None,
        "reward": reward,
        "prompt_messages": [{"role": "user", "content": "fix it"}],
        "response_messages": [{"role": "assistant", "content": "done"}],
        "metadata": {},
    }


def _session(context, session_id, *, trace=None, status="COMPLETED"):
    return {
        "session_id": session_id,
        "task_id": context.polar_task_id,
        "status": status,
        "trajectory": {
            "status": status,
            "metadata": {IDENTITY_METADATA_KEY: context.identity_metadata()},
            "traces": [_trace() if trace is None else trace],
            "error": None,
        },
        "timing": {"run_ms": 120.0},
        "node_id": "node-1",
        "error": None,
        "metadata": {},
    }


def _task_result(context, sessions, *, status="completed"):
    return {
        "task_id": context.polar_task_id,
        "status": status,
        "total_sessions": context.expected_samples,
        "completed_sessions": context.expected_samples,
        "results": sessions,
        "result_paths": [],
    }


class PolarAdapterTest(unittest.TestCase):
    def test_one_identity_per_session_with_strict_training_capabilities(self) -> None:
        context = _context()
        raw = _task_result(
            context,
            [
                _session(context, "session-b"),
                _session(context, "session-a", trace=_trace(reward=1.0)),
            ],
        )

        artifacts = adapt_polar_task_result(raw, context=context)

        self.assertEqual([item.identity.attempt_id for item in artifacts], [1, 2])
        self.assertTrue(artifacts[0].identity.episode_id.endswith("/session-a"))
        self.assertTrue(artifacts[1].identity.episode_id.endswith("/session-b"))
        self.assertEqual(artifacts[1].status, ProducerExecutionStatus.COMPLETED)
        for artifact in artifacts:
            self.assertIn(ProducerCapability.TOKEN_IDS, artifact.capabilities)
            self.assertIn(ProducerCapability.ACTION_MASK, artifact.capabilities)
            self.assertIn(ProducerCapability.BEHAVIOR_LOGPROBS, artifact.capabilities)
            self.assertIn(ProducerCapability.VERIFIER_EVIDENCE, artifact.capabilities)
            self.assertIn(ProducerCapability.POLICY_VERSION, artifact.capabilities)

    def test_missing_logprobs_is_observable_but_not_claimed(self) -> None:
        context = _context(samples=1)
        raw = _task_result(
            context,
            [_session(context, "session-a", trace=_trace(logprobs=False))],
        )
        artifact = adapt_polar_task_result(raw, context=context)[0]
        self.assertNotIn(ProducerCapability.BEHAVIOR_LOGPROBS, artifact.capabilities)
        self.assertTrue(any("logprobs" in issue for issue in artifact.issues))

    def test_identity_metadata_mismatch_fails_closed(self) -> None:
        context = _context(samples=1)
        session = _session(context, "session-a")
        session["trajectory"]["metadata"][IDENTITY_METADATA_KEY]["run_id"] = "wrong"
        with self.assertRaises(PolarSchemaError):
            adapt_polar_task_result(_task_result(context, [session]), context=context)

    def test_array_misalignment_fails_closed(self) -> None:
        context = _context(samples=1)
        trace = _trace()
        trace["loss_mask"] = [1]
        with self.assertRaises(PolarSchemaError):
            adapt_polar_task_result(
                _task_result(context, [_session(context, "session-a", trace=trace)]),
                context=context,
            )

    def test_nonterminal_task_is_not_imported(self) -> None:
        context = _context(samples=1)
        raw = {
            "task_id": context.polar_task_id,
            "status": "running",
            "total_sessions": 1,
            "completed_sessions": 0,
            "results": [],
        }
        with self.assertRaises(PolarResultNotReady):
            adapt_polar_task_result(raw, context=context)

    def test_timeout_is_infra_status_and_never_verifier_evidence(self) -> None:
        context = _context(samples=1)
        raw = _task_result(
            context,
            [_session(context, "session-timeout", status="TIMEOUT")],
            status="failed",
        )
        artifact = adapt_polar_task_result(raw, context=context)[0]
        self.assertEqual(artifact.status, ProducerExecutionStatus.TIMEOUT)
        self.assertNotIn(ProducerCapability.VERIFIER_EVIDENCE, artifact.capabilities)


class PolarLiveBatchProducerTest(unittest.TestCase):
    def test_submit_injects_reserved_identity_and_collects(self) -> None:
        context = _context(samples=1)
        calls = []

        def transport(method, url, payload, timeout):
            calls.append((method, url, payload, timeout))
            if method == "POST":
                return {"task_id": context.polar_task_id, "status": "running"}
            return _task_result(context, [_session(context, "session-a")])

        client = PolarHttpClient("http://polar:8080", transport=transport)
        producer = PolarLiveBatchProducer(client, producer_version=context.producer_version)
        task = PolarTaskRequest(
            task_id=context.polar_task_id,
            instruction="fix repository",
            num_samples=1,
            timeout_seconds=900,
            runtime={"backend": "docker", "image": "image"},
            agent={"harness": "pi", "model_name": "model"},
            builder={"strategy": "prefix_merging"},
            evaluator={"strategy": "swebench_harness"},
            metadata={"rollout_step": 7},
        )

        producer.submit(task, context=context)
        submitted_metadata = calls[0][2]["metadata"]
        self.assertEqual(
            submitted_metadata[IDENTITY_METADATA_KEY], context.identity_metadata()
        )
        artifacts = producer.collect(context=context)
        self.assertEqual(len(artifacts), 1)


if __name__ == "__main__":
    unittest.main()
