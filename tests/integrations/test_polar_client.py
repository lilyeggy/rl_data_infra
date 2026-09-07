from __future__ import annotations

import unittest

from src.integrations.polar import PolarClientError, PolarHttpClient, PolarTaskRequest


def _task() -> PolarTaskRequest:
    return PolarTaskRequest(
        task_id="task-polar-1",
        instruction="fix the repository",
        num_samples=4,
        timeout_seconds=900,
        runtime={"backend": "docker", "image": "fixture-image"},
        agent={"harness": "pi", "model_name": "openai/model"},
        builder={"strategy": "prefix_merging"},
        evaluator={"strategy": "test_on_output", "config": {"test_command": "pytest"}},
        metadata={"group_id": "group-1", "policy_version": "policy-v0"},
    )


class PolarHttpClientTest(unittest.TestCase):
    def test_submit_uses_documented_endpoint_and_preserves_training_metadata(self) -> None:
        calls = []

        def transport(method, url, payload, timeout):
            calls.append((method, url, payload, timeout))
            return {"task_id": "task-polar-1", "status": "running"}

        client = PolarHttpClient("http://polar:8080/", transport=transport)
        response = client.submit_task(_task())
        self.assertEqual(response["status"], "running")
        method, url, payload, _ = calls[0]
        self.assertEqual(method, "POST")
        self.assertEqual(url, "http://polar:8080/rollout/task/submit")
        self.assertEqual(payload["metadata"]["policy_version"], "policy-v0")
        self.assertEqual(payload["num_samples"], 4)

    def test_get_task_escapes_id_and_rejects_identity_mismatch(self) -> None:
        calls = []

        def transport(method, url, payload, timeout):
            calls.append(url)
            return {"task_id": "wrong", "status": "running"}

        client = PolarHttpClient("http://polar:8080", transport=transport)
        with self.assertRaises(PolarClientError):
            client.get_task("task/unsafe")
        self.assertTrue(calls[0].endswith("/rollout/task/task%2Funsafe"))

    def test_submit_response_must_match_requested_task(self) -> None:
        client = PolarHttpClient(
            "http://polar:8080",
            transport=lambda *_: {"task_id": "other", "status": "running"},
        )
        with self.assertRaises(PolarClientError):
            client.submit_task(_task())


if __name__ == "__main__":
    unittest.main()
