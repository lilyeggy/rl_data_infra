from __future__ import annotations

import json
import unittest

from src.capture.pi_adapter import (
    PiJsonAdapter,
    PiOutcomeDeclaration,
    read_pi_ndjson,
)
from src.contracts.agent_episode import ExecutionValidity, TaskStatus
from examples.legacy_scenarios.real_pi_v21_live import (
    CANDIDATE_POLICY,
    CONTROL_POLICY,
    build_prompt,
    task_file_payload,
)
from examples.legacy_scenarios.real_pi_v2 import verify_reference_answer


class RealPiV21LiveTest(unittest.TestCase):
    def test_prompts_are_deterministic_and_policies_are_explicit(self) -> None:
        control = build_prompt(1, "control")
        candidate = build_prompt(1, "candidate")

        self.assertIn("missing-1.json", control)
        self.assertIn("retry", control)
        self.assertIn("task-*.json", candidate)
        self.assertIn("find", candidate)
        self.assertNotEqual(control, candidate)
        self.assertEqual(build_prompt(1, "control"), control)
        self.assertIn("no markdown, no code fences", candidate)
        self.assertIn("exactly the string SUCCESS", candidate)

    def test_task_payload_matches_expected_failure_counts(self) -> None:
        for task_index in (1, 2, 3):
            payload = task_file_payload(task_index)
            self.assertEqual(payload["task_id"], f"task-{task_index}")
            failures = sum(
                record["status"] == "FAILURE" for record in payload["records"]
            )
            expected = {1: 2, 2: 2, 3: 3}[task_index]
            self.assertEqual(failures, expected)

    def test_strict_verifier_rejects_code_fenced_json(self) -> None:
        records = (
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "stopReason": "stop",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                '```json\n'
                                '{"task_id":"task-1","status":"SUCCESS","failure_count":2}'
                                '\n```'
                            ),
                        }
                    ],
                },
            },
        )
        declaration, _ = verify_reference_answer(records, task_index=1)

        self.assertEqual(declaration.task_status, TaskStatus.FAILURE)
        self.assertEqual(declaration.execution_validity, ExecutionValidity.VALID)

    def test_capture_evidence_path_accepts_real_ndjson_shape(self) -> None:
        answer = '{"task_id":"task-1","status":"SUCCESS","failure_count":2}'
        records, issues = read_pi_ndjson(
            "".join(
                json.dumps(record) + "\n"
                for record in (
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "stopReason": "toolUse",
                            "content": [
                                {
                                    "type": "toolCall",
                                    "id": "call-1",
                                    "name": "read",
                                    "arguments": {"path": "task-1.json"},
                                }
                            ],
                            "timestamp": 1787040000000,
                        },
                    },
                    {
                        "type": "tool_execution_end",
                        "toolCallId": "call-1",
                        "toolName": "read",
                        "isError": False,
                        "result": {"content": [{"type": "text", "text": answer}]},
                    },
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "stopReason": "stop",
                            "content": [{"type": "text", "text": answer}],
                            "timestamp": 1787040001000,
                        },
                    },
                    {"type": "agent_end", "willRetry": False},
                )
            )
        )
        self.assertFalse(issues)
        declaration, _ = verify_reference_answer(records, task_index=1)
        self.assertEqual(declaration.task_status, TaskStatus.SUCCESS)
        result = PiJsonAdapter().convert(
            records,
            run_id="run-test",
            episode_id="episode-test",
            trace_id="trace-test",
            config=__import__("src.capture.pi_adapter", fromlist=["PiRunConfig"]).PiRunConfig(),
            declared_outcome=declaration,
        )
        self.assertEqual(result.events[-1].attributes["task_status"], "SUCCESS")


if __name__ == "__main__":
    unittest.main()
