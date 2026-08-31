from __future__ import annotations

import unittest
from pathlib import Path

from src.assembly.episode_assembler import EpisodeAssembler
from src.capture.pi_adapter import (
    PiJsonAdapter,
    PiOutcomeDeclaration,
    PiRunConfig,
    read_pi_ndjson,
)
from src.contracts.agent_episode import (
    EpisodeVerifierStatus,
    ExecutionValidity,
    IntegrityState,
    TaskStatus,
)
from src.contracts.trace_event import EventStatus, EventType
from tests.execution_fixtures import make_episode_context


FIXTURES = Path(__file__).parents[1] / "fixtures" / "pi"


class PiJsonAdapterTest(unittest.TestCase):
    def _convert(self, fixture: str, *, model: str, success: bool):
        records, issues = read_pi_ndjson((FIXTURES / fixture).read_text())
        config = PiRunConfig(model=model)
        declaration = PiOutcomeDeclaration(
            task_status=TaskStatus.SUCCESS if success else TaskStatus.UNKNOWN,
            execution_validity=ExecutionValidity.VALID if success else ExecutionValidity.UNKNOWN,
            verifier_status=(
                EpisodeVerifierStatus.PASSED if success else EpisodeVerifierStatus.NOT_RUN
            ),
            score=1.0 if success else None,
        )
        return PiJsonAdapter().convert(
            records,
            run_id="run-pi",
            episode_id="episode-pi",
            trace_id="trace-pi",
            config=config,
            declared_outcome=declaration,
            source_issues=issues,
        )

    def test_gpt56_real_probe_maps_model_tool_usage_and_verifier(self) -> None:
        result = self._convert(
            "gpt56_luna_probe.ndjson", model="gpt-5.6-luna", success=True
        )
        types = [event.event_type for event in result.events]
        self.assertIn(EventType.TOOL_CALL, types)
        self.assertIn(EventType.TOOL_RESULT, types)
        self.assertIn(EventType.VERIFICATION_FINISHED, types)
        responses = [
            event for event in result.events if event.event_type is EventType.MODEL_RESPONSE
        ]
        self.assertEqual(responses[0].attributes["usage"]["input_tokens"], 754)
        self.assertFalse(result.backend_error_messages)
        self.assertTrue(all("ENCRYPTED" not in str(event.event_id) for event in result.events))

        context = make_episode_context(capabilities=result.capabilities)
        assembly = EpisodeAssembler().assemble(
            result.events, contexts={"episode-pi": context}
        )
        self.assertEqual(len(assembly.episodes), 1)
        episode = assembly.episodes[0]
        self.assertEqual(episode.integrity.state, IntegrityState.COMPLETE)
        self.assertEqual(episode.outcome.task_status, TaskStatus.SUCCESS)

    def test_backend_error_is_infra_invalid_even_when_process_would_exit_zero(self) -> None:
        result = self._convert(
            "mimo_backend_error.ndjson", model="mimo-v2.5", success=False
        )
        errors = [
            event
            for event in result.events
            if event.event_type is EventType.MODEL_RESPONSE
            and event.status is EventStatus.ERROR
        ]
        self.assertEqual(len(errors), 2)
        terminal = result.events[-1]
        self.assertEqual(terminal.attributes["execution_validity"], "INFRA_INVALID")
        self.assertEqual(terminal.attributes["task_status"], "UNKNOWN")
        self.assertEqual(terminal.attributes["verifier_status"], "NOT_RUN")
        self.assertEqual(len(result.backend_error_messages), 2)

    def test_bad_line_is_reported_without_discarding_neighbors(self) -> None:
        records, issues = read_pi_ndjson(
            '{"type":"session","id":"s"}\nnot-json\n{"type":"agent_end","willRetry":false}\n'
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(len(issues), 1)

    def test_public_fixture_redacts_plaintext_thinking_and_signatures(self) -> None:
        records, issues = read_pi_ndjson(
            '{"message":{"content":[{"thinking":"private chain",'
            '"thinkingSignature":"signed"}]}}\n'
        )
        self.assertFalse(issues)
        content = records[0]["message"]["content"][0]
        self.assertEqual(content["thinking"], "[REDACTED]")
        self.assertEqual(content["thinkingSignature"], "[REDACTED]")

    def test_structured_enoent_becomes_file_not_found_fact(self) -> None:
        records = (
            {
                "type": "tool_execution_end",
                "toolCallId": "call-enoent",
                "toolName": "read",
                "isError": True,
                "result": {
                    "isError": True,
                    "content": [
                        {
                            "type": "text",
                            "text": "ENOENT: no such file or directory, access '/workspace/missing.json'",
                        }
                    ],
                },
            },
        )
        result = PiJsonAdapter().convert(
            records,
            run_id="run-pi",
            episode_id="episode-pi-enoent",
            trace_id="trace-pi-enoent",
            config=PiRunConfig(),
            declared_outcome=PiOutcomeDeclaration(
                task_status=TaskStatus.FAILURE,
                execution_validity=ExecutionValidity.VALID,
                verifier_status=EpisodeVerifierStatus.FAILED,
            ),
            normalize_tool_errors=True,
        )

        tool_result = next(
            event for event in result.events if event.event_type is EventType.TOOL_RESULT
        )
        self.assertEqual(tool_result.attributes["error_code"], "FILE_NOT_FOUND")
        self.assertEqual(tool_result.attributes["error_path"], "/workspace/missing.json")
        self.assertEqual(tool_result.attributes["error_source"], "pi_tool_protocol")
        self.assertIn("ENOENT", str(tool_result.attributes["result"]))

    def test_non_file_tool_error_is_not_relabelled(self) -> None:
        records = (
            {
                "type": "tool_execution_end",
                "toolCallId": "call-permission",
                "toolName": "read",
                "isError": True,
                "result": {"content": [{"type": "text", "text": "EACCES: permission denied"}]},
            },
        )
        result = PiJsonAdapter().convert(
            records,
            run_id="run-pi",
            episode_id="episode-pi-permission",
            trace_id="trace-pi-permission",
            config=PiRunConfig(),
            declared_outcome=PiOutcomeDeclaration(
                task_status=TaskStatus.FAILURE,
                execution_validity=ExecutionValidity.VALID,
                verifier_status=EpisodeVerifierStatus.FAILED,
            ),
            normalize_tool_errors=True,
        )

        tool_result = next(
            event for event in result.events if event.event_type is EventType.TOOL_RESULT
        )
        self.assertNotIn("error_code", tool_result.attributes)

    def test_command_is_fixed_model_read_only_and_has_no_fallback(self) -> None:
        command = PiRunConfig().command("probe")
        self.assertEqual(command[0], "pi")
        self.assertIn("gpt-5.6-luna", command)
        self.assertIn("read,grep,find,ls", command)
        self.assertNotIn("bash", command)
        self.assertNotIn("--models", command)


if __name__ == "__main__":
    unittest.main()
