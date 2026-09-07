from __future__ import annotations

import unittest
from dataclasses import replace

from src.contracts.agent_episode import (
    EpisodeOutcome,
    EpisodeVerifierStatus,
    ExecutionValidity,
    TaskStatus,
)
from src.contracts.capabilities import Capability, capabilities_for_record
from src.contracts.rollout_record import RolloutStatus, VerifierStatus
from examples.legacy_scenarios.demo_v22 import build_synthetic_multi_agent_episode
from src.errors import CapabilityMissingError
from src.exporters.training_view import (
    build_preference_pairs_with_reasons,
    project_episode_to_rollout,
)


class TrainingViewExporterTest(unittest.TestCase):
    def test_episode_projection_preserves_identity_and_semantics(self) -> None:
        episode = build_synthetic_multi_agent_episode()
        record = project_episode_to_rollout(episode)

        self.assertEqual(record.trajectory_id, episode.episode_id)
        self.assertEqual(record.task_id, episode.task_id)
        self.assertEqual(record.verifier_status, VerifierStatus.PASSED)
        self.assertEqual(record.rollout_status, RolloutStatus.COMPLETED)
        self.assertEqual(record.reward, 1.0)
        self.assertEqual(record.source_payload_sha256, episode.checksum)
        self.assertEqual(record.opaque_metadata["episode_checksum"], episode.checksum)
        self.assertTrue(record.tool_events)
        self.assertIsNone(record.token_ids)
        self.assertIsNone(record.old_logprobs)
        self.assertIsNone(record.action_mask)

    def test_infra_invalid_episode_never_receives_fabricated_reward(self) -> None:
        episode = build_synthetic_multi_agent_episode()
        invalid = replace(
            episode,
            outcome=EpisodeOutcome(
                task_status=TaskStatus.UNKNOWN,
                execution_validity=ExecutionValidity.INFRA_INVALID,
                verifier_status=EpisodeVerifierStatus.NOT_RUN,
                score=None,
                evidence_event_ids=episode.outcome.evidence_event_ids,
            ),
        )

        record = project_episode_to_rollout(invalid)

        self.assertIsNone(record.reward)
        self.assertEqual(record.rollout_status, RolloutStatus.INCOMPLETE)

    def test_capability_gate_rejects_token_requiring_consumer(self) -> None:
        episode = build_synthetic_multi_agent_episode()
        record = project_episode_to_rollout(episode)
        available = capabilities_for_record(record)

        self.assertNotIn(Capability.TOKEN_IDS, available)
        with self.assertRaises(CapabilityMissingError):
            from src.contracts.capabilities import require_capabilities

            require_capabilities(
                available,
                frozenset({Capability.TOKEN_IDS, Capability.ACTION_MASK}),
                context="on-policy-rl-consumer",
            )

    def test_preference_pairs_choose_success_over_valid_failure(self) -> None:
        from src.assembly.episode_assembler import EpisodeAssembler
        from src.contracts.trace_event import EventStatus, EventType
        from tests.execution_fixtures import (
            make_complete_event_stream,
            make_episode_context,
        )

        success_events = make_complete_event_stream(episode_id="episode-pref-success")
        failure_events = list(
            make_complete_event_stream(episode_id="episode-pref-failure")
        )
        # The failure side carries REAL verifier FAILED evidence: the
        # VERIFICATION_FINISHED event is FAILED/passed=False (not just a
        # terminal that claims FAILURE while the verifier event still passed).
        failure_events = [
            replace(
                event,
                status=EventStatus.FAILED,
                attributes={**event.attributes, "passed": False, "score": 0.0},
            )
            if event.event_type is EventType.VERIFICATION_FINISHED
            else event
            for event in failure_events
        ]
        terminal_index = next(
            index
            for index, event in enumerate(failure_events)
            if event.event_type is EventType.EPISODE_FINISHED
        )
        terminal = failure_events[terminal_index]
        failure_events[terminal_index] = replace(
            terminal,
            attributes={
                **terminal.attributes,
                "task_status": "FAILURE",
                "execution_validity": "VALID",
                "verifier_status": "FAILED",
                "score": 0.0,
            },
        )
        assembler = EpisodeAssembler()
        context = make_episode_context(task_id="task-pref-1")
        success = assembler.assemble(
            success_events, contexts={"episode-pref-success": context}
        ).episodes[0]
        failure = assembler.assemble(
            tuple(failure_events), contexts={"episode-pref-failure": context}
        ).episodes[0]

        pairs, rejections = build_preference_pairs_with_reasons((success, failure))
        self.assertEqual(rejections, ())
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].chosen_episode_id, success.episode_id)
        self.assertEqual(pairs[0].rejected_episode_id, failure.episode_id)
        self.assertEqual(pairs[0].chosen_verifier, "PASSED")
        self.assertEqual(pairs[0].rejected_verifier, "FAILED")
        self.assertEqual(
            pairs[0].chosen_rollout_checksum,
            project_episode_to_rollout(success).checksum,
        )

    def test_preference_pair_rejected_on_model_revision_mismatch(self) -> None:
        from dataclasses import replace

        from src.assembly.episode_assembler import EpisodeAssembler
        from src.contracts.trace_event import EventStatus, EventType
        from src.exporters.training_view import build_preference_pairs_with_reasons
        from tests.execution_fixtures import (
            make_complete_event_stream,
            make_episode_context,
        )

        def build(episode_id: str, revision: str, fail: bool):
            events = list(make_complete_event_stream(episode_id=episode_id))
            if fail:
                events = [
                    replace(
                        e,
                        status=EventStatus.FAILED,
                        attributes={**e.attributes, "passed": False, "score": 0.0},
                    )
                    if e.event_type is EventType.VERIFICATION_FINISHED
                    else e
                    for e in events
                ]
            idx = next(
                i
                for i, e in enumerate(events)
                if e.event_type is EventType.EPISODE_FINISHED
            )
            events[idx] = replace(
                events[idx],
                attributes={
                    **events[idx].attributes,
                    "task_status": "FAILURE" if fail else "SUCCESS",
                    "execution_validity": "VALID",
                    "verifier_status": "FAILED" if fail else "PASSED",
                    "score": 0.0 if fail else 1.0,
                },
            )
            context = make_episode_context(
                task_id="task-pref-rev",
                model_manifest=replace(
                    make_episode_context().model_manifest, revision=revision
                ),
            )
            return EpisodeAssembler().assemble(
                tuple(events), contexts={episode_id: context}
            ).episodes[0]

        success = build("ep-pref-s", revision="rev-A", fail=False)
        failure = build("ep-pref-f", revision="rev-B", fail=True)

        pairs, rejections = build_preference_pairs_with_reasons((success, failure))
        self.assertEqual(pairs, ())
        self.assertEqual(len(rejections), 1)
        self.assertIn("model_revision", rejections[0].identity_mismatch)
        self.assertTrue(any("identity mismatch" in r for r in rejections[0].reasons))


if __name__ == "__main__":
    unittest.main()
