from __future__ import annotations

import unittest
from dataclasses import replace

from src.assembly.episode_assembler import EpisodeAssembler
from src.contracts.agent_episode import (
    AgentEpisode,
    IntegrityState,
    TaskStatus,
)
from tests.execution_fixtures import (
    make_complete_event_stream,
    make_episode_context,
)


class EpisodeAssemblerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assembler = EpisodeAssembler()
        self.context = make_episode_context()

    def test_assembles_complete_episode_and_roundtrips(self) -> None:
        result = self.assembler.assemble(
            make_complete_event_stream(),
            contexts={"episode-001": self.context},
        )

        self.assertEqual(len(result.episodes), 1)
        episode = result.episodes[0]
        self.assertIs(episode.integrity.state, IntegrityState.COMPLETE)
        self.assertIs(episode.outcome.task_status, TaskStatus.SUCCESS)
        self.assertEqual(episode.termination.reason, "VERIFIER_PASSED")
        self.assertEqual(AgentEpisode.from_dict(episode.to_dict()), episode)

    def test_out_of_order_and_exact_duplicate_produce_stable_canonical_output(self) -> None:
        events = make_complete_event_stream()
        ordered = self.assembler.assemble(
            events,
            contexts={"episode-001": self.context},
        ).episodes[0]
        replayed = self.assembler.assemble(
            tuple(reversed(events)) + (events[2],),
            contexts={"episode-001": self.context},
        ).episodes[0]

        self.assertEqual(replayed.events, ordered.events)
        self.assertEqual(
            replayed.integrity.output_checksum, ordered.integrity.output_checksum
        )
        self.assertEqual(replayed.integrity.duplicate_event_count, 1)

    def test_missing_terminal_and_sequence_gap_are_partial_not_discarded(self) -> None:
        events = make_complete_event_stream()
        partial = tuple(event for event in events if event.sequence not in (1, 6))

        episode = self.assembler.assemble(
            partial,
            contexts={"episode-001": self.context},
        ).episodes[0]

        self.assertIs(episode.integrity.state, IntegrityState.PARTIAL)
        self.assertIn(1, episode.integrity.sequence_gaps)
        self.assertEqual(episode.termination.reason, "CAPTURE_INTERRUPTED")
        self.assertIs(episode.outcome.task_status, TaskStatus.UNKNOWN)

    def test_conflicting_duplicate_marks_episode_corrupt_and_quarantines_replay(self) -> None:
        events = make_complete_event_stream()
        conflict = replace(events[1], attributes={"different": True})

        result = self.assembler.assemble(
            events + (conflict,),
            contexts={"episode-001": self.context},
        )

        self.assertIs(result.episodes[0].integrity.state, IntegrityState.CORRUPT)
        self.assertEqual(result.quarantined_events[0].reason_code, "CONFLICTING_EVENT_ID")

    def test_missing_context_is_quarantined_instead_of_inferred(self) -> None:
        result = self.assembler.assemble(make_complete_event_stream(), contexts={})

        self.assertFalse(result.episodes)
        self.assertEqual(len(result.quarantined_events), 7)
        self.assertTrue(
            all(
                item.reason_code == "EPISODE_CONTEXT_MISSING"
                for item in result.quarantined_events
            )
        )


if __name__ == "__main__":
    unittest.main()
