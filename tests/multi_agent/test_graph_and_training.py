from __future__ import annotations

import unittest
from dataclasses import replace

from src.contracts.trace_event import EventType
from examples.legacy_scenarios.demo_v22 import build_synthetic_multi_agent_episode
from src.multi_agent import (
    EvidenceStatus,
    RelationType,
    build_multi_agent_execution_graph,
    build_multi_agent_training_views,
)
from tests.execution_fixtures import make_complete_event_stream
from tests.execution_fixtures import make_episode_context
from src.assembly.episode_assembler import EpisodeAssembler


class MultiAgentGraphAndTrainingTest(unittest.TestCase):
    def test_explicit_fixture_builds_observed_graph(self) -> None:
        episode = build_synthetic_multi_agent_episode()
        graph = build_multi_agent_execution_graph(episode)

        self.assertEqual(graph.graph_status, EvidenceStatus.OBSERVED)
        self.assertEqual(len(graph.agents), 2)
        self.assertEqual(graph.field_status["parent_child"], EvidenceStatus.OBSERVED)
        self.assertEqual(graph.field_status["message_routing"], EvidenceStatus.OBSERVED)
        self.assertEqual(graph.field_status["reward_ownership"], EvidenceStatus.OBSERVED)
        self.assertEqual(
            graph.field_status["credit_assignment"], EvidenceStatus.NOT_OBSERVABLE
        )
        self.assertEqual(
            sum(item.relation_type is RelationType.PARENT_CHILD for item in graph.relations),
            1,
        )
        self.assertEqual(
            sum(item.relation_type is RelationType.TOOL_OWNERSHIP for item in graph.relations),
            1,
        )
        self.assertEqual(graph.checksum, build_multi_agent_execution_graph(episode).checksum)

    def test_missing_agent_metadata_is_not_observable(self) -> None:
        episode = EpisodeAssembler().assemble(
            make_complete_event_stream(),
            contexts={"episode-001": make_episode_context()},
        ).episodes[0]

        graph = build_multi_agent_execution_graph(episode)

        self.assertEqual(graph.graph_status, EvidenceStatus.NOT_OBSERVABLE)
        self.assertFalse(graph.agents)
        self.assertEqual(
            graph.field_status["reward_ownership"], EvidenceStatus.NOT_OBSERVABLE
        )

    def test_partial_action_ownership_is_insufficient_evidence(self) -> None:
        episode = build_synthetic_multi_agent_episode()
        events = list(episode.events)
        tool_call = next(event for event in events if event.event_type is EventType.TOOL_CALL)
        changed = replace(
            tool_call,
            attributes={
                key: value for key, value in tool_call.attributes.items() if key != "agent_id"
            },
        )
        events[events.index(tool_call)] = changed
        partial = replace(episode, events=tuple(events))

        graph = build_multi_agent_execution_graph(partial)

        self.assertEqual(graph.graph_status, EvidenceStatus.INSUFFICIENT_EVIDENCE)
        self.assertIn(tool_call.event_id, graph.unowned_event_ids)

    def test_conflicting_parent_metadata_is_invalid(self) -> None:
        episode = build_synthetic_multi_agent_episode()
        events = list(episode.events)
        researcher_event = next(
            event
            for event in events
            if event.attributes.get("agent_id") == "agent-researcher"
        )
        changed_attributes = dict(researcher_event.attributes)
        changed_attributes["parent_agent_id"] = "agent-unknown"
        events[events.index(researcher_event)] = replace(
            researcher_event,
            attributes=changed_attributes,
        )
        invalid = replace(episode, events=tuple(events))

        graph = build_multi_agent_execution_graph(invalid)

        self.assertEqual(graph.graph_status, EvidenceStatus.INVALID)
        self.assertEqual(graph.field_status["parent_child"], EvidenceStatus.INVALID)

    def test_training_view_does_not_infer_agent_level_rl(self) -> None:
        episode = build_synthetic_multi_agent_episode()
        graph = build_multi_agent_execution_graph(episode)
        views = build_multi_agent_training_views(
            episode,
            graph,
            target_policy_by_agent={
                "agent-orchestrator": "policy-orchestrator-v1",
                "agent-researcher": "policy-researcher-v1",
            },
        )

        self.assertEqual(len(views), 2)
        self.assertTrue(all(view.sft_candidate for view in views))
        self.assertTrue(all(not view.on_policy_rl_candidate for view in views))
        orchestrator = next(view for view in views if view.agent_id == "agent-orchestrator")
        researcher = next(view for view in views if view.agent_id == "agent-researcher")
        self.assertEqual(orchestrator.reward_ownership_status, EvidenceStatus.OBSERVED)
        self.assertEqual(researcher.reward_ownership_status, EvidenceStatus.NOT_OBSERVABLE)
        self.assertIn("MODEL_TOKEN_IDS", orchestrator.missing_capabilities)
        self.assertIn(
            "agent-level reward ownership is not observable",
            researcher.exclusion_reasons,
        )
        self.assertIn(
            "agent-level credit assignment is not observable",
            researcher.exclusion_reasons,
        )


if __name__ == "__main__":
    unittest.main()
