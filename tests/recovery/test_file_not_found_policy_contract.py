from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.recovery.file_not_found_policy import (
    BoundedFileNotFoundRecoveryPolicy,
    FILE_NOT_FOUND_POLICY_VERSION,
    RecoveryAction,
    RecoveryDecision,
    RecoveryInput,
)
from src.errors import ContractValidationError


FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "recovery"
    / "v2.1"
    / "recovery-input-cases.json"
)


class FileNotFoundPolicyContractTest(unittest.TestCase):
    def test_input_is_immutable_and_preserves_observable_context(self) -> None:
        value = RecoveryInput(
            error_code="FILE_NOT_FOUND",
            error_path="workspace/missing.json",
            scope_key="workspace",
            discovery_cache={"workspace": ("workspace/task.json",)},
            remaining_discovery_budget=1,
            trigger_event_ids=("evt-error",),
            attributes={"source": "tool-result"},
        )

        self.assertEqual(value.discovery_cache["workspace"], ("workspace/task.json",))
        self.assertEqual(value.trigger_event_ids, ("evt-error",))
        with self.assertRaises(TypeError):
            value.attributes["source"] = "changed"  # type: ignore[index]

    def test_decision_is_deterministic_and_checksumed(self) -> None:
        value = RecoveryDecision(
            action=RecoveryAction.DISCOVER,
            reason_code="FILE_NOT_FOUND",
            trigger_event_ids=("evt-error",),
            scope_key="workspace",
            tool_name="find",
            arguments={"pattern": "*.json", "path": "workspace"},
            budget_before=1,
            budget_after=0,
            cache_hit=False,
        )

        self.assertEqual(value.policy_version, FILE_NOT_FOUND_POLICY_VERSION)
        self.assertEqual(value.to_dict()["action"], "DISCOVER")
        self.assertEqual(
            value.checksum, RecoveryDecision.from_dict(value.to_dict()).checksum
        )

    def test_input_fixture_is_explicit_and_does_not_freeze_transitions(self) -> None:
        payload = json.loads(FIXTURE.read_text())

        self.assertEqual(payload["schema_version"], "recovery-input-fixtures/v1")
        self.assertTrue(payload["cases"])
        for case in payload["cases"]:
            value = RecoveryInput(**{key: item for key, item in case.items() if key != "case_id"})
            self.assertTrue(value.trigger_event_ids)
            self.assertTrue(value.scope_key)

    def test_first_file_not_found_uses_one_bounded_discovery(self) -> None:
        decision = BoundedFileNotFoundRecoveryPolicy().decide(
            RecoveryInput(
                error_code="FILE_NOT_FOUND",
                error_path="workspace/missing-1.json",
                scope_key="workspace",
                remaining_discovery_budget=1,
                trigger_event_ids=("evt-error",),
            )
        )

        self.assertEqual(decision.action, RecoveryAction.DISCOVER)
        self.assertEqual(decision.tool_name, "find")
        self.assertEqual(decision.arguments["pattern"], "*-1.json")
        self.assertEqual(decision.arguments["path"], "workspace")
        self.assertEqual(decision.budget_before, 1)
        self.assertEqual(decision.budget_after, 0)
        self.assertFalse(decision.cache_hit)

    def test_unique_cached_candidate_reads_without_spending_budget(self) -> None:
        decision = BoundedFileNotFoundRecoveryPolicy().decide(
            RecoveryInput(
                error_code="FILE_NOT_FOUND",
                error_path="/workspace/missing-1.json",
                scope_key="/workspace",
                discovery_cache={"/workspace": ("task-1.json",)},
                remaining_discovery_budget=0,
            )
        )

        self.assertEqual(decision.action, RecoveryAction.READ)
        self.assertEqual(decision.tool_name, "read")
        self.assertEqual(decision.arguments["path"], "/workspace/task-1.json")
        self.assertEqual(decision.budget_before, 0)
        self.assertEqual(decision.budget_after, 0)
        self.assertTrue(decision.cache_hit)

    def test_empty_and_ambiguous_cache_terminate_deterministically(self) -> None:
        policy = BoundedFileNotFoundRecoveryPolicy()
        base = {
            "error_code": "FILE_NOT_FOUND",
            "error_path": "workspace/missing.json",
            "scope_key": "workspace",
            "remaining_discovery_budget": 0,
        }

        empty = policy.decide(RecoveryInput(**base, discovery_cache={"workspace": ()}))
        ambiguous = policy.decide(
            RecoveryInput(
                **base,
                discovery_cache={
                    "workspace": ("workspace/z.json", "workspace/a.json", "workspace/a.json")
                },
            )
        )

        self.assertEqual(empty.action, RecoveryAction.TERMINATE)
        self.assertEqual(empty.reason_code, "NO_DISCOVERY_CANDIDATE")
        self.assertEqual(ambiguous.action, RecoveryAction.TERMINATE)
        self.assertEqual(ambiguous.reason_code, "AMBIGUOUS_DISCOVERY_CANDIDATES")
        self.assertEqual(
            ambiguous.selected_candidates,
            ("workspace/a.json", "workspace/z.json"),
        )

    def test_budget_exhaustion_and_unsafe_candidates_do_not_search_or_read(self) -> None:
        policy = BoundedFileNotFoundRecoveryPolicy()
        exhausted = policy.decide(
            RecoveryInput(
                error_code="FILE_NOT_FOUND",
                error_path="workspace/missing.json",
                scope_key="workspace",
                remaining_discovery_budget=0,
            )
        )
        unsafe = policy.decide(
            RecoveryInput(
                error_code="FILE_NOT_FOUND",
                error_path="/workspace/missing.json",
                scope_key="/workspace",
                discovery_cache={"/workspace": ("../../outside.json",)},
                remaining_discovery_budget=0,
            )
        )
        relative_escape = policy.decide(
            RecoveryInput(
                error_code="FILE_NOT_FOUND",
                error_path="missing.json",
                scope_key=".",
                discovery_cache={".": ("../outside.json",)},
                remaining_discovery_budget=0,
            )
        )

        self.assertEqual(exhausted.action, RecoveryAction.TERMINATE)
        self.assertEqual(exhausted.reason_code, "RECOVERY_BUDGET_EXHAUSTED")
        self.assertEqual(unsafe.action, RecoveryAction.TERMINATE)
        self.assertEqual(unsafe.reason_code, "NO_DISCOVERY_CANDIDATE")
        self.assertEqual(relative_escape.reason_code, "NO_DISCOVERY_CANDIDATE")

    def test_pattern_extraction_is_path_only_and_deterministic(self) -> None:
        policy = BoundedFileNotFoundRecoveryPolicy

        self.assertEqual(policy.extract_pattern("/workspace/missing-17.json"), "*-17.json")
        self.assertEqual(policy.extract_pattern("workspace/task-*.json"), "task-*.json")
        self.assertEqual(policy.extract_pattern("workspace/config.json"), "config.json")

    def test_budget_cannot_increase_and_policy_is_explicit(self) -> None:
        with self.assertRaises(ContractValidationError):
            RecoveryDecision(
                action=RecoveryAction.READ,
                reason_code="UNIQUE_CANDIDATE",
                budget_before=0,
                budget_after=1,
            )

        decision = BoundedFileNotFoundRecoveryPolicy().decide(
            RecoveryInput(
                error_code="PERMISSION_DENIED",
                error_path="missing.json",
                scope_key="workspace",
                remaining_discovery_budget=1,
            )
        )
        self.assertEqual(decision.action, RecoveryAction.NO_ACTION)


if __name__ == "__main__":
    unittest.main()
