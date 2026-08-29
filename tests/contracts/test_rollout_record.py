from __future__ import annotations

import unittest

from src.contracts import (
    Capability,
    RolloutRecord,
    capabilities_for_record,
    require_capabilities,
)
from src.errors import CapabilityMissingError, ContractValidationError
from tests.contract_fixtures import make_record


class RolloutRecordTest(unittest.TestCase):
    def test_valid_record_is_immutable_and_exposes_capabilities(self) -> None:
        record = make_record()

        self.assertEqual(record.trainable_token_count, 3)
        self.assertEqual(set(capabilities_for_record(record)), set(Capability))
        with self.assertRaises(TypeError):
            record.opaque_metadata["new"] = "value"  # type: ignore[index]

    def test_roundtrip_and_checksum_are_deterministic(self) -> None:
        first = make_record(opaque_metadata={"b": 2, "a": 1})
        second = make_record(opaque_metadata={"a": 1, "b": 2})

        restored = RolloutRecord.from_dict(first.to_dict())

        self.assertEqual(restored, first)
        self.assertEqual(first.checksum, second.checksum)

    def test_rejects_unknown_top_level_field(self) -> None:
        value = make_record().to_dict()
        value["polar_gateway_node"] = "localhost-node-01"

        with self.assertRaisesRegex(ContractValidationError, "opaque_metadata"):
            RolloutRecord.from_dict(value)

    def test_rejects_mask_without_tokens_and_mismatched_mask(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "without token_ids"):
            make_record(token_ids=None, loss_mask=(1,))
        with self.assertRaisesRegex(ContractValidationError, "same length"):
            make_record(loss_mask=(1, 1))

    def test_rejects_both_action_and_loss_mask(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "not both"):
            make_record(action_mask=(0, 1, 1, 1))

    def test_accepts_old_logprobs_for_all_or_trainable_tokens(self) -> None:
        full = make_record(old_logprobs=(-0.4, -0.3, -0.2, -0.1))
        trainable = make_record(old_logprobs=(-0.3, -0.2, -0.1))

        self.assertEqual(len(full.old_logprobs or ()), 4)
        self.assertEqual(len(trainable.old_logprobs or ()), 3)
        with self.assertRaisesRegex(ContractValidationError, "old_logprobs length"):
            make_record(old_logprobs=(-0.1,))

    def test_accepts_response_aligned_logprobs_and_validates_prompt_boundary(self) -> None:
        response_aligned = make_record(
            loss_mask=(0, 1, 0, 1),
            prompt_token_count=1,
            old_logprobs=(-0.3, 0.0, -0.1),
        )

        self.assertEqual(len(response_aligned.old_logprobs or ()), 3)
        with self.assertRaisesRegex(ContractValidationError, "between zero"):
            make_record(prompt_token_count=5)
        with self.assertRaisesRegex(ContractValidationError, "without token_ids"):
            make_record(
                token_ids=None,
                prompt_token_count=0,
                loss_mask=None,
                old_logprobs=None,
            )

    def test_rejects_non_finite_reward_and_reversed_timestamps(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "reward must be finite"):
            make_record(reward=float("nan"))
        with self.assertRaisesRegex(ContractValidationError, "ended_at"):
            make_record(
                started_at="2026-08-10T01:00:00Z",
                ended_at="2026-08-10T00:00:00Z",
            )

    def test_missing_fields_are_capability_absence_not_fabricated_values(self) -> None:
        record = make_record(
            group_id=None,
            policy_version=None,
            token_ids=None,
            prompt_token_count=None,
            loss_mask=None,
            old_logprobs=None,
            reward=None,
            verifier_evidence_ref=None,
            tool_events=None,
        )

        self.assertEqual(capabilities_for_record(record), frozenset())
        self.assertIsNone(record.token_ids)
        self.assertIsNone(record.old_logprobs)
        with self.assertRaises(CapabilityMissingError):
            require_capabilities(
                capabilities_for_record(record),
                {Capability.TOKEN_IDS, Capability.POLICY_VERSION},
                context="test trainer",
            )


if __name__ == "__main__":
    unittest.main()
