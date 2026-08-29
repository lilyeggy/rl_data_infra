from __future__ import annotations

import unittest

from src.contracts import (
    Capability,
    ResampleRequest,
    RolloutBatch,
    TrainingReadyBatch,
)
from src.errors import CapabilityMissingError, ContractValidationError
from tests.contract_fixtures import make_record


class BatchContractTest(unittest.TestCase):
    def test_rollout_batch_uses_capability_intersection(self) -> None:
        complete = make_record()
        without_logprobs = make_record(
            trajectory_id="trajectory-002",
            source_record_id="fixture-record-002",
            old_logprobs=None,
        )
        batch = RolloutBatch.from_records(
            (complete, without_logprobs),
            source_adapter="fixture",
            source_adapter_version="v1",
        )

        self.assertNotIn(Capability.OLD_LOGPROBS, batch.capabilities)
        self.assertIn(Capability.TOKEN_IDS, batch.capabilities)
        self.assertEqual(batch.checksum, batch.checksum)

    def test_rollout_batch_rejects_duplicate_trajectory(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "duplicate trajectory_id"):
            RolloutBatch.from_records(
                (make_record(), make_record(source_record_id="another-source")),
                source_adapter="fixture",
                source_adapter_version="v1",
            )

    def test_training_batch_accepts_only_policy_consistent_group(self) -> None:
        records = (
            make_record(),
            make_record(
                trajectory_id="trajectory-002",
                source_record_id="fixture-record-002",
                reward=0.0,
            ),
        )
        batch = TrainingReadyBatch(
            records=records,
            task_id="task-calculator-001",
            group_id="group-001",
            policy_version="policy-v0",
            source_batch_checksum="c" * 64,
            processing_versions=("failure-classifier/v1", "group-builder/v1"),
        )

        self.assertTrue(batch.batch_id.startswith("training-"))
        self.assertEqual(len(batch.records), 2)

    def test_training_batch_rejects_mixed_policy(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "must match"):
            TrainingReadyBatch(
                records=(make_record(policy_version="policy-v1"),),
                task_id="task-calculator-001",
                group_id="group-001",
                policy_version="policy-v0",
                source_batch_checksum="c" * 64,
                processing_versions=("group-builder/v1",),
            )

    def test_training_batch_rejects_missing_training_capability(self) -> None:
        with self.assertRaises(CapabilityMissingError):
            TrainingReadyBatch(
                records=(make_record(token_ids=None, loss_mask=None, old_logprobs=None),),
                task_id="task-calculator-001",
                group_id="group-001",
                policy_version="policy-v0",
                source_batch_checksum="c" * 64,
                processing_versions=("group-builder/v1",),
            )

    def test_resample_request_is_deterministic_and_validated(self) -> None:
        request = ResampleRequest(
            task_id="task-calculator-001",
            group_id="group-001",
            policy_version="policy-v0",
            required_count=1,
            reason="GROUP_INCOMPLETE",
            sampling_constraints={"temperature": 0.8},
            source_hint="polar",
        )

        self.assertTrue(request.request_id.startswith("resample-"))
        self.assertEqual(request.checksum, request.checksum)
        with self.assertRaisesRegex(ContractValidationError, "positive"):
            ResampleRequest(
                task_id="task-calculator-001",
                group_id="group-001",
                policy_version="policy-v0",
                required_count=0,
                reason="GROUP_INCOMPLETE",
            )


if __name__ == "__main__":
    unittest.main()
