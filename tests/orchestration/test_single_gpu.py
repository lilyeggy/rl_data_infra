from __future__ import annotations

import unittest

from src.errors import ContractValidationError
from src.orchestration import CyclePhase, GpuOwner, SingleGpuCycleState


class SingleGpuCycleStateTest(unittest.TestCase):
    def test_complete_cycle_enforces_resource_handoff_and_evidence(self) -> None:
        state = SingleGpuCycleState.create(
            cycle_id="cycle-1", baseline_policy_fingerprint="a" * 64
        )
        self.assertEqual(state.gpu_owner, GpuOwner.NONE)
        state = state.advance(CyclePhase.ROLLOUT_RUNNING)
        self.assertEqual(state.gpu_owner, GpuOwner.SERVING)
        state = state.advance(CyclePhase.ROLLOUT_FROZEN, evidence_checksum="b" * 64)
        self.assertEqual(state.gpu_owner, GpuOwner.NONE)
        state = state.advance(CyclePhase.DATASET_CERTIFIED, evidence_checksum="c" * 64)
        state = state.advance(CyclePhase.TRAINING_RUNNING)
        self.assertEqual(state.gpu_owner, GpuOwner.TRAINING)
        state = state.advance(
            CyclePhase.CHECKPOINT_READY,
            evidence_checksum="d" * 64,
            candidate_policy_fingerprint="e" * 64,
        )
        state = state.advance(CyclePhase.EVALUATION_RUNNING)
        self.assertEqual(state.gpu_owner, GpuOwner.SERVING)
        state = state.advance(CyclePhase.COMPLETE, evidence_checksum="f" * 64)
        self.assertEqual(state.gpu_owner, GpuOwner.NONE)
        self.assertEqual(SingleGpuCycleState.from_dict(state.to_dict()), state)

    def test_cannot_train_before_rollout_and_dataset_are_frozen(self) -> None:
        state = SingleGpuCycleState.create(
            cycle_id="cycle-order", baseline_policy_fingerprint="a" * 64
        )
        with self.assertRaises(ContractValidationError):
            state.advance(CyclePhase.TRAINING_RUNNING)

    def test_checkpoint_requires_new_policy_and_checksum(self) -> None:
        state = SingleGpuCycleState.create(
            cycle_id="cycle-checkpoint", baseline_policy_fingerprint="a" * 64
        )
        state = state.advance(CyclePhase.ROLLOUT_RUNNING)
        state = state.advance(CyclePhase.ROLLOUT_FROZEN, evidence_checksum="b" * 64)
        state = state.advance(CyclePhase.DATASET_CERTIFIED, evidence_checksum="c" * 64)
        state = state.advance(CyclePhase.TRAINING_RUNNING)
        with self.assertRaises(ContractValidationError):
            state.advance(
                CyclePhase.CHECKPOINT_READY,
                evidence_checksum="d" * 64,
                candidate_policy_fingerprint="a" * 64,
            )

    def test_failure_is_terminal_and_requires_report(self) -> None:
        state = SingleGpuCycleState.create(
            cycle_id="cycle-fail", baseline_policy_fingerprint="a" * 64
        )
        failed = state.advance(CyclePhase.FAILED, evidence_checksum="f" * 64)
        with self.assertRaises(ContractValidationError):
            failed.advance(CyclePhase.ROLLOUT_RUNNING)


if __name__ == "__main__":
    unittest.main()
