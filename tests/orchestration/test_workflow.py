from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.errors import ContractValidationError
from src.orchestration import (
    CyclePhase,
    GpuOwner,
    PlannedCommand,
    SingleGpuCycleState,
    SingleGpuCycleStore,
    SingleGpuDryRunPlan,
)


def _state() -> SingleGpuCycleState:
    return SingleGpuCycleState.create(
        cycle_id="cycle-persisted", baseline_policy_fingerprint="a" * 64
    )


def _plan() -> SingleGpuDryRunPlan:
    return SingleGpuDryRunPlan(
        commands=(
            PlannedCommand(
                phase=CyclePhase.ROLLOUT_RUNNING,
                argv=("polar", "serve_rollout", "-c", "topology.yaml"),
            ),
            PlannedCommand(
                phase=CyclePhase.TRAINING_RUNNING,
                argv=("python", "-m", "slime.train", "--config", "train.yaml"),
            ),
            PlannedCommand(
                phase=CyclePhase.EVALUATION_RUNNING,
                argv=("python", "evaluate.py", "--manifest", "holdout.json"),
            ),
        )
    )


class SingleGpuWorkflowTest(unittest.TestCase):
    def test_store_is_resumable_and_uses_checksum_compare_and_swap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SingleGpuCycleStore(Path(temporary))
            initial = _state()
            path = store.create(initial)
            self.assertTrue(path.is_file())
            self.assertEqual(store.load(initial.cycle_id), initial)

            running = store.advance(
                initial.cycle_id,
                expected_state_checksum=initial.checksum,
                next_phase=CyclePhase.ROLLOUT_RUNNING,
            )
            self.assertEqual(running.gpu_owner, GpuOwner.SERVING)
            self.assertEqual(store.load(initial.cycle_id), running)
            with self.assertRaises(ContractValidationError):
                store.advance(
                    initial.cycle_id,
                    expected_state_checksum=initial.checksum,
                    next_phase=CyclePhase.ROLLOUT_FROZEN,
                    evidence_checksum="b" * 64,
                )

    def test_dry_run_returns_argv_without_executing_commands(self) -> None:
        plan = _plan()
        rollout = plan.next_command(_state())
        self.assertIsNotNone(rollout)
        assert rollout is not None
        self.assertEqual(rollout.gpu_owner, GpuOwner.SERVING)
        self.assertEqual(rollout.argv[0], "polar")

        state = _state().advance(CyclePhase.ROLLOUT_RUNNING)
        state = state.advance(CyclePhase.ROLLOUT_FROZEN, evidence_checksum="b" * 64)
        self.assertIsNone(plan.next_command(state))
        state = state.advance(CyclePhase.DATASET_CERTIFIED, evidence_checksum="c" * 64)
        training = plan.next_command(state)
        self.assertIsNotNone(training)
        assert training is not None
        self.assertEqual(training.gpu_owner, GpuOwner.TRAINING)

    def test_plan_requires_all_three_resource_owning_phases(self) -> None:
        with self.assertRaises(ContractValidationError):
            SingleGpuDryRunPlan(
                commands=(
                    PlannedCommand(
                        phase=CyclePhase.ROLLOUT_RUNNING,
                        argv=("polar", "serve_rollout"),
                    ),
                )
            )

    def test_truncated_state_fails_closed_and_keeps_original_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SingleGpuCycleStore(Path(temporary))
            path = store.create(_state())
            path.write_text('{"schema_version":', encoding="utf-8")
            with self.assertRaisesRegex(ContractValidationError, "unreadable"):
                store.load("cycle-persisted")

    def test_empty_cycle_identity_is_rejected_before_lock_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = SingleGpuCycleStore(Path(temporary))
            with self.assertRaises(ContractValidationError):
                store.advance(
                    "",
                    expected_state_checksum="a" * 64,
                    next_phase=CyclePhase.ROLLOUT_RUNNING,
                )
            self.assertEqual(tuple(Path(temporary).iterdir()), ())


if __name__ == "__main__":
    unittest.main()
