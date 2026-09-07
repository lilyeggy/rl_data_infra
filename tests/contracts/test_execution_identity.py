from __future__ import annotations

import unittest

from src.contracts.execution_identity import ExecutionIdentity
from src.errors import ContractValidationError


class ExecutionIdentityTest(unittest.TestCase):
    def test_roundtrip_and_checksum_are_stable(self) -> None:
        identity = ExecutionIdentity(
            run_id="run-1",
            task_id="task-1",
            episode_id="episode-1",
            attempt_id=1,
            producer_id="pi-direct",
            producer_version="pi-direct/v1",
            group_id="group-1",
            policy_fingerprint="a" * 64,
            sampling_fingerprint="b" * 64,
        )
        restored = ExecutionIdentity.from_dict(identity.to_dict())
        self.assertEqual(restored, identity)
        self.assertEqual(restored.checksum, identity.checksum)

    def test_attempt_and_unknown_fields_fail_closed(self) -> None:
        with self.assertRaises(ContractValidationError):
            ExecutionIdentity(
                run_id="run-1",
                task_id="task-1",
                episode_id="episode-1",
                attempt_id=0,
                producer_id="pi-direct",
                producer_version="pi-direct/v1",
            )
        value = {
            "schema_version": "execution-identity/v1",
            "run_id": "run-1",
            "task_id": "task-1",
            "episode_id": "episode-1",
            "attempt_id": 1,
            "producer_id": "pi-direct",
            "producer_version": "pi-direct/v1",
            "group_id": None,
            "policy_fingerprint": None,
            "sampling_fingerprint": None,
            "unexpected": True,
        }
        with self.assertRaises(ContractValidationError):
            ExecutionIdentity.from_dict(value)


if __name__ == "__main__":
    unittest.main()
