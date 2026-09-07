from __future__ import annotations

import unittest

from src.contracts.execution_bundle import ExecutionBundle
from src.contracts.execution_identity import ExecutionIdentity
from src.errors import ContractValidationError


def _identity() -> ExecutionIdentity:
    return ExecutionIdentity(
        run_id="run-bundle",
        task_id="task-bundle",
        episode_id="episode-bundle",
        attempt_id=1,
        producer_id="polar-live",
        producer_version="polar-live/v1",
        policy_fingerprint="a" * 64,
        sampling_fingerprint="b" * 64,
    )


class ExecutionBundleTest(unittest.TestCase):
    def test_roundtrip_preserves_content_addressed_links(self) -> None:
        bundle = ExecutionBundle(
            identity=_identity(),
            episode_checksum="c" * 64,
            source_artifact_checksums=("d" * 64,),
            policy_trace_checksums=("e" * 64,),
            verifier_report_checksum="f" * 64,
        )
        restored = ExecutionBundle.from_dict(bundle.to_dict())
        self.assertEqual(restored, bundle)
        self.assertEqual(restored.checksum, bundle.checksum)

    def test_source_evidence_is_mandatory(self) -> None:
        with self.assertRaises(ContractValidationError):
            ExecutionBundle(
                identity=_identity(),
                episode_checksum="c" * 64,
                source_artifact_checksums=(),
            )

    def test_v1_and_unknown_fields_are_not_silently_migrated(self) -> None:
        value = ExecutionBundle(
            identity=_identity(),
            episode_checksum="c" * 64,
            source_artifact_checksums=("d" * 64,),
        ).to_dict()
        value["schema_version"] = "execution-bundle/v1"
        with self.assertRaises(ContractValidationError):
            ExecutionBundle.from_dict(value)
        value["schema_version"] = "execution-bundle/v2"
        value["certification_checksums"] = []
        with self.assertRaises(ContractValidationError):
            ExecutionBundle.from_dict(value)


if __name__ == "__main__":
    unittest.main()
