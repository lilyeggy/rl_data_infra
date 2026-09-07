from __future__ import annotations

import unittest

from src.contracts._json import sha256_json
from src.contracts.agent_episode import EpisodeVerifierStatus
from src.contracts.manifests import EvaluatorManifest
from src.contracts.verifier_report import LocalVerifierReport, VerifierExecutionStatus


class LocalVerifierReportTest(unittest.TestCase):
    def test_round_trip_preserves_verifier_attestation(self) -> None:
        report = LocalVerifierReport(
            identity_checksum="a" * 64,
            evaluator=EvaluatorManifest(
                name="pytest",
                revision="pytest-r1",
                config_digest=sha256_json({}),
            ),
            command=("pytest", "-q"),
            execution_status=VerifierExecutionStatus.FAILED,
            verifier_status=EpisodeVerifierStatus.FAILED,
            score=0,
            producer_artifact_checksum="b" * 64,
            output_artifact_checksums=("c" * 64,),
            created_at="2026-08-22T00:00:00Z",
        )
        self.assertEqual(LocalVerifierReport.from_dict(report.to_dict()), report)


if __name__ == "__main__":
    unittest.main()
