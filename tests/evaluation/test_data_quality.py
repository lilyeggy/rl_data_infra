from __future__ import annotations

import unittest

from src.certification import ConsumerProfile, ConsumerVerdict, EligibilityDecision
from src.contracts.execution_identity import ExecutionIdentity
from src.evaluation import build_data_quality_report
from src.producers import ProducerArtifact, ProducerCapability, ProducerExecutionStatus


def _decision(episode_id: str, verdict: ConsumerVerdict, reasons=()):
    return EligibilityDecision(
        episode_id=episode_id,
        episode_checksum="a" * 64,
        profile=ConsumerProfile.ON_POLICY_RL,
        verdict=verdict,
        reasons=tuple(reasons),
        episode_certification_checksum="b" * 64,
        training_eligibility_checksum=None,
    )


class DataQualityReportTest(unittest.TestCase):
    def test_counts_capabilities_duplicates_rejections_and_orphans(self) -> None:
        artifact = ProducerArtifact(
            identity=ExecutionIdentity(
                run_id="run-quality",
                task_id="task-quality",
                episode_id="ep-quality",
                attempt_id=1,
                producer_id="polar",
                producer_version="stable@abc",
            ),
            status=ProducerExecutionStatus.COMPLETED,
            capabilities=frozenset({ProducerCapability.TOKEN_IDS}),
            payload={"trajectory": {}},
        )
        report = build_data_quality_report(
            (artifact, artifact),
            (
                _decision("ep-quality", ConsumerVerdict.ELIGIBLE),
                _decision("ep-orphan", ConsumerVerdict.REJECTED, ("bad verifier",)),
            ),
        )
        self.assertEqual(report.duplicate_artifact_count, 1)
        self.assertEqual(report.orphan_decision_count, 1)
        self.assertEqual(dict(report.capability_counts)["TOKEN_IDS"], 2)
        self.assertEqual(report.rejection_reason_counts, (("bad verifier", 1),))
        self.assertEqual(report.checksum, report.checksum)


if __name__ == "__main__":
    unittest.main()
