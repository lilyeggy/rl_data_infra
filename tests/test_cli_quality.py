from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from src.certification import ConsumerProfile, ConsumerVerdict, EligibilityDecision
from src.cli import _quality_report
from src.contracts.execution_identity import ExecutionIdentity
from src.producers import ProducerArtifact, ProducerExecutionStatus


class QualityReportCliTest(unittest.TestCase):
    def test_persisted_records_can_be_reloaded_and_reported(self) -> None:
        identity = ExecutionIdentity(
            run_id="run-cli-quality",
            task_id="task-cli-quality",
            episode_id="episode-cli-quality",
            attempt_id=1,
            producer_id="polar",
            producer_version="stable@abc",
        )
        artifact = ProducerArtifact(
            identity=identity,
            status=ProducerExecutionStatus.INFRA_INVALID,
            capabilities=frozenset(),
            payload={"error": "worker crashed"},
            issues=("worker crashed",),
        )
        decision = EligibilityDecision(
            episode_id=identity.episode_id,
            episode_checksum="a" * 64,
            profile=ConsumerProfile.ON_POLICY_RL,
            verdict=ConsumerVerdict.REJECTED,
            reasons=("RL requires a completed producer execution",),
            episode_certification_checksum="b" * 64,
            training_eligibility_checksum=None,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / "artifacts.jsonl"
            decisions = root / "decisions.jsonl"
            output = root / "quality.json"
            artifacts.write_text(json.dumps(artifact.to_dict()) + "\n", encoding="utf-8")
            decisions.write_text(json.dumps(decision.to_dict()) + "\n", encoding="utf-8")
            with redirect_stdout(StringIO()):
                result = _quality_report(
                    Namespace(
                        artifacts=str(artifacts),
                        decisions=str(decisions),
                        output=str(output),
                    )
                )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result, 0)
            self.assertEqual(payload["producer_status_counts"], {"INFRA_INVALID": 1})
            self.assertEqual(payload["decision_verdict_counts"], {"REJECTED": 1})
            self.assertEqual(len(payload["checksum"]), 64)


if __name__ == "__main__":
    unittest.main()
