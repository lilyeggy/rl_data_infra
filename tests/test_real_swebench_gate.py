"""real_swebench packaging: only verifier-certified episodes become SFT candidates."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from examples.legacy_scenarios.real_swebench import build_package

SESSION = 1787086024000


class RealSwebenchGateTest(unittest.TestCase):
    def _write_instance(self, root: Path, instance_id: str, *, resolved: bool) -> None:
        inst = root / instance_id
        inst.mkdir(parents=True, exist_ok=True)
        records = [
            {"type": "session", "id": f"session-{instance_id}", "timestamp": SESSION},
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "fixed the bug"}],
                    "timestamp": SESSION + 10,
                    "stopReason": "stop",
                },
            },
        ]
        (inst / "trace.ndjson").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        )
        (inst / "eval.json").write_text(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "resolved": resolved,
                    "status": "RESOLVED" if resolved else "UNRESOLVED",
                    "ftp": {"t1": resolved},
                    "ptp": {"p1": True},
                }
            )
        )

    def test_only_certified_episodes_become_sft_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            self._write_instance(results, "pallets__flask-1", resolved=True)
            self._write_instance(results, "psf__requests-1", resolved=False)
            dataset = {
                "pallets__flask-1": {
                    "repo": "pallets/flask",
                    "base_commit": "abc123",
                    "FAIL_TO_PASS": ["t1"],
                    "PASS_TO_PASS": ["p1"],
                },
                "psf__requests-1": {
                    "repo": "psf/requests",
                    "base_commit": "def456",
                    "FAIL_TO_PASS": ["t1"],
                    "PASS_TO_PASS": ["p1"],
                },
            }
            dataset_file = root / "dataset.json"
            dataset_file.write_text(json.dumps(dataset))

            out = root / "out"
            # The run_* assembly follows the exact same path as production.
            try:
                summary = build_package(
                    results, out, dataset_file=dataset_file
                )
            except RuntimeError as exc:
                self.fail(f"build_package failed on synthetic fixtures: {exc}")

            candidates = json.loads((out / "training-candidates.json").read_text())
            certs = json.loads((out / "episode-certifications.json").read_text())
            cert_by_ep = {c["episode_id"]: c for c in certs}

            sft_ids = {
                c["episode_id"]
                for c in candidates
                if c["sft_candidate"]
            }
            # every SFT candidate came from a verifier-certified episode
            for episode_id in sft_ids:
                self.assertEqual(
                    cert_by_ep[episode_id]["certification_status"], "CERTIFIED"
                )
            # the unresolved (verifier FAILED) run is NOT an SFT candidate
            self.assertNotIn("episode-swebench-psf__requests-1", sft_ids)
            self.assertEqual(cert_by_ep["episode-swebench-psf__requests-1"][
                "certification_status"], "CERTIFIED")
            self.assertEqual(
                cert_by_ep["episode-swebench-pallets__flask-1"][
                    "certification_status"],
                "CERTIFIED",
            )
            self.assertEqual(summary["resolved"], 1)


if __name__ == "__main__":
    unittest.main()
