from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.watch_learning_loop import required_json_error


class RequiredJsonGateTest(unittest.TestCase):
    def test_accepts_matching_gate_and_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "report.json"
            artifact.write_text(json.dumps({"decision": "REJECT"}))
            stage = {"required_json": {"decision": "ACCEPT"}}
            self.assertIn("expected 'ACCEPT'", required_json_error(stage, artifact) or "")
            artifact.write_text(json.dumps({"decision": "ACCEPT"}))
            self.assertIsNone(required_json_error(stage, artifact))

    def test_invalid_json_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "report.json"
            artifact.write_text("not-json")
            self.assertIsNotNone(
                required_json_error({"required_json": {"decision": "ACCEPT"}}, artifact)
            )


if __name__ == "__main__":
    unittest.main()
