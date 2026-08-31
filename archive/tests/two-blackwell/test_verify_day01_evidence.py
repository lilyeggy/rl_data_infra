"""ARCHIVED regression tests for the retired two-Blackwell evidence lock."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_day01_evidence import FAIL, PASS, WARN, verify_megatron_smoke, verify_sglang_response


class VerifyMegatronSmokeTest(unittest.TestCase):
    def write_smoke(self, directory: Path, **overrides):
        payload = {
            "seq_length": 4096,
            "tensor_model_parallel_size": 2,
            "grad_norm": 12.5,
            "checksum_after_load_A": "a",
            "checksum_after_step_B": "b",
            "checksum_after_reload_C": "b",
            "checksum_changed_after_step": True,
            "checksum_restored_after_reload": True,
        }
        payload.update(overrides)
        path = directory / "smoke.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_accepts_real_optimizer_step_and_reload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = verify_megatron_smoke(self.write_smoke(Path(temp_dir)), 4096)
        self.assertEqual(result.status, PASS)

    def test_rejects_checkpoint_reload_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.write_smoke(
                Path(temp_dir),
                checksum_after_reload_C="a",
                checksum_restored_after_reload=False,
            )
            result = verify_megatron_smoke(path, 4096)
        self.assertEqual(result.status, FAIL)
        self.assertIn("checkpoint reload", result.detail)


class VerifySglangResponseTest(unittest.TestCase):
    def write_response(self, directory: Path, include_token_ids: bool):
        payload = {
            "model": "/models/snapshots/0123456789012345678901234567890123456789",
            "choices": [
                {
                    "finish_reason": "stop",
                    "logprobs": {"content": [{"token": "2", "logprob": -0.1}]},
                }
            ],
            "usage": {"completion_tokens": 1},
        }
        if include_token_ids:
            payload["output_token_ids"] = [17]
        path = directory / "response.json"
        path.write_text(json.dumps(payload) + "\nhttp_code:200\n", encoding="utf-8")
        return path

    def test_warns_when_numeric_output_token_ids_are_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results = verify_sglang_response(
                self.write_response(Path(temp_dir), include_token_ids=False),
                "0123456789012345678901234567890123456789",
            )
        token_result = next(result for result in results if result.name == "sglang_numeric_output_token_ids")
        self.assertEqual(token_result.status, WARN)

    def test_accepts_native_numeric_output_token_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results = verify_sglang_response(
                self.write_response(Path(temp_dir), include_token_ids=True),
                "0123456789012345678901234567890123456789",
            )
        token_result = next(result for result in results if result.name == "sglang_numeric_output_token_ids")
        self.assertEqual(token_result.status, PASS)


if __name__ == "__main__":
    unittest.main()
