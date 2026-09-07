from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from src.capture.pi_adapter import PiRunConfig
from src.capture.pi_runner import run_pi_process


class PiRunnerTest(unittest.TestCase):
    @patch("src.capture.pi_runner.subprocess.run")
    def test_exit_zero_backend_error_is_not_protocol_success(self, mocked) -> None:
        mocked.return_value = subprocess.CompletedProcess(
            args=(),
            returncode=0,
            stdout=(
                '{"type":"session","id":"s"}\n'
                '{"type":"message_end","message":{"role":"assistant",'
                '"stopReason":"error","errorMessage":"500"}}\n'
            ),
            stderr="",
        )
        result = run_pi_process(
            config=PiRunConfig(), prompt="probe", cwd="/tmp", timeout_seconds=1
        )
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.protocol_settled)
        self.assertEqual(result.backend_error_messages, ("500",))
        self.assertFalse(mocked.call_args.kwargs["shell"])
        self.assertIs(mocked.call_args.kwargs["stdin"], subprocess.DEVNULL)

    @patch("src.capture.pi_runner.subprocess.run")
    def test_fixed_model_success_is_protocol_settled(self, mocked) -> None:
        mocked.return_value = subprocess.CompletedProcess(
            args=(),
            returncode=0,
            stdout=(
                '{"type":"session","id":"s"}\n'
                '{"type":"message_end","message":{"role":"assistant",'
                '"stopReason":"stop","content":[{"type":"text","text":"ok"}]}}\n'
            ),
            stderr="",
        )
        result = run_pi_process(config=PiRunConfig(), prompt="probe", cwd="/tmp")
        self.assertTrue(result.protocol_settled)
        self.assertIn("gpt-5.6-luna", result.command)


if __name__ == "__main__":
    unittest.main()
