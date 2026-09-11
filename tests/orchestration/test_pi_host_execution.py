"""Contract tests for the host execution orchestrator's Pi model config.

The generated Pi config decides how many tokens a single rollout generation
may ask for. It has to follow the caller's budget rather than a constant: the
old hardcoded 8192 exceeds the rollout's response window and the engine's
context on this deployment, so Pi was asking for generations that could never
be served.
"""

from __future__ import annotations

import unittest

from src.orchestration.pi_host_execution import _models_config


def _model_entry(config: dict) -> dict:
    return config["providers"]["local-qwen-proxy"]["models"][0]


class ModelsConfigTest(unittest.TestCase):
    def test_default_keeps_the_historical_cap(self) -> None:
        entry = _model_entry(_models_config(base_url="http://127.0.0.1:1", model="m"))
        self.assertEqual(entry["maxTokens"], 8192)

    def test_caller_cap_is_applied(self) -> None:
        entry = _model_entry(
            _models_config(base_url="http://127.0.0.1:1", model="m", max_tokens=1024)
        )
        self.assertEqual(entry["maxTokens"], 1024)
        self.assertEqual(entry["id"], "m")


if __name__ == "__main__":
    unittest.main()
