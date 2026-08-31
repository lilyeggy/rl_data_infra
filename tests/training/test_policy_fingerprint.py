"""Strict PolicyFingerprint tests."""

from __future__ import annotations

import unittest

from src.training.policy_fingerprint import PolicyFingerprint


def _fp(**overrides):
    base = {
        "provider": "openai",
        "model_id": "gpt-5.6",
        "base_model_revision": "rev-1",
        "adapter_revision": "adapter-1",
        "tokenizer_revision": "tok-1",
        "chat_template_checksum": "chat-a",
        "tool_schema_checksum": "tools-a",
        "sampling_config": {"temperature": 0.0},
        "temperature": 0.0,
        "top_p": 1.0,
        "policy_generation": "gen-1",
    }
    base.update(overrides)
    return PolicyFingerprint(**base)


class PolicyFingerprintTest(unittest.TestCase):
    def test_complete_policy_is_complete(self) -> None:
        self.assertTrue(_fp().is_complete())

    def test_missing_adapter_revision_is_incomplete(self) -> None:
        fp = _fp(adapter_revision=None)
        self.assertFalse(fp.is_complete())
        self.assertIn("adapter_revision", fp.missing_fields())

    def test_same_model_different_adapter_rejected(self) -> None:
        a = _fp()
        b = _fp(adapter_revision="adapter-2")
        # same model_id, different adapter => NOT on-policy equal
        self.assertFalse(a.matches(b))

    def test_different_temperature_rejected(self) -> None:
        self.assertFalse(_fp().matches(_fp(temperature=0.8)))

    def test_different_top_p_rejected(self) -> None:
        self.assertFalse(_fp().matches(_fp(top_p=0.9)))

    def test_exact_match_accepted(self) -> None:
        self.assertTrue(_fp().matches(_fp()))

    def test_incomplete_never_matches(self) -> None:
        a = _fp(temperature=None)
        b = _fp()
        self.assertFalse(a.is_complete())
        self.assertFalse(a.matches(b))
        # even full-equal apart from one missing field fails
        self.assertFalse(_fp().matches(_fp(chat_template_checksum=None)))

    def test_serialization_roundtrip(self) -> None:
        fp = _fp()
        self.assertEqual(PolicyFingerprint.from_dict(fp.to_dict()), fp)


if __name__ == "__main__":
    unittest.main()
