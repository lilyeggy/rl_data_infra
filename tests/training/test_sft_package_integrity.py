from __future__ import annotations

import hashlib
import json

import pytest

from scripts.train_qwen_lora_sft import _load_verified_package
from src.contracts._json import sha256_json


def _package(tmp_path):
    config = {"dataset": "train-turns.jsonl", "student_model": "model"}
    tools = [{"type": "function", "function": {"name": "read"}}]
    row = {
        "turn_id": "turn-1",
        "messages": [{"role": "assistant", "content": "ok"}],
        "tools": tools,
        "tools_checksum": sha256_json(tools),
    }
    data = json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    (tmp_path / "lora-config.json").write_text(json.dumps(config))
    (tmp_path / "train-turns.jsonl").write_bytes(data)
    manifest = {
        "lora_config_checksum": sha256_json(config),
        "train_turns_sha256": hashlib.sha256(data).hexdigest(),
        "turn_example_count": 1,
        "tools_checksum": sha256_json(tools),
    }
    (tmp_path / "training-package-manifest.json").write_text(json.dumps(manifest))
    return manifest, config, row


def test_verified_package_loads(tmp_path) -> None:
    expected = _package(tmp_path)
    manifest, config, rows = _load_verified_package(tmp_path)
    assert manifest == expected[0]
    assert config == expected[1]
    assert rows == [expected[2]]


def test_modified_training_data_fails_closed(tmp_path) -> None:
    _package(tmp_path)
    with (tmp_path / "train-turns.jsonl").open("ab") as stream:
        stream.write(b"{}\n")
    with pytest.raises(ValueError, match="dataset checksum"):
        _load_verified_package(tmp_path)


def test_modified_lora_config_fails_closed(tmp_path) -> None:
    _package(tmp_path)
    (tmp_path / "lora-config.json").write_text(json.dumps({"dataset": "other.jsonl"}))
    with pytest.raises(ValueError, match="LoRA config checksum"):
        _load_verified_package(tmp_path)


def test_v6_package_requires_trainable_output_head(tmp_path) -> None:
    manifest, config, _ = _package(tmp_path)
    config["lora"] = {"target_modules": ["q_proj"]}
    manifest.update(
        {
            "schema_version": "teacher-sft-training-package/v6",
            "lora_config_checksum": sha256_json(config),
            "output_head_policy": {"name": "lora-lm-head/v1"},
        }
    )
    (tmp_path / "lora-config.json").write_text(json.dumps(config))
    (tmp_path / "training-package-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="requires lm_head"):
        _load_verified_package(tmp_path)
