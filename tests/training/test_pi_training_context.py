from __future__ import annotations

import json

import pytest

from scripts.capture_pi_training_context import capture
from scripts.build_sft_training_package import (
    MESSAGE_CHAR_CAP,
    _choose_gradient_accumulation,
    _qwen_runtime_messages,
    _turn_examples,
)
from src.contracts._json import sha256_json


def _write_evidence(path, *, cwd: str, tool_name: str = "read") -> None:
    row = {
        "request": {
            "messages": [
                {"role": "system", "content": f"Pi prompt\nCurrent working directory: {cwd}"},
                {"role": "user", "content": "fix it"},
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
        }
    }
    path.write_text(json.dumps(row) + "\n")


def test_capture_preserves_task_system_prompt_and_common_tools(tmp_path) -> None:
    one = tmp_path / "one.jsonl"
    two = tmp_path / "two.jsonl"
    _write_evidence(one, cwd="/work/one")
    _write_evidence(two, cwd="/work/two")
    result = capture([f"task-one={one}", f"task-two={two}"])
    assert result["tools_checksum"] == sha256_json(result["tools"])
    assert result["task_contexts"]["task-one"]["system_message"]["content"].endswith("/work/one")
    assert result["task_contexts"]["task-two"]["system_message"]["content"].endswith("/work/two")


def test_capture_rejects_tool_contract_drift(tmp_path) -> None:
    one = tmp_path / "one.jsonl"
    two = tmp_path / "two.jsonl"
    _write_evidence(one, cwd="/work/one")
    _write_evidence(two, cwd="/work/two", tool_name="bash")
    with pytest.raises(ValueError, match="tool contract differs"):
        capture([f"task-one={one}", f"task-two={two}"])


def test_qwen_runtime_messages_normalize_action_targets() -> None:
    messages = _qwen_runtime_messages([
        {"role": "user", "content": "x" * (MESSAGE_CHAR_CAP + 10)},
        {
            "role": "assistant",
            "content": "lengthy plan that must not hide the action",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "read", "arguments": '{"path":"a.py"}'},
            }],
        },
    ])
    assert len(messages[0]["content"]) > MESSAGE_CHAR_CAP
    assert "+10 chars truncated" in messages[0]["content"]
    assert messages[1]["content"] is None
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"path": "a.py"}


def test_qwen_runtime_messages_reject_invalid_tool_arguments() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        _qwen_runtime_messages([{
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "read", "arguments": "{"}}],
        }])


def test_paired_tool_error_action_is_retained_as_target() -> None:
    row = {
        "episode_id": "episode-1",
        "episode_checksum": "a" * 64,
        "task_id": "task-1",
        "messages": [
            {"role": "user", "content": "inspect"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "grep", "arguments": '{"pattern":"missing"}'},
                }],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "no matches"},
            {"role": "assistant", "content": "recovered"},
        ],
        "steps": [{
            "tool_call_id": "call-1",
            "result": "no matches",
            "result_status": "ERROR",
        }],
    }
    runtime = {
        "system_message": {"role": "system", "content": "Pi"},
        "tools": [{
            "type": "function",
            "function": {"name": "grep", "parameters": {"type": "object"}},
        }],
    }
    turns = _turn_examples(row, {"quality_score": 0.9}, runtime)
    assert [item["target_message_index"] for item in turns] == [2, 4]


def test_gradient_accumulation_preserves_exact_epochs() -> None:
    assert _choose_gradient_accumulation(105) == 3
    assert _choose_gradient_accumulation(235) == 5
    assert _choose_gradient_accumulation(7) == 1
