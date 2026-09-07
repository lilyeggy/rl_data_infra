#!/usr/bin/env python3
"""ARCHIVED: verify evidence from the retired two-Blackwell Day-1 host."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in an incomplete environment
    yaml = None


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required: install it with `python -m pip install pyyaml`")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return payload


def load_json_with_trailing_metadata(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    payload, end = decoder.raw_decode(raw.lstrip())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload, raw.lstrip()[end:].strip()


def verify_versions(lock: dict[str, Any]) -> list[CheckResult]:
    required = ("polar", "slime", "megatron_lm", "sglang", "mbridge")
    software = lock.get("software", {})
    invalid: list[str] = []
    for name in required:
        commit = software.get(name, {}).get("commit")
        if not isinstance(commit, str) or not FULL_SHA.fullmatch(commit):
            invalid.append(name)

    results = [
        CheckResult(
            "source_commits",
            FAIL if invalid else PASS,
            f"invalid or missing full commit: {', '.join(invalid)}" if invalid else "all required commits are full 40-character SHAs",
        )
    ]

    model_revision = lock.get("model", {}).get("revision")
    results.append(
        CheckResult(
            "model_revision",
            PASS if isinstance(model_revision, str) and FULL_SHA.fullmatch(model_revision) else FAIL,
            str(model_revision or "missing"),
        )
    )

    calculator_digest = lock.get("runtime_images", {}).get("calculator", {}).get("digest")
    results.append(
        CheckResult(
            "calculator_image_identity",
            PASS if isinstance(calculator_digest, str) and SHA256.fullmatch(calculator_digest) else FAIL,
            str(calculator_digest or "missing"),
        )
    )

    dirty = [entry for entry in software.values() if isinstance(entry, dict) and entry.get("dirty") is True]
    reproducible_patch = False
    for entry in dirty:
        for patch in entry.get("patches", []):
            if not isinstance(patch, dict):
                continue
            path = patch.get("path")
            checksum = patch.get("sha256")
            if isinstance(path, str) and isinstance(checksum, str) and SHA256.fullmatch(checksum):
                reproducible_patch = True
    if dirty:
        results.append(
            CheckResult(
                "dirty_patch_artifacts",
                PASS if reproducible_patch else WARN,
                "dirty source has a versioned patch path and checksum" if reproducible_patch else "dirty source is recorded, but no versioned patch path + sha256 pair was found",
            )
        )
    return results


def verify_required_artifacts(artifact_dir: Path) -> CheckResult:
    required = (
        "host-preflight.txt",
        "gpu-topology.txt",
        "source-versions.txt",
        "python-packages.txt",
        "container-images.txt",
    )
    missing = [name for name in required if not (artifact_dir / name).is_file()]
    return CheckResult(
        "environment_artifacts",
        FAIL if missing else PASS,
        f"missing: {', '.join(missing)}" if missing else "all required lightweight evidence files exist",
    )


def verify_artifact_checksums(lock: dict[str, Any], artifact_dir: Path) -> CheckResult:
    checksums = lock.get("evidence", {}).get("checksums", {})
    if not isinstance(checksums, dict) or not checksums:
        return CheckResult("artifact_checksums", FAIL, "evidence.checksums is empty or missing")

    problems: list[str] = []
    for name, expected in checksums.items():
        path = artifact_dir / str(name)
        if not path.is_file():
            problems.append(f"{name}: missing")
            continue
        actual = sha256_file(path)
        if actual != expected:
            problems.append(f"{name}: expected {expected}, got {actual}")
    return CheckResult(
        "artifact_checksums",
        FAIL if problems else PASS,
        "; ".join(problems) if problems else f"verified {len(checksums)} checksums",
    )


def verify_megatron_smoke(path: Path, expected_sequence_length: int) -> CheckResult:
    name = f"megatron_tp2_{expected_sequence_length}"
    if not path.is_file():
        return CheckResult(name, FAIL, f"missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return CheckResult(name, FAIL, f"cannot parse: {error}")

    problems: list[str] = []
    if payload.get("seq_length") != expected_sequence_length:
        problems.append(f"seq_length={payload.get('seq_length')}")
    if payload.get("tensor_model_parallel_size") != 2:
        problems.append(f"tensor_model_parallel_size={payload.get('tensor_model_parallel_size')}")
    if not isinstance(payload.get("grad_norm"), (int, float)) or payload["grad_norm"] <= 0:
        problems.append(f"grad_norm={payload.get('grad_norm')}")

    checksum_a = payload.get("checksum_after_load_A")
    checksum_b = payload.get("checksum_after_step_B")
    checksum_c = payload.get("checksum_after_reload_C")
    if not checksum_a or checksum_a == checksum_b:
        problems.append("optimizer step did not change checksum")
    if not checksum_b or checksum_b != checksum_c:
        problems.append("checkpoint reload did not restore post-step checksum")
    if payload.get("checksum_changed_after_step") is not True:
        problems.append("checksum_changed_after_step is not true")
    if payload.get("checksum_restored_after_reload") is not True:
        problems.append("checksum_restored_after_reload is not true")

    return CheckResult(
        name,
        FAIL if problems else PASS,
        "; ".join(problems) if problems else "non-zero gradient, optimizer update, and checkpoint reload verified",
    )


def _find_output_token_ids(payload: Any) -> list[int] | None:
    output_keys = {"output_token_ids", "completion_token_ids", "generated_token_ids"}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in output_keys and isinstance(value, list) and value and all(isinstance(item, int) for item in value):
                return value
            found = _find_output_token_ids(value)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_output_token_ids(item)
            if found:
                return found
    return None


def verify_sglang_response(path: Path, expected_model_revision: str | None) -> list[CheckResult]:
    if not path.is_file():
        return [CheckResult("sglang_response", FAIL, f"missing: {path}")]
    try:
        payload, trailing = load_json_with_trailing_metadata(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [CheckResult("sglang_response", FAIL, f"cannot parse: {error}")]

    results: list[CheckResult] = []
    model = payload.get("model")
    model_ok = isinstance(expected_model_revision, str) and isinstance(model, str) and expected_model_revision in model
    results.append(CheckResult("sglang_model_revision", PASS if model_ok else FAIL, str(model or "missing")))

    choices = payload.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    logprob_content = choice.get("logprobs", {}).get("content") if isinstance(choice.get("logprobs"), dict) else None
    completion_tokens = payload.get("usage", {}).get("completion_tokens") if isinstance(payload.get("usage"), dict) else None
    logprob_ok = isinstance(logprob_content, list) and bool(logprob_content)
    if logprob_ok and isinstance(completion_tokens, int):
        logprob_ok = len(logprob_content) == completion_tokens
    results.append(
        CheckResult(
            "sglang_logprobs",
            PASS if logprob_ok else FAIL,
            f"completion_tokens={completion_tokens}, logprob_entries={len(logprob_content) if isinstance(logprob_content, list) else 0}",
        )
    )

    finish_reason = choice.get("finish_reason")
    results.append(
        CheckResult(
            "sglang_finish_reason",
            PASS if finish_reason in {"stop", "length", "tool_calls"} else FAIL,
            str(finish_reason or "missing"),
        )
    )

    http_match = re.search(r"http_code:(\d+)", trailing)
    if http_match:
        results.append(
            CheckResult(
                "sglang_http_status",
                PASS if http_match.group(1) == "200" else FAIL,
                http_match.group(1),
            )
        )

    output_token_ids = _find_output_token_ids(payload)
    results.append(
        CheckResult(
            "sglang_numeric_output_token_ids",
            PASS if output_token_ids else WARN,
            f"captured {len(output_token_ids)} numeric output token IDs" if output_token_ids else "missing; do not recreate them by tokenizing generated text",
        )
    )
    return results


def verify_staged_execution(project_config_path: Path) -> CheckResult:
    if not project_config_path.is_file():
        return CheckResult("staged_execution", WARN, f"project config missing: {project_config_path}")
    try:
        config = load_yaml(project_config_path)
    except (OSError, RuntimeError, ValueError) as error:
        return CheckResult("staged_execution", FAIL, f"cannot parse project config: {error}")
    execution_mode = config.get("hardware", {}).get("execution_mode")
    return CheckResult(
        "staged_execution",
        PASS if execution_mode == "staged_shared_two_gpu" else FAIL,
        str(execution_mode or "missing"),
    )


def print_results(results: Iterable[CheckResult]) -> None:
    for result in results:
        print(f"{result.status:<4} {result.name}: {result.detail}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True, help="Path to configs/upstream-lock.yaml")
    parser.add_argument("--artifacts", type=Path, required=True, help="Path to artifacts/day-01")
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--strict-warnings", action="store_true", help="Return failure when any warning remains")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        lock = load_yaml(args.lock)
    except (OSError, RuntimeError, ValueError, yaml.YAMLError if yaml is not None else ValueError) as error:
        print(f"ERROR cannot load lock: {error}", file=sys.stderr)
        return 2

    results: list[CheckResult] = []
    results.extend(verify_versions(lock))
    results.append(verify_required_artifacts(args.artifacts))
    results.append(verify_artifact_checksums(lock, args.artifacts))
    results.append(verify_megatron_smoke(args.artifacts / "backward-smoke-4096-tp2.json", 4096))
    results.append(verify_megatron_smoke(args.artifacts / "backward-smoke-8192-tp2.json", 8192))
    results.extend(verify_sglang_response(args.artifacts / "inference-response.json", lock.get("model", {}).get("revision")))
    results.append(verify_staged_execution(args.project_config))
    print_results(results)

    failed = any(result.status == FAIL for result in results)
    warned = any(result.status == WARN for result in results)
    if failed or (warned and args.strict_warnings):
        print("\nDAY1_STATUS=FAILED")
        return 1
    print("\nDAY1_STATUS=PASS_WITH_NOTES" if warned else "\nDAY1_STATUS=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
