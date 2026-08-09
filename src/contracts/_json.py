"""Deterministic JSON helpers used by the versioned contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from src.errors import ContractValidationError


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def freeze_json(value: Any, path: str = "$") -> Any:
    """Validate and recursively freeze a JSON-compatible value."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractValidationError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        if any(not isinstance(key, str) for key in value):
            raise ContractValidationError(f"{path} contains a non-string object key")
        for key in sorted(value):
            frozen[key] = freeze_json(value[key], f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(freeze_json(item, f"{path}[{index}]") for index, item in enumerate(value))
    raise ContractValidationError(
        f"{path} contains non-JSON value of type {type(value).__name__}"
    )


def thaw_json(value: Any) -> Any:
    """Return ordinary dict/list containers suitable for json.dumps."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        thaw_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def validate_sha256(value: str | None, field_name: str) -> None:
    if value is not None and not SHA256_RE.fullmatch(value):
        raise ContractValidationError(f"{field_name} must be a lowercase SHA256 hex digest")
