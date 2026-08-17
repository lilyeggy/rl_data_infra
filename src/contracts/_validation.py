"""Small validation helpers shared by execution-data contracts."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Mapping, TypeVar

from src.contracts._json import freeze_json
from src.errors import ContractValidationError


EnumT = TypeVar("EnumT", bound=Enum)


def required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field_name} must be a non-empty string")
    return value


def optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return required_text(value, field_name)


def utc_instant(value: Any, field_name: str) -> datetime:
    required_text(value, field_name)
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ContractValidationError(f"{field_name} must be an ISO-8601 instant") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ContractValidationError(f"{field_name} must use the UTC offset")
    return parsed


def frozen_object(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field_name} must be an object")
    return freeze_json(value, f"$.{field_name}")


def strict_fields(value: Any, allowed: set[str], object_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{object_name} must be an object")
    unknown = sorted(set(value) - allowed)
    missing = sorted(allowed - set(value))
    if unknown:
        raise ContractValidationError(f"unknown {object_name} fields: {unknown}")
    if missing:
        raise ContractValidationError(f"missing {object_name} fields: {missing}")
    return value


def enum_member(value: Any, enum_type: type[EnumT], field_name: str) -> EnumT:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(
            f"{field_name} must be one of {[item.value for item in enum_type]}"
        ) from exc
