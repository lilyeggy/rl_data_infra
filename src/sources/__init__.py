"""Public source adapter API."""

from src.sources.base import AdapterResult, SourceAdapter
from src.sources.jsonl import JSONL_SCHEMA_VERSION, JsonlSourceAdapter, dumps_jsonl
from src.sources.polar import PolarSourceAdapter

__all__ = [
    "AdapterResult",
    "JSONL_SCHEMA_VERSION",
    "JsonlSourceAdapter",
    "PolarSourceAdapter",
    "SourceAdapter",
    "dumps_jsonl",
]
