"""Replaceable raw/manifest object backend boundary for V2.3."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.contracts._json import sha256_bytes
from src.errors import ContractValidationError


STORAGE_BACKEND_VERSION = "storage-backend/v1"


@dataclass(frozen=True, slots=True)
class PartitionKey:
    run_id: str
    episode_id: str

    @property
    def value(self) -> str:
        return f"{self.run_id}/{self.episode_id}"


class StorageBackend(Protocol):
    """Object API required by a production partitioned event store."""

    def put_raw(self, key: str, payload: bytes) -> str:
        """Create/verify an immutable raw object and return its checksum."""
        ...

    def get_raw(self, key: str) -> bytes:
        """Read an immutable raw object."""
        ...

    def put_manifest(self, key: str, payload: bytes) -> str:
        """Atomically replace derived manifest metadata."""
        ...

    def get_manifest(self, key: str) -> bytes | None:
        """Read derived manifest metadata if present."""
        ...


class LocalFilesystemObjectBackend:
    """Filesystem implementation of the replaceable V2.3 object boundary."""

    def __init__(self, root: str | Path, *, durable: bool = True) -> None:
        self.root = Path(root)
        self.raw_root = self.root / "raw"
        self.manifest_root = self.root / "manifests"
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.manifest_root.mkdir(parents=True, exist_ok=True)
        self.durable = durable

    def put_raw(self, key: str, payload: bytes) -> str:
        return self._put_immutable(self.raw_root / self._safe_key(key), payload)

    def get_raw(self, key: str) -> bytes:
        return (self.raw_root / self._safe_key(key)).read_bytes()

    def put_manifest(self, key: str, payload: bytes) -> str:
        path = self.manifest_root / self._safe_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            if self.durable:
                os.fsync(handle.fileno())
        os.replace(temporary, path)
        return sha256_bytes(payload)

    def get_manifest(self, key: str) -> bytes | None:
        path = self.manifest_root / self._safe_key(key)
        return path.read_bytes() if path.exists() else None

    @staticmethod
    def _safe_key(key: str) -> Path:
        if not isinstance(key, str) or not key.strip():
            raise ContractValidationError("storage object key must be non-empty")
        candidate = Path(key)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ContractValidationError("storage object key escapes backend root")
        return candidate

    def _put_immutable(self, path: Path, payload: bytes) -> str:
        if not isinstance(payload, bytes):
            raise TypeError("storage object payload must be bytes")
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256_bytes(payload)
        if path.exists():
            existing = path.read_bytes()
            if existing != payload:
                raise ContractValidationError(
                    f"immutable storage object conflict for {path.as_posix()}"
                )
            return digest
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            if self.durable:
                os.fsync(handle.fileno())
        return digest
