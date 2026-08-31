"""Bounded workspace snapshots and content-addressed change evidence."""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass
from pathlib import Path

from src.capture.event_writer import ArtifactStore
from src.contracts._json import canonical_json_bytes, sha256_bytes
from src.contracts.artifacts import ArtifactRef

WORKSPACE_EVIDENCE_VERSION = "workspace-evidence/v1"


@dataclass(frozen=True, slots=True)
class WorkspaceFile:
    path: str
    sha256: str
    size_bytes: int
    content: bytes | None


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    files: tuple[WorkspaceFile, ...]
    complete: bool
    issues: tuple[str, ...]


def snapshot_workspace(
    root: str | Path,
    *,
    max_files: int = 10_000,
    max_captured_bytes: int = 64 * 1024 * 1024,
) -> WorkspaceSnapshot:
    """Capture regular files without following symlinks or reading Git internals."""

    workspace = Path(root).resolve()
    files: list[WorkspaceFile] = []
    issues: list[str] = []
    captured_bytes = 0
    candidates = sorted(
        (
            path
            for path in workspace.rglob("*")
            if ".git" not in path.relative_to(workspace).parts
            and not path.is_symlink()
            and path.is_file()
        ),
        key=lambda path: path.relative_to(workspace).as_posix(),
    )
    complete = len(candidates) <= max_files
    if not complete:
        issues.append(f"workspace contains more than {max_files} regular files")
    for path in candidates[:max_files]:
        relative = path.relative_to(workspace).as_posix()
        try:
            size_bytes = path.stat().st_size
            retain = captured_bytes + size_bytes <= max_captured_bytes
            if retain:
                content = path.read_bytes()
                digest = sha256_bytes(content)
            else:
                content = None
                digest_builder = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest_builder.update(chunk)
                digest = digest_builder.hexdigest()
        except OSError as exc:
            complete = False
            issues.append(f"could not read {relative}: {exc}")
            continue
        if not retain:
            complete = False
            issues.append(f"content capture budget exceeded at {relative}")
        else:
            captured_bytes += size_bytes
        files.append(
            WorkspaceFile(
                path=relative,
                sha256=digest,
                size_bytes=size_bytes,
                content=content,
            )
        )
    return WorkspaceSnapshot(tuple(files), complete, tuple(issues))


def store_workspace_change_evidence(
    before: WorkspaceSnapshot,
    after: WorkspaceSnapshot,
    *,
    store: ArtifactStore,
    created_at: str,
) -> tuple[ArtifactRef, ...]:
    """Persist a machine-readable diff and a best-effort unified text patch."""

    before_index = {item.path: item for item in before.files}
    after_index = {item.path: item for item in after.files}
    added = sorted(set(after_index) - set(before_index))
    deleted = sorted(set(before_index) - set(after_index))
    modified = sorted(
        path
        for path in set(before_index) & set(after_index)
        if before_index[path].sha256 != after_index[path].sha256
    )
    report = {
        "schema_version": WORKSPACE_EVIDENCE_VERSION,
        "complete": before.complete and after.complete,
        "issues": [*before.issues, *after.issues],
        "added": [_file_description(after_index[path]) for path in added],
        "modified": [
            {
                "path": path,
                "before_sha256": before_index[path].sha256,
                "after_sha256": after_index[path].sha256,
                "before_size_bytes": before_index[path].size_bytes,
                "after_size_bytes": after_index[path].size_bytes,
            }
            for path in modified
        ],
        "deleted": [_file_description(before_index[path]) for path in deleted],
    }
    refs = [
        store.put(
            canonical_json_bytes(report),
            kind="workspace-diff",
            media_type="application/json",
            created_at=created_at,
        )
    ]
    patch = _unified_patch(before_index, after_index, added, modified, deleted)
    if patch:
        refs.append(
            store.put(
                patch.encode(),
                kind="workspace-patch",
                media_type="text/x-diff; charset=utf-8",
                created_at=created_at,
            )
        )
    return tuple(refs)


def _file_description(value: WorkspaceFile) -> dict[str, object]:
    return {
        "path": value.path,
        "sha256": value.sha256,
        "size_bytes": value.size_bytes,
    }


def _text(value: WorkspaceFile | None) -> list[str] | None:
    if value is None or value.content is None or b"\0" in value.content:
        return None
    try:
        return value.content.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError:
        return None


def _unified_patch(
    before: dict[str, WorkspaceFile],
    after: dict[str, WorkspaceFile],
    added: list[str],
    modified: list[str],
    deleted: list[str],
) -> str:
    chunks: list[str] = []
    for path in [*added, *modified, *deleted]:
        before_text = _text(before.get(path))
        after_text = _text(after.get(path))
        if path in added:
            before_text = []
        if path in deleted:
            after_text = []
        if before_text is None or after_text is None:
            continue
        chunks.extend(
            difflib.unified_diff(
                before_text,
                after_text,
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
    return "".join(chunks)
