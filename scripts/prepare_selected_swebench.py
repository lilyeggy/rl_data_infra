#!/usr/bin/env python3
"""Prepare isolated, cache-efficient host worktrees for selected SWE-bench tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from src.contracts._json import canonical_json_bytes


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _workspace_name(instance_id: str) -> str:
    repository, issue = instance_id.rsplit("-", 1)
    owner, name = repository.split("__", 1)
    prefix = name if owner in {"pallets", "pytest-dev", "pylint-dev", "sympy"} else f"{owner}-{name}"
    return f"{prefix}-{issue}"


def _prepare_repository(repo: str, cache_root: Path) -> Path:
    mirror = cache_root / f"{repo.replace('/', '--')}.git"
    if not mirror.exists():
        _run(["git", "init", "--bare", str(mirror)])
        _run([
            "git", "--git-dir", str(mirror), "remote", "add", "origin",
            f"https://github.com/{repo}.git",
        ])
    if not (mirror / "HEAD").is_file():
        raise ValueError(f"invalid repository mirror: {mirror}")
    return mirror


def prepare(
    *, selected: dict[str, Any], instances: list[str], workspace_root: Path,
    venv_root: Path, cache_root: Path, python: str,
) -> list[dict[str, Any]]:
    cache_root.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    venv_root.mkdir(parents=True, exist_ok=True)
    manifest_root = workspace_root / "_prep-manifests"
    manifest_root.mkdir(exist_ok=True)
    results = []
    for instance_id in instances:
        task = selected.get(instance_id)
        if not isinstance(task, dict):
            raise ValueError(f"unknown selected instance: {instance_id}")
        repo = task["repo"]
        commit = task["base_commit"]
        name = _workspace_name(instance_id)
        workspace = workspace_root / name
        venv = venv_root / name
        mirror = _prepare_repository(repo, cache_root)
        commit_exists = subprocess.run(
            ["git", "--git-dir", str(mirror), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=False,
        ).returncode == 0
        if not commit_exists:
            _run([
                "git", "--git-dir", str(mirror), "fetch", "--depth=1", "origin", commit,
            ])
        if workspace.exists():
            if not (workspace / ".git").exists():
                raise ValueError(f"refusing to reuse non-git workspace: {workspace}")
            if subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=workspace, text=True
            ).strip():
                raise ValueError(f"refusing to reset dirty workspace: {workspace}")
            _run(["git", "checkout", "--detach", commit], cwd=workspace)
        else:
            _run([
                "git", "--git-dir", str(mirror), "worktree", "add", "--detach",
                str(workspace), commit,
            ])
        if not (venv / "bin" / "python").is_file():
            if venv.exists():
                raise ValueError(f"refusing to reuse incomplete venv: {venv}")
            _run([python, "-m", "venv", str(venv)])
            _run([
                str(venv / "bin" / "python"), "-m", "pip", "install",
                "pytest==7.4.4", "mpmath<1.4",
            ])
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=workspace, text=True
        ).strip()
        if head != commit:
            raise ValueError(f"workspace commit mismatch for {instance_id}: {head}")
        manifest = {
            "schema_version": "selected-swebench-host-preparation/v1",
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": commit,
            "workspace": str(workspace.resolve()),
            # Preserve the venv entrypoint rather than resolving its symlink
            # to the base interpreter; pyvenv.cfg is part of the runtime identity.
            "venv_python": str((venv / "bin" / "python").absolute()),
            "mirror": str(mirror.resolve()),
            "selected_task_sha256": hashlib.sha256(
                canonical_json_bytes(task)
            ).hexdigest(),
        }
        (manifest_root / f"{name}.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        results.append(manifest)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--venv-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--instances", nargs="+", required=True)
    args = parser.parse_args()
    selected = json.loads(args.selected.read_text())
    results = prepare(
        selected=selected, instances=args.instances,
        workspace_root=args.workspace_root.resolve(), venv_root=args.venv_root.resolve(),
        cache_root=args.cache_root.resolve(), python=args.python,
    )
    print(json.dumps({"prepared": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
