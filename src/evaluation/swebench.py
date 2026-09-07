"""Pure fail-closed SWE-style patch and test verdicts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _infra_invalid(instance_id: str, message: str, eval_worktree: str) -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "resolved": False,
        "status": "INFRA_INVALID",
        "error": message,
        "ftp": {},
        "ptp": {},
        "eval_worktree": str(Path(eval_worktree)),
    }


def patch_apply_status(
    instance_id: str,
    agent_applied: bool,
    test_applied: bool,
    eval_wt: str,
) -> dict[str, Any] | None:
    """Treat patch application failures as infrastructure-invalid."""

    if not agent_applied:
        return _infra_invalid(instance_id, "agent patch apply failed", eval_wt)
    if not test_applied:
        return _infra_invalid(instance_id, "test patch apply failed", eval_wt)
    return None


def decide_swebench_result(
    instance_id: str,
    fail_to_pass: Mapping[str, Mapping[str, Any]],
    pass_to_pass: Mapping[str, Mapping[str, Any]],
    eval_worktree: str,
) -> dict[str, Any]:
    """Return RESOLVED, UNRESOLVED, or INFRA_INVALID with full evidence."""

    ftp_infra = [key for key, value in fail_to_pass.items() if value.get("infra_invalid")]
    ptp_infra = [key for key, value in pass_to_pass.items() if value.get("infra_invalid")]
    if ftp_infra or ptp_infra:
        return _infra_invalid(
            instance_id,
            "verifier executor failure/timeout (not reward 0): "
            + ", ".join(ftp_infra + ptp_infra),
            eval_worktree,
        )
    resolved = (
        bool(fail_to_pass)
        and all(value.get("passed") for value in fail_to_pass.values())
        and bool(pass_to_pass)
        and all(value.get("passed") for value in pass_to_pass.values())
    )
    return {
        "instance_id": instance_id,
        "resolved": resolved,
        "status": "RESOLVED" if resolved else "UNRESOLVED",
        "ftp": {key: value.get("passed") for key, value in fail_to_pass.items()},
        "ftp_details": dict(fail_to_pass),
        "ptp": {key: value.get("passed") for key, value in pass_to_pass.items()},
        "ptp_results": dict(pass_to_pass),
        "eval_worktree": eval_worktree,
    }
