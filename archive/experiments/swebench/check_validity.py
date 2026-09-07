"""ARCHIVED: base-validity gate for the old SWE-bench runner.

For each instance, evaluate with an EMPTY agent patch (pristine base_commit +
test_patch only). If the FAIL_TO_PASS tests PASS with no fix, the instance is
INVALID in our no-Docker environment (dependency versions differ from the
official pinned Docker env, so the test passes at base) and must be excluded —
otherwise an empty-patch model would score a false-positive "resolved".

Writes results to swebench/validity/validity.json.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/root/agentic-rl")
from pathlib import Path

import experiments.swebench.run_swebench as R


def main() -> None:
    sel = json.loads(R.SEL.read_text())
    out: dict[str, dict] = {}
    for iid, d in sel.items():
        venv = R.VENS / iid
        if not (venv / "bin" / "python").exists():
            out[iid] = {"skip": "no venv"}
            print(iid, "SKIP no venv", flush=True)
            continue
        # reset the eval worktree to pristine base so the check is clean
        wt = (R.WORKTREES / f"eval-{iid}").resolve()
        if wt.exists():
            R.run(["git", "-C", str(wt), "reset", "--hard", "--quiet", d["base_commit"]], check=False)
            R.run(["git", "-C", str(wt), "clean", "-fdq"], check=False)
        inst_dir = R.ROOT / "validity" / iid
        inst_dir.mkdir(parents=True, exist_ok=True)
        (inst_dir / "agent.patch").write_text("")  # empty patch => base validity
        (inst_dir / "untracked.json").write_text("{}")
        try:
            ev = R.eval_instance(iid, sel, inst_dir)
            ftp = ev.get("ftp", {})
            ptp = ev.get("ptp_sanity", {})
            # FULL validity (no-Docker): target tests must FAIL at base (else the
            # fix isn't needed) AND baseline PASS_TO_PASS must PASS at base (else
            # the env is incompatible and the instance can never resolve).
            ftp_fail_at_base = not all(ftp.values())
            ptp_pass_at_base = all(ptp.values()) and bool(ptp)
            out[iid] = {
                "resolved_empty": ev["resolved"],
                "ftp": ftp, "ptp": ptp,
                "valid": bool(ftp_fail_at_base and ptp_pass_at_base),
            }
        except Exception as exc:  # noqa: BLE001
            out[iid] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        print(iid, out[iid], flush=True)
    (R.ROOT / "validity").mkdir(parents=True, exist_ok=True)
    (R.ROOT / "validity" / "validity.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    valid = [k for k, v in out.items() if v.get("valid")]
    print("\nFULLY-VALID (ftp fails AND ptp passes at base):", len(valid))
    for k in valid:
        print("  ", k)
    print("DONE")


if __name__ == "__main__":
    main()
