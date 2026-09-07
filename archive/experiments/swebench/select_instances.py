"""ARCHIVED: select instances for the old host-specific SWE-bench runner and emit the
selected_instances.json format that experiments/swebench/run_swebench.py expects.

Usage: python3 select_instances.py <n_total> <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PARQUET = Path(__file__).parent / "swebench_verified.parquet"
# repos already cloned on the server (reuse clones => fast env setup)
CLONED_REPOS = ["sympy/sympy", "pytest-dev/pytest", "psf/requests", "pallets/flask"]
# instances already attempted (don't re-select for fresh data)
DONE = {
    "pallets__flask-5014", "psf__requests-2931", "psf__requests-5414", "psf__requests-6028",
    "pylint-dev__pylint-6903", "pylint-dev__pylint-7277", "pytest-dev__pytest-10051",
    "pytest-dev__pytest-10081", "sympy__sympy-22914", "sympy__sympy-23534",
    "sympy__sympy-23824", "sympy__sympy-23950", "sympy__sympy-24213", "sympy__sympy-24539",
}


def _as_list(v):
    if isinstance(v, (list, tuple)):
        return list(v)
    try:
        return list(json.loads(v))
    except Exception:
        return [str(v)]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/stage0-instances.json"
    df = pd.read_parquet(PARQUET)
    df = df[df["repo"].isin(CLONED_REPOS) & ~df["instance_id"].isin(DONE)]
    # spread across repos, prefer smaller FAIL_TO_PASS (cleaner signal)
    df = df.assign(_nftp=df["FAIL_TO_PASS"].apply(lambda v: len(_as_list(v)))).sort_values("_nftp")
    sel = df.head(n)
    result = {}
    for _, r in sel.iterrows():
        result[r["instance_id"]] = {
            "repo": r["repo"],
            "base_commit": r["base_commit"],
            "version": str(r["version"]),
            "problem_statement": r["problem_statement"],
            "test_patch": r["test_patch"],
            "FAIL_TO_PASS": _as_list(r["FAIL_TO_PASS"]),
            "PASS_TO_PASS": _as_list(r["PASS_TO_PASS"]),
        }
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    from collections import Counter
    print(f"selected {len(result)} -> {out}")
    print("by repo:", dict(Counter(v["repo"] for v in result.values())))


if __name__ == "__main__":
    main()
