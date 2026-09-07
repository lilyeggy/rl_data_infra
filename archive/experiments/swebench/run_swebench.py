"""ARCHIVED: old SWE-bench subset runner with host-specific paths.

Per-instance pipeline:
  1. env     : clone repo once (mirror), git worktree at base_commit, venv,
               pip install -e . (test deps per repo)
  2. pi run  : real Pi (opencode-go/deepseek-v4-flash) in the worktree with
               read/bash/write/edit/grep/find/ls/glob tools; NDJSON captured
  3. evaluate: fresh worktree at base_commit + agent diff + test_patch,
               run FAIL_TO_PASS (+ PASS_TO_PASS sanity) with pytest

The runner never modifies the golden patch and never shows the test patch to
the agent. Results are written as JSON for the data-plane assembly step.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "/root/agentic-rl")
from src.capture.pi_adapter import PiRunConfig, dump_pi_ndjson, read_pi_ndjson
from src.capture.pi_runner import run_pi_process

ROOT = Path("/root/rivermind-data/swebench")
CLONES = ROOT / "repos"
WORKTREES = ROOT / "worktrees"
VENS = ROOT / "vens"
RESULTS = ROOT / "results"
SEL = ROOT / "selected_instances.json"

PI_MODEL = "deepseek-v4-flash"
PI_PROVIDER = "opencode-go"
PI_TOOLS = ("read", "bash", "write", "edit", "grep", "find", "ls", "glob")
PI_TIMEOUT = 600.0

# local OpenAI-compatible server (experiments/local_model/openai_server.py)
LOCAL_BASE_URL = "http://127.0.0.1:8000"


def _local_health() -> str | None:
    try:
        with urllib.request.urlopen(f"{LOCAL_BASE_URL}/health", timeout=5) as r:
            return json.loads(r.read()).get("adapter") or ""
    except Exception:
        return None


def ensure_local_server(adapter: str, model_path: str = "", force: bool = False) -> None:
    """(Re)start the local-qwen server with `adapter` (empty = base).

    force=True restarts even if healthy — the server accumulates GPU memory
    fragmentation over long multi-turn SWE contexts and eventually OOMs (500),
    which silently invalidates a run. Restarting per instance gives each a
    clean ~15GB server so the measurement reflects the MODEL, not the server.
    """
    cur = _local_health()
    if not force and cur is not None and (cur or "") == (adapter or ""):
        print(f"  local server up, adapter={cur or 'base'}", flush=True)
        return
    subprocess.run(["pkill", "-9", "-f", "openai_server.py"], check=False)
    time.sleep(3)
    model_env = f"LOCAL_MODEL={model_path} " if model_path else ""
    subprocess.Popen(
        f"cd /root/agentic-rl && PYTHONPATH=/root/agentic-rl "
        f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"{model_env}"
        f"LOCAL_ADAPTER={adapter} PORT=8000 setsid nohup python3 "
        f"experiments/local_model/openai_server.py "
        f"> /root/rivermind-data/qwen-server.log 2>&1 </dev/null &",
        shell=True,
    )
    for _ in range(120):  # up to ~6 min to load 7B
        cur = _local_health()
        if cur is not None and (cur or "") == (adapter or ""):
            print(f"  local server ready, adapter={cur or 'base'}", flush=True)
            return
        time.sleep(3)
    raise RuntimeError("local server failed to start")


MIRRORS = {
    "pallets/flask": [
        "https://gitee.com/mirrors/flask.git",
        "https://gh-proxy.com/https://github.com/pallets/flask.git",
        "https://gitcode.com/gh_mirrors/fl/flask.git",
    ],
    "psf/requests": [
        "https://gitee.com/mirrors/requests.git",
        "https://gh-proxy.com/https://github.com/psf/requests.git",
    ],
    "pytest-dev/pytest": [
        "https://gitee.com/mirrors/pytest.git",
        "https://gh-proxy.com/https://github.com/pytest-dev/pytest.git",
    ],
    "sympy/sympy": [
        "https://gitee.com/mirrors/sympy.git",
        "https://gh-proxy.com/https://github.com/sympy/sympy.git",
    ],
}

# extra test deps per repo (beyond `pip install -e .`)
TEST_DEPS = {
    "psf/requests": ["pytest-mock", "pytest-httpbin", "pytest"],
    "pallets/flask": ["pytest"],
    "pytest-dev/pytest": ["pytest"],
    "sympy/sympy": ["pytest"],
}

PROMPT_TEMPLATE = (
    "We are currently solving the following issue within the {repo} repository. "
    "The repository is checked out at commit {base_commit} in the current "
    "working directory.\n\n"
    "--- BEGIN ISSUE ---\n"
    "{problem}\n"
    "--- END ISSUE ---\n\n"
    "Make the changes necessary to fix the issue directly in the working tree. "
    "You may read files, search, edit, write, and run bash commands. When you "
    "have a fix, validate it by running the relevant existing tests with bash "
    "before finishing. Do not output a summary of the diff; the changes in the "
    "working tree are the answer."
)


def run(cmd, cwd=None, timeout=1200, check=True):
    print("  $", " ".join(str(c) for c in cmd[:6]), "...", flush=True)
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"cmd failed rc={p.returncode}: {p.stderr[-2000:]}")
    return p


def repo_of(instance_id: str) -> str:
    org, rest = instance_id.split("__", 1)
    return f"{org}/{rest.rsplit('-', 1)[0]}"


def clone_repos():
    CLONES.mkdir(parents=True, exist_ok=True)
    for repo, urls in MIRRORS.items():
        dest = CLONES / repo.split("/")[1]
        if (dest / ".git").exists():
            print("clone cache exists:", dest)
            continue
        cloned = False
        for url in urls:
            print("cloning", repo, "via", url, flush=True)
            try:
                run(["git", "clone", "--quiet", url, str(dest)], timeout=1800)
                cloned = True
                break
            except Exception as exc:
                print("  mirror failed:", exc, flush=True)
                if dest.exists():
                    shutil.rmtree(dest, ignore_errors=True)
        if not cloned:
            raise RuntimeError(f"no mirror worked for {repo}")


def worktree_of(instance_id: str) -> Path:
    return WORKTREES / instance_id


def venv_of(instance_id: str) -> Path:
    return VENS / instance_id


def setup_instance(instance_id: str, sel: dict, *, for_eval: bool = False) -> Path:
    """Create venv + editable install for an instance's worktree."""
    d = sel[instance_id]
    repo = repo_of(instance_id)
    clone = CLONES / repo.split("/")[-1]
    wt = worktree_of(instance_id) if not for_eval else worktree_of(instance_id) / ".." / f"eval-{instance_id}"
    wt = wt.resolve()
    if not wt.exists():
        wt.mkdir(parents=True)
        run(["git", "-C", str(clone), "worktree", "add", "--detach", "--quiet", str(wt), d["base_commit"]])
    venv = venv_of(instance_id)
    py = venv / "bin" / "python"
    if not (venv / "bin" / "pip").exists():
        run(["python3", "-m", "venv", str(venv)])
        run([str(venv / "bin" / "pip"), "install", "--quiet", "--upgrade", "pip"])
    run([str(venv / "bin" / "pip"), "install", "--quiet", "-e", str(wt)], timeout=900)
    for dep in TEST_DEPS[repo]:
        run([str(venv / "bin" / "pip"), "install", "--quiet", dep], timeout=900)
    return wt


def run_pi(instance_id: str, sel: dict, wt: Path) -> Path:
    d = sel[instance_id]
    repo = repo_of(instance_id)
    # reset worktree to pristine base_commit so each model arm starts clean
    # (a previous arm's edits must not leak into this run)
    run(["git", "-C", str(wt), "reset", "--hard", "--quiet", d["base_commit"]], check=False)
    run(["git", "-C", str(wt), "clean", "-fdq"], check=False)
    prompt = PROMPT_TEMPLATE.format(
        repo=repo, base_commit=d["base_commit"], problem=d["problem_statement"]
    )
    config = PiRunConfig(
        model=PI_MODEL, provider=PI_PROVIDER, thinking="minimal", tools=PI_TOOLS
    )
    capture = run_pi_process(
        config=config, prompt=prompt, cwd=wt, timeout_seconds=PI_TIMEOUT
    )
    inst_dir = RESULTS / instance_id
    inst_dir.mkdir(parents=True, exist_ok=True)
    (inst_dir / "raw.ndjson").write_text(capture.stdout)
    records, issues = read_pi_ndjson(capture.stdout)
    (inst_dir / "trace.ndjson").write_text(dump_pi_ndjson(records))
    return inst_dir


def agent_patch(instance_id: str, wt: Path, inst_dir: Path) -> dict:
    """Diff of tracked files + snapshot of untracked files."""
    p = run(["git", "-C", str(wt), "diff", "--binary"], check=False)
    diff = p.stdout
    untracked = run(
        ["git", "-C", str(wt), "ls-files", "--others", "--exclude-standard"], check=False
    ).stdout.splitlines()
    files: dict[str, str] = {}
    for name in untracked:
        f = wt / name
        if f.is_file() and f.stat().st_size < 1_000_000:
            files[name] = f.read_text(errors="replace")
    (inst_dir / "agent.patch").write_text(diff)
    (inst_dir / "untracked.json").write_text(json.dumps(files, ensure_ascii=False))
    return {"has_diff": bool(diff.strip()), "untracked": list(files)}


def eval_instance(instance_id: str, sel: dict, inst_dir: Path) -> dict:
    d = sel[instance_id]
    repo = repo_of(instance_id)
    clone = CLONES / repo.split("/")[1]
    eval_wt = (WORKTREES / f"eval-{instance_id}").resolve()
    if not eval_wt.exists():
        eval_wt.mkdir(parents=True)
        run(["git", "-C", str(clone), "worktree", "add", "--detach", "--quiet", str(eval_wt), d["base_commit"]])
    # ALWAYS reset the eval worktree to pristine base_commit. A previous arm's
    # eval may have left its agent patch applied here; without this reset the
    # leftover fix contaminates the next eval (empty patch would falsely pass).
    run(["git", "-C", str(eval_wt), "reset", "--hard", "--quiet", d["base_commit"]], check=False)
    run(["git", "-C", str(eval_wt), "clean", "-fdq"], check=False)
    # apply agent patch (tracked) then untracked file copies
    ap = inst_dir / "agent.patch"
    if ap.read_text().strip():
        p = run(["git", "-C", str(eval_wt), "apply", "--allow-empty", str(ap)], check=False)
        if p.returncode != 0:
            p2 = run(["patch", "-p1", "-d", str(eval_wt), "-i", str(ap)], check=False)
            agent_applied = p2.returncode == 0
        else:
            agent_applied = True
    else:
        agent_applied = True
    infra = patch_apply_status(instance_id, agent_applied, True, str(eval_wt))
    if infra is not None:
        return infra
    untracked = json.loads((inst_dir / "untracked.json").read_text())
    for name, content in untracked.items():
        f = eval_wt / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
    # apply test patch
    tp = eval_wt / ".swebench_test.patch"
    tp.write_text(d["test_patch"])
    p = run(["git", "-C", str(eval_wt), "apply", "--allow-empty", str(tp)], check=False)
    if p.returncode != 0:
        tp.unlink(missing_ok=True)
        p2 = run(["patch", "-p1", "-d", str(eval_wt), "-i", str(tp)], check=False)
        test_applied = p2.returncode == 0
    else:
        test_applied = True
    infra = patch_apply_status(instance_id, agent_applied, test_applied, str(eval_wt))
    if infra is not None:
        tp.unlink(missing_ok=True)
        return infra
    tp.unlink(missing_ok=True)
    # reinstall editable (worktree path changed)
    venv = venv_of(instance_id)
    try:
        run([str(venv / "bin" / "pip"), "install", "--quiet", "-e", str(eval_wt)], timeout=900)
    except Exception as exc:  # noqa: BLE001
        return _infra_invalid(instance_id, f"env install failed: {exc}", eval_wt)

    test_files = {
        line[6:].split(":")[0]
        for line in d["test_patch"].splitlines()
        if line.startswith("+++ b/")
    }
    nodeid_map = {}
    for fname in test_files:
        nodeid_map[fname] = fname

    def run_test(nodeid: str, timeout=900) -> dict:
        """Return passed + whether the run was INFRA_INVALID (timeout/env error).
        A timeout or executor exception is NOT a normal task failure and must
        not become reward 0."""
        cmd = [str(venv / "bin" / "python"), "-m", "pytest", "-q", "--no-header", nodeid]
        try:
            p = subprocess.run(cmd, cwd=eval_wt, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            return {"passed": False, "infra_invalid": True, "error": "timeout", "tail": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"passed": False, "infra_invalid": True, "error": "executor_error", "tail": str(exc)}
        out = (p.stdout + p.stderr)[-3000:]
        return {"passed": p.returncode == 0, "infra_invalid": False, "returncode": p.returncode, "tail": out}

    ftp = {}
    for nodeid in d["FAIL_TO_PASS"]:
        full = nodeid if "::" in nodeid else f"{next(iter(test_files), 'tests')}::{nodeid}"
        ftp[nodeid] = run_test(full)
    # Full PASS_TO_PASS set must run; regressions here count against resolution
    ptp = {}
    for nodeid in d["PASS_TO_PASS"]:
        full = nodeid if "::" in nodeid else f"{next(iter(test_files), 'tests')}::{nodeid}"
        ptp[nodeid] = run_test(full)

    result = decide_eval_result(instance_id, ftp, ptp, str(eval_wt))
    return result


def _infra_invalid(instance_id: str, message: str, eval_wt: Path) -> dict:
    """An eval that could not produce a trustworthy label."""
    return {
        "instance_id": instance_id,
        "resolved": False,
        "status": "INFRA_INVALID",
        "error": message,
        "ftp": {},
        "ptp": {},
        "eval_worktree": str(eval_wt),
    }


def patch_apply_status(
    instance_id: str,
    agent_applied: bool,
    test_applied: bool,
    eval_wt: str,
) -> dict | None:
    """Pure INFRA_INVALID gate for patch application.

    If the agent patch or the test patch cannot be applied to a pristine
    checkout, the run is infrastructure-invalid (no trustworthy label) and is
    never treated as a plain task failure / reward 0.
    """
    if not agent_applied:
        return _infra_invalid(instance_id, "agent patch apply failed", Path(eval_wt))
    if not test_applied:
        return _infra_invalid(instance_id, "test patch apply failed", Path(eval_wt))
    return None


def decide_eval_result(
    instance_id: str,
    ftp: dict,
    ptp: dict,
    eval_wt: str,
) -> dict:
    """Pure, testable SWE-bench verdict.

    ``ftp``/``ptp`` map nodeid -> test result dict with ``passed`` and
    ``infra_invalid`` flags.  Any infra-invalid test yields INFRA_INVALID (the
    executor failure/timeout is never treated as a plain task failure).
    Resolution requires every FAIL_TO_PASS and every PASS_TO_PASS to pass.
    """
    ftp_infra = [k for k, v in ftp.items() if v.get("infra_invalid")]
    ptp_infra = [k for k, v in ptp.items() if v.get("infra_invalid")]
    if ftp_infra or ptp_infra:
        return _infra_invalid(
            instance_id,
            "verifier executor failure/timeout (not reward 0): "
            + ", ".join(ftp_infra + ptp_infra),
            Path(eval_wt),
        )
    resolved = (
        bool(ftp)
        and all(v.get("passed") for v in ftp.values())
        and bool(ptp)
        and all(v.get("passed") for v in ptp.values())
    )
    return {
        "instance_id": instance_id,
        "resolved": resolved,
        "status": "RESOLVED" if resolved else "UNRESOLVED",
        "ftp": {k: v.get("passed") for k, v in ftp.items()},
        "ftp_details": ftp,
        "ptp": {k: v.get("passed") for k, v in ptp.items()},
        "ptp_results": ptp,
        "eval_worktree": eval_wt,
    }


def main():
    global PI_MODEL, PI_PROVIDER
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--phase", default="all", choices=["all", "setup", "pi", "eval"])
    parser.add_argument("--provider", default=PI_PROVIDER)
    parser.add_argument("--model", default=PI_MODEL)
    parser.add_argument("--adapter", default="", help="local-qwen LoRA adapter path ('' = base)")
    parser.add_argument("--local-model", default="", help="local-qwen base model path (e.g. coder-7b)")
    parser.add_argument("--tag", default="", help="results namespace (e.g. base-7b / sft-7b)")
    parser.add_argument("--sel", default="", help="path to selected_instances.json (default ROOT/selected_instances.json)")
    args = parser.parse_args()
    PI_MODEL, PI_PROVIDER = args.model, args.provider

    global RESULTS, SEL
    if args.sel:
        SEL = Path(args.sel)
    if args.tag:
        RESULTS = ROOT / "results" / args.tag

    sel = json.loads(SEL.read_text())
    instances = list(sel) if not args.only else [x.strip() for x in args.only.split(",")]
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.provider == "local-qwen":
        ensure_local_server(args.adapter, args.local_model)
    clone_repos()

    summary = {}
    for iid in instances:
        print("=" * 70, flush=True)
        print("INSTANCE", iid, flush=True)
        # fresh server per instance => clean GPU memory (avoids OOM-500 drift)
        if args.provider == "local-qwen" and args.phase in ("all", "pi"):
            ensure_local_server(args.adapter, args.local_model, force=True)
        inst_dir = RESULTS / iid
        inst_dir.mkdir(parents=True, exist_ok=True)
        rec = {"instance_id": iid}
        try:
            if args.phase in ("all", "setup"):
                wt = setup_instance(iid, sel)
                rec["env_ok"] = True
            if args.phase in ("all", "pi"):
                wt = worktree_of(iid).resolve()
                if not (inst_dir / "trace.ndjson").exists():
                    run_pi(iid, sel, wt)
                rec["pi_ok"] = True
            if args.phase in ("all", "eval"):
                inst_dir2 = inst_dir if (inst_dir / "agent.patch").exists() else inst_dir
                if not (inst_dir / "agent.patch").exists() or not (inst_dir / "eval.json").exists():
                    wt = worktree_of(iid).resolve()
                    ap = agent_patch(iid, wt, inst_dir)
                    rec["patch"] = ap
                    ev = eval_instance(iid, sel, inst_dir)
                    rec["eval"] = {k: v for k, v in ev.items() if k != "ftp_details"}
                    rec["ftp_details"] = ev["ftp_details"]
                    (inst_dir / "eval.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1))
            summary[iid] = rec
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            summary[iid] = rec
            print("ERROR", iid, exc, flush=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print("DONE")


if __name__ == "__main__":
    main()
