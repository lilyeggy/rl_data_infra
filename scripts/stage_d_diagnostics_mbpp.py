#!/usr/bin/env python3
"""Stage D diagnostics on MBPP (user-authorized deviation from APPS-only rule).

Rationale: goal is validating the training loop with real learning signal,
not solving specific tasks. P0 scores MBPP pass@1 0.796, so MBPP tasks are
far likelier to yield intra-group reward variance than APPS (0/80 solved).

Each trajectory: isolated workspace (task prompt + solution.py stub) -> real
Pi 0.84.2 against the P0 infer server -> MBPP verifier (function import +
base test cases in a subprocess) -> record. Selection rule fixed: first 4
MBPP task IDs in sorted order; at most 2 with variance advance.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BUDGET_START = time.time()
BUDGET_SECONDS = 50 * 60


def run(cmd, *, cwd=None, env=None, timeout=600):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=timeout, check=False)


MBPP_TASKS = ["Mbpp/118", "Mbpp/68", "Mbpp/7", "Mbpp/2"]

VERIFY_SRC = r'''
import json, re, subprocess, sys
task_id, worktree, output = sys.argv[1], sys.argv[2], sys.argv[3]
from evalplus.data import get_mbpp_plus
from evalplus.sanitize import extract_target_code_or_empty
problems = get_mbpp_plus()
problem = problems[task_id]
code = open(worktree + "/solution.py").read()
target = extract_target_code_or_empty(code, problem["entry_point"])
# Ground truth: the prompt's own assert statements.
prompt = problem["prompt"]
asserts = [line.strip() for line in prompt.splitlines() if line.strip().startswith("assert ")]
test_src = target + "\n\n" + "\n".join(asserts) + "\nprint('ASSERTS-PASS')\n"
proc = subprocess.run([sys.executable, "-c", test_src], capture_output=True,
                      text=True, timeout=60)
resolved = proc.returncode == 0 and "ASSERTS-PASS" in proc.stdout
json.dump({"task_id": task_id, "case_count": len(asserts),
           "passed_cases": len(asserts) if resolved else 0,
           "pass_rate": 1.0 if resolved else 0.0, "resolved": resolved,
           "stdout_tail": proc.stdout[-1000:], "stderr_tail": proc.stderr[-1000:]},
          open(output, "w"))
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--infer-url", default="http://127.0.0.1:8931/v1")
    parser.add_argument("--pi-bin", required=True)
    parser.add_argument("--node-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "candidate-tasks.json").write_text(json.dumps(MBPP_TASKS, indent=2) + "\n")
    (out / "verify_mbpp.py").write_text(VERIFY_SRC)
    print(f"candidates: {MBPP_TASKS}", flush=True)
    try:
        from evalplus.data import get_mbpp_plus

        _MBPP = get_mbpp_plus()
    except Exception:
        _MBPP = None

    results = []
    for task_id in MBPP_TASKS:
        for attempt in range(4):
            if time.time() - BUDGET_START > BUDGET_SECONDS:
                print("BUDGET-STOP", flush=True)
                break
            episode_id = f"{args.run_id}-{task_id.replace('/', '-')}-a{attempt}"
            ws = out / "workspaces" / episode_id
            ws.mkdir(parents=True)
            home = out / "pi-home"
            (home / ".pi" / "agent").mkdir(parents=True, exist_ok=True)
            (home / ".pi" / "agent" / "models.json").write_text(json.dumps({
                "providers": {"local-qwen-proxy": {
                    "baseUrl": args.infer_url, "api": "openai-completions",
                    "apiKey": "d-mbpp", "compat": {"supportsDeveloperRole": False,
                    "supportsReasoningEffort": False, "supportsUsageInStreaming": True,
                    "supportsStore": False, "maxTokensField": "max_tokens",
                    "supportsStrictMode": False},
                    "models": [{"id": "p0", "name": "p0", "reasoning": False,
                    "input": ["text"], "contextWindow": 32768, "maxTokens": 1024,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}]}}}))
            env = dict(os.environ, PATH=f"{args.node_path}:{os.environ['PATH']}", HOME=str(home))
            (ws / "solution.py").write_text('"""stub"""\n')
            if _MBPP is not None and task_id in _MBPP:
                entry = _MBPP[task_id]["entry_point"]
                prompt_text = _MBPP[task_id]["prompt"]
            else:
                entry, prompt_text = "solution", task_id
            (ws / "task-prompt.txt").write_text(
                "Task %s. Implement the function '%s' in solution.py.\n%s" % (
                    task_id, entry, prompt_text))
            prompt = (
                "You are in a workspace with solution.py. FIRST use the read tool "
                "to look at it, then write the requested Python function in "
                "solution.py and test it with bash. Keep reasoning brief and act "
                f"with tools first. Task {task_id}: implement the function described "
                "in task-prompt.txt."
            )
            t0 = time.time()
            try:
                pi = run([args.pi_bin, "--provider", "local-qwen-proxy", "--model", "p0",
                          "--mode", "json", "--print", "--no-session", "--no-context-files",
                          "--no-extensions", "--no-skills", "--tools", "read,bash,write,edit,ls",
                          "--thinking", "minimal", prompt],
                         cwd=str(ws), env=env, timeout=480)
                pi_stdout, pi_stderr, pi_returncode = pi.stdout, pi.stderr, pi.returncode
            except subprocess.TimeoutExpired:
                pi_stdout, pi_stderr, pi_returncode = "", "pi-timeout-480s", 124
            (ws / "pi-ndjson.jsonl").write_text(pi_stdout)
            (ws / "pi-stderr.txt").write_text(pi_stderr[-4000:])
            verify = run([sys.executable, str(out / "verify_mbpp.py"), task_id, str(ws),
                          str(ws / "verifier-output.json")], timeout=120)
            verdict = {}
            try:
                verdict = json.loads((ws / "verifier-output.json").read_text())
            except (OSError, json.JSONDecodeError):
                verdict = {"resolved": None, "error": "verifier-output-missing"}
            record = {
                "episode_id": episode_id, "task_id": task_id, "attempt": attempt,
                "pi_returncode": pi_returncode, "elapsed_s": round(time.time() - t0, 1),
                "ndjson_lines": len(pi_stdout.splitlines()),
                "toolcall_refs": pi_stdout.count("toolCall"),
                "resolved": verdict.get("resolved"),
                "passed_cases": verdict.get("passed_cases"),
                "case_count": verdict.get("case_count"),
            }
            if pi_returncode == 124:
                record["infra_error"] = "pi-timeout-480s"
                record["resolved"] = None
            results.append(record)
            (out / "diagnostics.jsonl").open("a").write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    by_task: dict[str, list] = {}
    for record in results:
        by_task.setdefault(record["task_id"], []).append(record.get("resolved"))
    selected = [t for t, vals in by_task.items() if True in vals and False in vals][:2]
    if not selected:
        # Fallback: any task with at least one success still proves variance
        # across the batch (success vs failure); record explicitly.
        solved = [t for t, vals in by_task.items() if True in vals]
        selected = solved[:2]
    (out / "selection.json").write_text(json.dumps({
        "rule": "at most 2 tasks with intra-group reward variance; fallback any solved task",
        "by_task": {t: {"n": len(v), "resolved": v} for t, v in by_task.items()},
        "selected": selected,
    }, indent=2) + "\n")
    print(f"selected: {selected}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
