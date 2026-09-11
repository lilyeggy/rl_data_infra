#!/usr/bin/env python3
"""Stage D diagnostics (DEVIATED single-GPU1): 4 candidate tasks x 4 real Pi trajectories.

Each trajectory: isolated workspace (problem + initial solution.py + public
examples only) -> real Pi 0.84.2 (read/bash/write/edit/ls) against the local
P0 infer server -> verify_apps.py in an isolated env -> Episode/Bundle/
ON_POLICY_RL certification via the repo contracts.

Selection rule (fixed before running): from the train manifest, take the first
4 task IDs in sorted order that are NOT in the holdout set. From diagnostic
results, take at most 2 tasks WITH intra-group reward variance for the formal
cycle. Full candidate results + rule are saved; this is training selection,
never reported as effect evaluation.

 Pi executions are sequential (max 1 concurrent on single GPU1).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

REPO = Path("/home/cxr/agentic/code")
sys.path.insert(0, str(REPO))

BUDGET_START = time.time()
BUDGET_SECONDS = 50 * 60  # stay inside the overall window


def run(cmd, *, cwd=None, env=None, timeout=600):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=timeout, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--infer-url", default="http://127.0.0.1:8931/v1")
    parser.add_argument("--pi-bin", required=True)
    parser.add_argument("--node-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--task-ids", default=None)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path("/home/cxr/agentic/datasets/apps/train.manifest.json").read_text())
    holdout = set(json.loads(Path("/home/cxr/agentic/rl-runs/apps-rl-cycle-003/holdout_eval_tasks.json").read_text()))
    if args.task_ids:
        candidates = args.task_ids.split(",")
    else:
        offset = args.offset
        candidates = sorted(t for t in manifest["tasks"] if t not in holdout)[offset:offset + 4]
    (out / "candidate-tasks.json").write_text(json.dumps(candidates, indent=2) + "\n")
    print(f"candidates: {candidates}", flush=True)

    results = []
    for task_id in candidates:
        task = manifest["tasks"][task_id]
        for attempt in range(4):
            if time.time() - BUDGET_START > BUDGET_SECONDS:
                print("BUDGET-STOP", flush=True)
                break
            episode_id = f"{args.run_id}-{task_id}-a{attempt}"
            ws = out / "workspaces" / episode_id
            ws.mkdir(parents=True)
            (ws / "problem.txt").write_text(task["question"][:4000])
            (ws / "solution.py").write_text('import sys\n\ndef solve():\n    data = sys.stdin.read().strip().split()\n    print("TODO")\n\nif __name__ == "__main__":\n    solve()\n')
            prompt = (
                "You are in a workspace with solution.py. FIRST use the read tool "
                "to look at solution.py, then implement the programming problem "
                "(stdin/stdout) and test it with bash. Keep reasoning brief and "
                "act with tools first. "
                f"Problem:\n{task['question'][:3000]}"
            )
            home = out / "pi-home"
            (home / ".pi" / "agent").mkdir(parents=True, exist_ok=True)
            (home / ".pi" / "agent" / "models.json").write_text(json.dumps({
                "providers": {"local-qwen-proxy": {
                    "baseUrl": args.infer_url, "api": "openai-completions",
                    "apiKey": "d-stage", "compat": {"supportsDeveloperRole": False,
                    "supportsReasoningEffort": False, "supportsUsageInStreaming": True,
                    "supportsStore": False, "maxTokensField": "max_tokens",
                    "supportsStrictMode": False},
                    "models": [{"id": "p0", "name": "p0", "reasoning": False,
                    "input": ["text"], "contextWindow": 32768, "maxTokens": 1024,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}]}}}))
            env = dict(os.environ, PATH=f"{args.node_path}:{os.environ['PATH']}", HOME=str(home))
            t0 = time.time()
            try:
                pi = run([args.pi_bin, "--provider", "local-qwen-proxy", "--model", "p0",
                          "--mode", "json", "--print", "--no-session", "--no-context-files",
                          "--no-extensions", "--no-skills", "--tools", "read,bash,write,edit,ls",
                          "--thinking", "minimal", prompt],
                         cwd=str(ws), env=env, timeout=480)
                pi_stdout, pi_stderr, pi_returncode = pi.stdout, pi.stderr, pi.returncode
            except subprocess.TimeoutExpired:
                # Infrastructure timeout: record as infra error, never as reward 0.
                pi_stdout, pi_stderr, pi_returncode = "", "pi-timeout-480s", 124
            (ws / "pi-ndjson.jsonl").write_text(pi_stdout)
            (ws / "pi-stderr.txt").write_text(pi_stderr[-4000:])
            # Verifier: need APPS manifest shape; reuse verify_apps.py manifest format.
            verifier_manifest = {"tasks": {task_id: {
                "input_output": task.get("input_output", {})}}}
            (ws / "verifier-manifest.json").write_text(json.dumps(verifier_manifest))
            verify = run([sys.executable, str(REPO / "scripts" / "verify_apps.py"),
                          "--manifest", str(ws / "verifier-manifest.json"),
                          "--task-id", task_id, "--source-worktree", str(ws),
                          "--output", str(ws / "verifier-output.json"),
                          "--python", sys.executable],
                         timeout=120)
            verdict = {}
            try:
                verdict = json.loads((ws / "verifier-output.json").read_text())
            except (OSError, json.JSONDecodeError):
                verdict = {"resolved": None, "error": "verifier-output-missing"}
            tool_calls = pi_stdout.count("toolCall")
            record = {
                "episode_id": episode_id, "task_id": task_id, "attempt": attempt,
                "pi_returncode": pi_returncode, "elapsed_s": round(time.time() - t0, 1),
                "ndjson_lines": len(pi_stdout.splitlines()),
                "toolcall_refs": tool_calls,
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
    # Selection: tasks with variance (both resolved true and false present).
    by_task: dict[str, list] = {}
    for record in results:
        by_task.setdefault(record["task_id"], []).append(record.get("resolved"))
    selected = [t for t, vals in by_task.items()
                if True in vals and False in vals][:2]
    (out / "selection.json").write_text(json.dumps({
        "rule": "at most 2 tasks with intra-group reward variance (True and False present)",
        "by_task": {t: {"n": len(v), "resolved": v} for t, v in by_task.items()},
        "selected": selected,
    }, indent=2) + "\n")
    print(f"selected: {selected}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
