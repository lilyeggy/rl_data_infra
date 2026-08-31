#!/usr/bin/env python3
"""Run MBPP coding tasks through the certified Pi host data plane."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import traceback
from pathlib import Path

from src.contracts._json import sha256_json
from src.contracts.execution_identity import ExecutionIdentity
from src.contracts.manifests import EvaluatorManifest, HarnessManifest
from src.orchestration import PiHostExecutionOrchestrator, PiHostExecutionSpec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pi", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--min-task-id", type=int, default=1)
    parser.add_argument("--max-task-id", type=int, default=974)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    args.dataset = args.dataset.resolve(); args.workspace_root = args.workspace_root.resolve(); args.output_root = args.output_root.resolve()
    tasks = json.loads(args.dataset.read_text())
    items = [item for item in sorted(tasks.items(), key=lambda item: int(item[0]))
             if args.min_task_id <= int(item[0]) <= args.max_task_id][:args.limit]
    args.workspace_root.mkdir(parents=True, exist_ok=True); args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for task_id, task in items:
        safe_id = f"mbpp-{task_id}"
        workspace = args.workspace_root / safe_id
        output = args.output_root / f"{safe_id}-attempt-{args.attempt}"
        workspace.mkdir(parents=True, exist_ok=True)
        if output.exists():
            results.append({"task_id": task_id, "status": "SKIPPED", "reason": "output_exists"}); continue
        solution = workspace / "solution.py"
        solution.write_text("# Implement the requested function here.\n")
        identity = ExecutionIdentity(
            run_id=f"mbpp-{task_id}-baseline-v1-{args.attempt}", task_id=f"mbpp-{task_id}",
            episode_id=f"episode-mbpp-{task_id}-{args.attempt}", attempt_id=args.attempt,
            producer_id="pi-host", producer_version="pi-host/v1", group_id="mbpp-stage1",
            policy_fingerprint=sha256_json({"policy": "mbpp-visible-tests-v1"}),
            sampling_fingerprint=sha256_json({"thinking": "minimal", "tools": ["read", "bash", "write", "edit", "ls"]}),
        )
        tests = "\n".join([*task.get("test_list", []), *task.get("challenge_test_list", [])])
        prompt = ("Implement the requested Python function in solution.py. Work inside the workspace, "
                  "run the supplied tests, inspect failures, and fix the implementation.\n\n"
                  f"Task: {task['text']}\n\nTests:\n{tests}\n\n"
                  "Do not modify the verifier or test harness; leave the final implementation in solution.py. "
                  "After the tests pass, send one concise final response stating what you changed and that "
                  "the tests passed; do not end the task immediately after a tool call.")
        verifier = (args.python, str((Path(__file__).resolve().parent / "verify_mbpp.py")),
                    "--manifest", str(args.dataset), "--task-id", task_id,
                    "--source-worktree", str(workspace), "--python", args.python,
                    "--output", str(output / "verifier-output.json"))
        spec = PiHostExecutionSpec(
            identity=identity, pi=args.pi, workspace=str(workspace), prompt=prompt,
            tools=("read", "bash", "write", "edit", "ls"), timeout_seconds=args.timeout,
            upstream_url=args.upstream_url, provider="local-openai-compatible", provider_api="openai-completions",
            model=args.model, model_revision=args.model_revision,
            harness_manifest=HarnessManifest(name="pi", version="0.84.2", revision="pi-json/v0.84.2",
                config_digest=sha256_json({"benchmark": "mbpp", "tests": "visible-plus-challenge"}),
                policy_flags={"benchmark": "mbpp", "policy": "visible-plus-challenge-v1"}),
            evaluator_manifest=EvaluatorManifest(name="mbpp-execution-verifier", revision="mbpp-verifier/v1",
                config_digest=sha256_json({"task": task, "policy": "run-all-tests"})), verifier_command=verifier,
            task_snapshot=sha256_json({"task_id": task_id, "task": task}),
            experiment_manifest_ref=f"mbpp:{args.dataset}",
            sampling_config={"thinking": "minimal", "tools": ["read", "bash", "write", "edit", "ls"]},
            require_rl_evidence=False)
        try:
            results.append(PiHostExecutionOrchestrator().run(spec, output_dir=output))
        except Exception as exc:
            results.append({"task_id": task_id, "status": "ERROR", "error": str(exc), "traceback": traceback.format_exc()})
    (args.output_root / f"batch-{args.attempt}.json").write_text(json.dumps({"results": results}, indent=2) + "\n")
    print(json.dumps({"completed": len(results), "eligible": sum(item.get("sft_verdict") == "ELIGIBLE" for item in results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
