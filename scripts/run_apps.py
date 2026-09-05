#!/usr/bin/env python3
"""Run APPS train tasks through the Pi Host Data Plane."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import traceback
from pathlib import Path

from src.contracts._json import sha256_json
from src.contracts.execution_identity import ExecutionIdentity
from src.contracts.manifests import EvaluatorManifest, HarnessManifest
from src.orchestration import PiHostExecutionOrchestrator, PiHostExecutionSpec


def _run_one_task(args: argparse.Namespace, task_id: str, task: dict) -> dict:
    """Run one isolated task; safe to call from a bounded worker pool."""

    workspace = args.workspace_root / task_id
    output = args.output_root / f"{task_id}-attempt-1"
    if output.exists():
        return {"task_id": task_id, "status": "SKIPPED", "reason": "output_exists"}
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "solution.py").write_text(
        "# Implement the requested program here.\n", encoding="utf-8"
    )
    identity = ExecutionIdentity(
        run_id=f"{task_id}-{args.run_label}",
        task_id=task_id,
        episode_id=f"episode-{task_id}-{args.run_label}",
        attempt_id=1,
        producer_id="pi-host",
        producer_version="pi-host/v1",
        group_id="apps-train-v1",
        policy_fingerprint=sha256_json({"policy": "apps-teacher-v1"}),
        sampling_fingerprint=sha256_json(
            {"thinking": "minimal", "tools": ["read", "bash", "write", "edit", "ls"]}
        ),
    )
    prompt = (
        "Implement the requested programming problem as a complete executable solution.py. "
        "The program reads the specified input from stdin and writes the answer to stdout. "
        "Work only in the workspace, inspect and test your solution, and do not "
        "modify the verifier.\n\n"
        f"Problem:\n{task['question']}\n\n"
        "Use the examples/tests in the task description when available. Run local "
        "checks before finishing."
    )
    if args.timeboxed:
        prompt += (
            "\n\nTime-boxed execution policy: implement the solution early and keep it "
            "runnable after every edit. Run only the stated examples and a small "
            "number of targeted checks. Do not write brute-force reference solvers, "
            "random stress tests, large benchmarks, or repeated exploratory scripts. "
            "Do not spend time optimizing a test harness; finish the solution and "
            "stop once the examples pass."
        )
    verifier = (
        args.python,
        str(Path(__file__).with_name("verify_apps.py")),
        "--manifest", str(args.dataset), "--task-id", task_id,
        "--source-worktree", str(workspace), "--python", args.python,
        "--output", str(output / "verifier-output.json"),
    )
    spec = PiHostExecutionSpec(
        identity=identity, pi=args.pi, workspace=str(workspace), prompt=prompt,
        tools=("read", "bash", "write", "edit", "ls"), timeout_seconds=args.timeout,
        upstream_url=args.upstream_url, provider=args.provider, provider_api=args.provider_api,
        model=args.model, model_revision=args.model_revision,
        harness_manifest=HarnessManifest(
            name="pi", version="0.84.2", revision="pi-json/v0.84.2",
            config_digest=sha256_json({"benchmark": "apps", "split": "train"}),
            policy_flags={"benchmark": "apps", "policy": "stdin-stdout-v1"},
        ),
        evaluator_manifest=EvaluatorManifest(
            name="apps-execution-verifier", revision="apps-verifier/v2",
            config_digest=sha256_json({
                "task": task, "policy": "all-public-cases",
                "io_serialization": "top-level-newlines/v2",
                "execution_policy": "timeboxed-v1" if args.timeboxed else "default-v1",
            }),
        ),
        verifier_command=verifier, task_snapshot=sha256_json(task),
        experiment_manifest_ref=f"apps:{args.dataset}",
        sampling_config={"thinking": "minimal", "tools": ["read", "bash", "write", "edit", "ls"]},
        require_rl_evidence=args.require_rl_evidence,
    )
    try:
        return PiHostExecutionOrchestrator().run(spec, output_dir=output)
    except Exception as exc:
        return {
            "task_id": task_id, "status": "ERROR", "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pi", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--upstream-url")
    parser.add_argument("--provider", default="opencode-go")
    parser.add_argument("--provider-api", default="openai-completions")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--task-ids-file", type=Path)
    parser.add_argument("--run-label", default="teacher-v1-1")
    parser.add_argument("--timeboxed", action="store_true")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="bounded concurrent task executions; 1 preserves serial behavior",
    )
    parser.add_argument("--require-rl-evidence", action="store_true")
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    args.workspace_root = args.workspace_root.resolve()
    args.output_root = args.output_root.resolve()
    args.pi = str(Path(args.pi).resolve())
    args.python = str(Path(args.python).resolve())
    manifest = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.task_ids_file:
        task_ids = [
            line.strip()
            for line in args.task_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        missing = [task_id for task_id in task_ids if task_id not in manifest["tasks"]]
        if missing:
            raise ValueError(f"task IDs missing from manifest: {missing}")
        tasks = [(task_id, manifest["tasks"][task_id]) for task_id in task_ids]
    else:
        tasks = list(manifest["tasks"].items())[: args.limit]
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    results_by_task: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_one_task, args, task_id, task): task_id
            for task_id, task in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            task_id = futures[future]
            results_by_task[task_id] = future.result()
            checkpoint = [
                results_by_task[item_id]
                for item_id, _ in tasks
                if item_id in results_by_task
            ]
            (args.output_root / "batch-1.json").write_text(
                json.dumps({"results": checkpoint}, indent=2) + "\n", encoding="utf-8"
            )
    results = [results_by_task[task_id] for task_id, _ in tasks]
    (args.output_root / "batch-1.json").write_text(
        json.dumps({"results": results}, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "completed": len(results),
                "eligible": sum(item.get("sft_verdict") == "ELIGIBLE" for item in results),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
