#!/usr/bin/env python3
"""Run APPS train tasks through the Pi Host Data Plane."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from src.contracts._json import sha256_json
from src.contracts.execution_identity import ExecutionIdentity
from src.contracts.manifests import EvaluatorManifest, HarnessManifest
from src.orchestration import PiHostExecutionOrchestrator, PiHostExecutionSpec


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
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--require-rl-evidence", action="store_true")
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    args.workspace_root = args.workspace_root.resolve()
    args.output_root = args.output_root.resolve()
    args.pi = str(Path(args.pi).resolve())
    args.python = str(Path(args.python).resolve())
    manifest = json.loads(args.dataset.read_text(encoding="utf-8"))
    tasks = list(manifest["tasks"].items())[: args.limit]
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for task_id, task in tasks:
        workspace = args.workspace_root / task_id
        output = args.output_root / f"{task_id}-attempt-1"
        if output.exists():
            results.append({"task_id": task_id, "status": "SKIPPED", "reason": "output_exists"})
            continue
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "solution.py").write_text(
            "# Implement the requested program here.\n", encoding="utf-8"
        )
        identity = ExecutionIdentity(
            run_id=f"{task_id}-teacher-v1-1",
            task_id=task_id,
            episode_id=f"episode-{task_id}-1",
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
        verifier = (
            args.python,
            str(Path(__file__).with_name("verify_apps.py")),
            "--manifest",
            str(args.dataset),
            "--task-id",
            task_id,
            "--source-worktree",
            str(workspace),
            "--python",
            args.python,
            "--output",
            str(output / "verifier-output.json"),
        )
        spec = PiHostExecutionSpec(
            identity=identity,
            pi=args.pi,
            workspace=str(workspace),
            prompt=prompt,
            tools=("read", "bash", "write", "edit", "ls"),
            timeout_seconds=args.timeout,
            upstream_url=args.upstream_url,
            provider=args.provider,
            provider_api=args.provider_api,
            model=args.model,
            model_revision=args.model_revision,
            harness_manifest=HarnessManifest(
                name="pi",
                version="0.84.2",
                revision="pi-json/v0.84.2",
                config_digest=sha256_json({"benchmark": "apps", "split": "train"}),
                policy_flags={"benchmark": "apps", "policy": "stdin-stdout-v1"},
            ),
            evaluator_manifest=EvaluatorManifest(
                name="apps-execution-verifier",
                revision="apps-verifier/v2",
                config_digest=sha256_json(
                    {
                        "task": task,
                        "policy": "all-public-cases",
                        "io_serialization": "top-level-newlines/v2",
                    }
                ),
            ),
            verifier_command=verifier,
            task_snapshot=sha256_json(task),
            experiment_manifest_ref=f"apps:{args.dataset}",
            sampling_config={
                "thinking": "minimal",
                "tools": ["read", "bash", "write", "edit", "ls"],
            },
            require_rl_evidence=args.require_rl_evidence,
        )
        try:
            results.append(PiHostExecutionOrchestrator().run(spec, output_dir=output))
        except Exception as exc:
            results.append(
                {
                    "task_id": task_id,
                    "status": "ERROR",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
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
