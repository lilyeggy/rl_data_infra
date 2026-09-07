#!/usr/bin/env python3
"""Run frozen BigCodeBench tasks through Pi with containerized verification."""

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
    parser.add_argument("--pi", required=True); parser.add_argument("--python", required=True)
    parser.add_argument("--upstream-url", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True); parser.add_argument("--image", required=True)
    parser.add_argument("--verifier-entrypoint", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1); parser.add_argument("--limit", type=int, default=148)
    parser.add_argument("--prompt-mode", choices=("instruct", "complete"), default="instruct")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--require-rl-evidence", action="store_true",
                        help="fail closed unless every model call has native RL evidence")
    args = parser.parse_args()
    manifest = json.loads(args.dataset.read_text())
    tasks = list(manifest["tasks"].items())[: args.limit]
    args.workspace_root.mkdir(parents=True, exist_ok=True); args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for task_id, task in tasks:
        safe_id = task_id.replace("/", "_")
        workspace = args.workspace_root / safe_id; output = args.output_root / f"{safe_id}-attempt-{args.attempt}"
        if output.exists():
            results.append({"task_id": task_id, "status": "SKIPPED", "reason": "output_exists"}); continue
        workspace.mkdir(parents=True, exist_ok=True)
        template = task["code_prompt"] if args.prompt_mode == "instruct" else task["complete_prompt"]
        (workspace / "solution.py").write_text(template + "    pass\n")
        identity = ExecutionIdentity(
            run_id=f"bigcodebench-{safe_id}-v0.1.1-{args.attempt}", task_id=task_id,
            episode_id=f"episode-bigcodebench-{safe_id}-{args.attempt}", attempt_id=args.attempt,
            producer_id="pi-host", producer_version="pi-host/v1", group_id="bigcodebench-hard-v0.1.1",
            policy_fingerprint=sha256_json({"policy": f"bigcodebench-{args.prompt_mode}-v1"}),
            sampling_fingerprint=sha256_json({"thinking": "minimal", "tools": ["read", "bash", "write", "edit", "ls"]}),
        )
        task_prompt = task["instruct_prompt"] if args.prompt_mode == "instruct" else task["complete_prompt"]
        prompt = (
            "Complete the requested implementation in solution.py. Preserve all provided imports, docstring, and function signature. "
            "Use only the workspace; do not modify tests or evaluator files. You may inspect and edit solution.py.\n\n"
            f"Task ({args.prompt_mode} mode):\n{task_prompt}\n\n"
            "Leave a complete executable implementation in solution.py, then give a concise final response."
        )
        verifier = (args.python, str(Path(__file__).with_name("verify_bigcodebench.py")), "--manifest", str(args.dataset),
                    "--task-id", task_id, "--source-worktree", str(workspace), "--output", str(output / "verifier-output.json"),
                    "--image", args.image, "--verifier-entrypoint", str(args.verifier_entrypoint))
        spec = PiHostExecutionSpec(
            identity=identity, pi=args.pi, workspace=str(workspace), prompt=prompt,
            tools=("read", "bash", "write", "edit", "ls"), timeout_seconds=args.timeout,
            upstream_url=args.upstream_url, provider="local-openai-compatible", provider_api="openai-completions",
            model=args.model, model_revision=args.model_revision,
            harness_manifest=HarnessManifest(name="pi", version="0.84.2", revision="pi-json/v0.84.2",
                config_digest=sha256_json({"benchmark": "bigcodebench", "prompt": f"{args.prompt_mode}-v1"}),
                policy_flags={"benchmark": "bigcodebench", "policy": f"{args.prompt_mode}-v1"}),
            evaluator_manifest=EvaluatorManifest(name="bigcodebench-official-container", revision="v0.1.1",
                config_digest=sha256_json({"task": task, "runtime": "docker-restricted-v1"})),
            verifier_command=verifier, task_snapshot=sha256_json(task),
            experiment_manifest_ref=f"bigcodebench-hard:{args.dataset}",
            sampling_config={"thinking": "minimal", "tools": ["read", "bash", "write", "edit", "ls"]},
            require_rl_evidence=args.require_rl_evidence,
        )
        try: results.append(PiHostExecutionOrchestrator().run(spec, output_dir=output))
        except Exception as exc: results.append({"task_id": task_id, "status": "ERROR", "error": str(exc), "traceback": traceback.format_exc()})
    (args.output_root / f"batch-{args.attempt}.json").write_text(json.dumps({"results": results}, indent=2) + "\n")
    print(json.dumps({"completed": len(results), "eligible": sum(x.get("sft_verdict") == "ELIGIBLE" for x in results),
                      "rl_eligible": sum(x.get("on_policy_rl_verdict") == "ELIGIBLE" for x in results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
