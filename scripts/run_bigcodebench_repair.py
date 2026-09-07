#!/usr/bin/env python3
"""Run one explicit verifier-feedback repair turn for failed BigCodeBench tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.contracts._json import sha256_json
from src.contracts.execution_identity import ExecutionIdentity
from src.contracts.manifests import EvaluatorManifest, HarnessManifest
from src.orchestration import PiHostExecutionOrchestrator, PiHostExecutionSpec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--initial-output-root", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pi", required=True); parser.add_argument("--python", required=True)
    parser.add_argument("--upstream-url", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True); parser.add_argument("--image", required=True)
    parser.add_argument("--verifier-entrypoint", type=Path, required=True); parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args(); manifest = json.loads(args.dataset.read_text())["tasks"]
    args.output_root.mkdir(parents=True, exist_ok=True); results = []
    for initial in sorted(args.initial_output_root.glob("*/summary.json")):
        summary = json.loads(initial.read_text())
        if summary.get("verifier_status") == "PASSED": continue
        task_id = summary["task_id"]; task = manifest[task_id]; safe_id = task_id.replace("/", "_")
        output = args.output_root / f"{safe_id}-repair-attempt-2"
        if output.exists(): results.append({"task_id": task_id, "status": "SKIPPED"}); continue
        workspace = args.workspace_root / safe_id
        verdict = json.loads((initial.parent / "verifier-output.json").read_text())
        feedback = json.dumps({k: verdict.get(k) for k in ("status", "details", "reason")}, ensure_ascii=False)[:8000]
        identity = ExecutionIdentity(run_id=f"bigcodebench-{safe_id}-repair-v1-2", task_id=task_id,
            episode_id=f"episode-bigcodebench-{safe_id}-repair-2", attempt_id=2, producer_id="pi-host",
            producer_version="pi-host/v1", group_id="bigcodebench-hard-v0.1.1-repair",
            policy_fingerprint=sha256_json({"policy": "bigcodebench-verifier-feedback-repair-v1"}),
            sampling_fingerprint=sha256_json({"thinking": "minimal", "tools": ["read", "bash", "write", "edit", "ls"]}))
        prompt = ("Repair the existing solution.py for the requested task. Do not change its function signature, tests, or evaluator. "
                  "The isolated official verifier rejected the previous attempt; use its failure summary to make a targeted fix.\n\n"
                  f"Task:\n{task['instruct_prompt']}\n\nVerifier feedback:\n{feedback}\n\n"
                  "Read solution.py first, then edit it. Leave the corrected complete implementation in solution.py.")
        verifier = (args.python, str(Path(__file__).with_name("verify_bigcodebench.py")), "--manifest", str(args.dataset),
            "--task-id", task_id, "--source-worktree", str(workspace), "--output", str(output / "verifier-output.json"),
            "--image", args.image, "--verifier-entrypoint", str(args.verifier_entrypoint))
        spec = PiHostExecutionSpec(identity=identity, pi=args.pi, workspace=str(workspace), prompt=prompt,
            tools=("read", "bash", "write", "edit", "ls"), timeout_seconds=args.timeout, upstream_url=args.upstream_url,
            provider="local-openai-compatible", provider_api="openai-completions", model=args.model, model_revision=args.model_revision,
            harness_manifest=HarnessManifest(name="pi", version="0.84.2", revision="pi-json/v0.84.2",
                config_digest=sha256_json({"benchmark": "bigcodebench-hard", "policy": "verifier-feedback-repair-v1"}),
                policy_flags={"benchmark": "bigcodebench-hard", "policy": "verifier-feedback-repair-v1"}),
            evaluator_manifest=EvaluatorManifest(name="bigcodebench-official-container", revision="v0.1.1",
                config_digest=sha256_json({"task": task, "runtime": "docker-restricted-v1"})), verifier_command=verifier,
            task_snapshot=sha256_json(task), experiment_manifest_ref=f"bigcodebench-hard:{args.dataset}",
            sampling_config={"thinking": "minimal", "tools": ["read", "bash", "write", "edit", "ls"]}, require_rl_evidence=False)
        results.append(PiHostExecutionOrchestrator().run(spec, output_dir=output))
    (args.output_root / "batch-repair.json").write_text(json.dumps({"results": results}, indent=2) + "\n")
    print(json.dumps({"repaired": len(results), "eligible": sum(x.get("sft_verdict") == "ELIGIBLE" for x in results)}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
