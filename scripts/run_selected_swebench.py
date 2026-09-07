#!/usr/bin/env python3
"""Run a selected SWE-bench subset through certified Pi host executions.

The script does not claim the official SWE-bench Docker evaluator.  Its
evaluator is explicitly the repository's isolated test-patch verifier, pinned
per selected task.  Every attempted run still becomes an immutable bundle;
only verifier-valid runs can later be admitted to learning datasets.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import traceback
from pathlib import Path

from src.contracts._json import sha256_json
from src.contracts.execution_identity import ExecutionIdentity
from src.contracts.manifests import EvaluatorManifest, HarnessManifest
from src.orchestration import PiHostExecutionOrchestrator, PiHostExecutionSpec


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--venv-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pi", required=True)
    parser.add_argument("--upstream-url")
    parser.add_argument("--provider", default="local-openai-compatible")
    parser.add_argument("--provider-api", default="openai-completions")
    parser.add_argument("--model", default="qwen2.5-coder-14b-instruct")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--policies", type=Path, default=Path("configs/harness-policies.json"))
    parser.add_argument("--policy", default="baseline-v1")
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=420)
    parser.add_argument(
        "--tools", default="read,bash,write,edit,ls",
        help="comma-separated Pi tools; omit rg/fd-backed tools on minimal hosts",
    )
    parser.add_argument(
        "--allow-response-only", action="store_true",
        help="allow SFT Teacher runs without token-level RL evidence",
    )
    return parser.parse_args()


def workspace_name(instance_id: str) -> str:
    repository, issue = instance_id.rsplit("-", 1)
    owner, name = repository.split("__", 1)
    # Keep the existing Requests naming (owner helps distinguish its task
    # directories), while common upstream project names stay compact.
    prefix = name if owner in {"pallets", "pytest-dev", "pylint-dev", "sympy"} else f"{owner}-{name}"
    return f"{prefix}-{issue}"


def main() -> int:
    args = arguments()
    # Verifier subprocesses run inside isolated evaluation worktrees. Resolve
    # every host path before building their commands so a caller's relative
    # --venv-root cannot become invalid after the verifier changes cwd.
    args.selected = args.selected.resolve()
    args.workspace_root = args.workspace_root.resolve()
    args.venv_root = args.venv_root.resolve()
    args.output_root = args.output_root.resolve()
    args.policies = args.policies.resolve()
    selected = json.loads(args.selected.read_text())
    policies = json.loads(args.policies.read_text())["policies"]
    policy = policies[args.policy]
    instance_ids = args.instances or sorted(selected)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for instance_id in instance_ids:
        task = selected[instance_id]
        workspace = args.workspace_root / workspace_name(instance_id)
        python = args.venv_root / workspace_name(instance_id) / "bin" / "python"
        if not workspace.is_dir() or not python.is_file():
            results.append({"instance_id": instance_id, "status": "SKIPPED", "reason": "workspace_or_venv_missing"})
            continue
        subprocess.run(["git", "-C", str(workspace), "reset", "--hard", task["base_commit"]], check=True)
        subprocess.run(["git", "-C", str(workspace), "clean", "-fdq"], check=True)
        policy_fingerprint = sha256_json({"harness": "pi", "policy_name": args.policy, "policy": policy,
                                          "provider": args.provider, "model": args.model,
                                          "model_revision": args.model_revision})
        tool_names = [item.strip() for item in args.tools.split(",") if item.strip()]
        if not tool_names:
            raise ValueError("--tools must contain at least one tool")
        sampling = {"thinking": "minimal", "tools": tool_names}
        identity = ExecutionIdentity(run_id=f"swe-{instance_id}-{args.policy}-{args.attempt}", task_id=instance_id,
            episode_id=f"episode-swe-{instance_id}-{args.policy}-{args.attempt}", attempt_id=args.attempt,
            producer_id="pi-teacher" if args.upstream_url is None else "pi-host",
            producer_version="pi-teacher/v1" if args.upstream_url is None else "pi-host/v1",
            group_id=f"swe-selected:{args.policy}",
            policy_fingerprint=policy_fingerprint, sampling_fingerprint=sha256_json(sampling))
        prompt = "You are fixing a real repository task.\n\nIssue:\n" + task["problem_statement"] + "\n\nExecution policy:\n- " + "\n- ".join(policy["instructions"])
        verifier = (str(python), "scripts/verify_swebench_selected.py", "--selected", str(args.selected.resolve()),
                    "--instance-id", instance_id, "--source-worktree", str(workspace.resolve()), "--python", str(python),
                    "--output", str((args.output_root / identity.run_id / "verifier-output.json").resolve()))
        spec = PiHostExecutionSpec(identity=identity, pi=args.pi, workspace=str(workspace), prompt=prompt,
            tools=tuple(sampling["tools"]), timeout_seconds=args.timeout, upstream_url=args.upstream_url,
            provider=args.provider, provider_api=args.provider_api, model=args.model,
            model_revision=args.model_revision, harness_manifest=HarnessManifest(name="pi", version="0.84.2", revision="pi-json/v0.84.2",
                config_digest=sha256_json({"tools": sampling["tools"], "policy": policy}), policy_flags={"policy_name": args.policy, "policy": policy}),
            evaluator_manifest=EvaluatorManifest(name="selected-swebench-isolated-test-patch", revision="verify-swebench-selected/v7",
                config_digest=sha256_json({"selected": selected[instance_id], "python": str(python),
                    "candidate_patch_policy": "exclude-evaluator-controlled-test-files/v1",
                    "lossy_selector_policy": "conservative-function-expansion/v1",
                    "bare_selector_policy": "definition-scan-then-pytest-collection/v1",
                "temporary_directory_policy": "per-verifier-group/v1",
                    "python_compatibility_policy": "stdlib-collections-abc-aliases/v1",
                    "environment_materialization_policy": "allowlisted-generated-files/v1"})), verifier_command=verifier,
            task_snapshot=task["base_commit"], experiment_manifest_ref=f"selected-swebench:{args.selected.resolve()}",
            sampling_config=sampling, require_rl_evidence=not args.allow_response_only)
        try:
            results.append(PiHostExecutionOrchestrator().run(spec, output_dir=args.output_root / identity.run_id))
        except Exception as exc:
            diagnostic = {
                "instance_id": instance_id,
                "status": "ERROR",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            (args.output_root / identity.run_id / "execution-error.json").write_text(
                json.dumps(diagnostic, indent=2) + "\n"
            )
            results.append(diagnostic)
    rendered = {"policy": args.policy, "instances": instance_ids, "results": results}
    batch_key = sha256_json({"policy": args.policy, "attempt": args.attempt, "instances": instance_ids})[:12]
    (args.output_root / f"batch-{args.policy}-{args.attempt}-{batch_key}.json").write_text(
        json.dumps(rendered, indent=2) + "\n"
    )
    print(json.dumps(rendered, indent=2))
    return 0 if all(item.get("status") != "ERROR" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
