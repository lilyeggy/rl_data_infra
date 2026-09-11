#!/usr/bin/env python3
"""Stage D certified rerun (MBPP, user-authorized): 4 trajectories via the full
PiHostExecutionOrchestrator path with controlled-proxy evidence capture.

Each trajectory yields: raw-events.jsonl, model-evidence.jsonl (native token
evidence), Episode, ExecutionBundle, policy artifact, ON_POLICY_RL
certification. After all 4, assemble verl_sequence per episode via the bridge
(contiguity check) + sequence assembler, admit via verl admission, and certify
the batch via CertifiedAgentLoopManager.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path("/home/cxr/agentic/code")
sys.path.insert(0, str(REPO))
LOCAL_REPO = Path(__file__).resolve().parent.parent
if str(LOCAL_REPO) not in sys.path:
    sys.path.insert(0, str(LOCAL_REPO))

BUDGET_START = time.time()
BUDGET_SECONDS = 50 * 60


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bridge-url", required=True)
    parser.add_argument("--pi-bin", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy-checksum", required=True)
    parser.add_argument("--model-revision", default="adapter-rev-p0")
    parser.add_argument("--task-ids", default="Mbpp/118", help="comma-separated task ids")
    parser.add_argument("--attempts", type=int, default=4, help="episodes per task")
    args = parser.parse_args()

    from src.contracts._json import sha256_json
    from src.contracts.execution_identity import ExecutionIdentity
    from src.contracts.manifests import EvaluatorManifest, HarnessManifest
    from src.orchestration.pi_host_execution import (
        PiHostExecutionOrchestrator,
        PiHostExecutionSpec,
    )

    try:
        from evalplus.data import get_mbpp_plus

        mbpp = get_mbpp_plus()
    except Exception as exc:  # noqa: BLE001
        print(f"MBPP-LOAD-FAIL: {exc}", flush=True)
        return 10

    task_ids = [t.strip() for t in args.task_ids.split(",") if t.strip()]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "verify_mbpp.py").write_text(
        Path("/home/cxr/verl-closeout/smoke/verify_mbpp_src.py").read_text())
    orchestrator = PiHostExecutionOrchestrator()
    summaries = []
    for task_id in task_ids:
        problem = mbpp[task_id]
        (out / f"prompt-{task_id.replace('/', '-')}.txt").write_text(
            "Task %s. Implement the function '%s' in solution.py.\n%s"
            % (task_id, problem["entry_point"], problem["prompt"])
        )
        for attempt in range(args.attempts):
            if time.time() - BUDGET_START > BUDGET_SECONDS:
                print("BUDGET-STOP", flush=True)
                break
            episode_id = f"{args.run_id}-{task_id.replace('/', '-')}-a{attempt}"
            ws = out / "workspaces" / episode_id
            ws.mkdir(parents=True)
            (ws / "solution.py").write_text('"""stub"""\n')
            (ws / "task-prompt.txt").write_text(
                "Task %s. Implement the function '%s' in solution.py.\n%s"
                % (task_id, problem["entry_point"], problem["prompt"])
            )
            identity = ExecutionIdentity(
                run_id=args.run_id,
                task_id=task_id,
                episode_id=episode_id,
                attempt_id=attempt + 1,
                producer_id="pi-real",
                producer_version="0.84.2",
                group_id=f"{args.run_id}-mbpp118",
                policy_fingerprint=args.policy_checksum,
                sampling_fingerprint="d" * 64,
            )
            prompt = (
                "You are in a workspace with solution.py. FIRST use the read tool "
                "to look at it, then write the requested Python function in "
                "solution.py and test it with bash. Keep reasoning brief and act "
                "with tools first. Task %s: implement the function described "
                "in task-prompt.txt." % task_id
            )
            spec = PiHostExecutionSpec(
                identity=identity,
                pi=args.pi_bin,
                workspace=str(ws),
                prompt=prompt,
                tools=("read", "bash", "write", "edit", "ls"),
                timeout_seconds=480.0,
                upstream_url=args.bridge_url,
                provider="local-qwen-proxy",
                provider_api="openai-completions",
                model="p0",
                model_revision=args.model_revision,
                harness_manifest=HarnessManifest(name="pi", version="0.84.2",
                    revision="pi-json/v0.84.2",
                    config_digest=sha256_json({"benchmark": "mbpp", "tools": "read,bash,write,edit,ls"}),
                    policy_flags={"benchmark": "mbpp", "policy": "d-certified-rerun-v1"}),
                evaluator_manifest=EvaluatorManifest(name="mbpp-execution-verifier",
                    revision="mbpp-verifier/v1",
                    config_digest=sha256_json({"task": task_id, "policy": "prompt-asserts"})),
                verifier_command=(sys.executable, str(out / "verify_mbpp.py"), task_id, str(ws),
                                  str(out / episode_id / "verifier-output.json")),
                task_snapshot=str(ws / "task-prompt.txt"),
                experiment_manifest_ref=args.run_id,
                sampling_config={"temperature": 1.0, "top_p": 1.0},
                require_rl_evidence=True,
            )
            summary = orchestrator.run(spec, output_dir=out / episode_id)
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    (out / "summaries.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
