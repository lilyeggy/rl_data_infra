#!/usr/bin/env python3
"""Run real on-policy APPS group rollouts for Agentic RL.

Loads Policy-v0 (Qwen2.5-Coder-14B + clean-v2 adapter) on CUDA,
performs G=4 rollouts for each selected APPS task, extracts token IDs,
action mask, native behavior logprobs, runs verify_apps.py,
assembles ExecutionBundle, certifies for ON_POLICY_RL, and exports
slime-admission.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.assembly import assemble_execution_bundle
from src.assembly.episode_assembler import EpisodeAssembler, EpisodeContext
from src.certification import ConsumerProfile, ConsumerVerdict, certify_for
from src.contracts._json import canonical_json_bytes, sha256_json
from src.contracts.agent_episode import (
    AgentEpisode,
    CaptureCapability,
    EpisodeVerifierStatus,
    ExecutionValidity,
    TaskStatus,
)
from src.contracts.dataset import DatasetManifest, DatasetPurpose, DatasetRole, DatasetSplit
from src.contracts.execution_bundle import ExecutionBundle
from src.contracts.execution_identity import ExecutionIdentity
from src.contracts.manifests import (
    EnvironmentManifest,
    EvaluatorManifest,
    HarnessManifest,
    ModelManifest,
)
from src.contracts.trace_event import (
    EventComponent,
    EventStatus,
    EventType,
    TraceEvent,
)
from src.integrations.slime import admit_on_policy_manifest
from src.learning import CertifiedArtifact, compile_dataset
from src.producers.base import (
    ProducerArtifact,
    ProducerCapability,
    ProducerExecutionStatus,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_python_code(text: str) -> str:
    """Extract python code from model output."""
    pattern = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip() + "\n"
    return text.strip() + "\n"


def make_events_for_rollout(
    *,
    run_id: str,
    episode_id: str,
    trace_id: str,
    prompt: str,
    response_text: str,
    passed: bool,
    input_tokens: int,
    output_tokens: int,
) -> tuple[TraceEvent, ...]:
    return (
        TraceEvent(
            event_id=f"evt-{episode_id}-req",
            run_id=run_id,
            episode_id=episode_id,
            trace_id=trace_id,
            span_id="span-model",
            parent_span_id=None,
            sequence=0,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            event_type=EventType.MODEL_REQUEST,
            component=EventComponent.MODEL,
            status=EventStatus.STARTED,
            attempt=1,
            attributes={"messages": [{"role": "user", "content": prompt}], "input_tokens": input_tokens},
            artifact_refs=(),
        ),
        TraceEvent(
            event_id=f"evt-{episode_id}-resp",
            run_id=run_id,
            episode_id=episode_id,
            trace_id=trace_id,
            span_id="span-model",
            parent_span_id=None,
            sequence=1,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            event_type=EventType.MODEL_RESPONSE,
            component=EventComponent.MODEL_BACKEND,
            status=EventStatus.SUCCEEDED,
            attempt=1,
            attributes={
                "content": [{"type": "text", "text": response_text}],
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            },
            artifact_refs=(),
        ),
        TraceEvent(
            event_id=f"evt-{episode_id}-verif-start",
            run_id=run_id,
            episode_id=episode_id,
            trace_id=trace_id,
            span_id="span-eval",
            parent_span_id=None,
            sequence=2,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            event_type=EventType.VERIFICATION_STARTED,
            component=EventComponent.EVALUATOR,
            status=EventStatus.STARTED,
            attempt=1,
            attributes={},
            artifact_refs=(),
        ),
        TraceEvent(
            event_id=f"evt-{episode_id}-verif-end",
            run_id=run_id,
            episode_id=episode_id,
            trace_id=trace_id,
            span_id="span-eval",
            parent_span_id=None,
            sequence=3,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            event_type=EventType.VERIFICATION_FINISHED,
            component=EventComponent.EVALUATOR,
            status=EventStatus.SUCCEEDED if passed else EventStatus.FAILED,
            attempt=1,
            attributes={"passed": passed},
            artifact_refs=(),
        ),
        TraceEvent(
            event_id=f"evt-{episode_id}-finished",
            run_id=run_id,
            episode_id=episode_id,
            trace_id=trace_id,
            span_id="span-episode",
            parent_span_id=None,
            sequence=4,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            event_type=EventType.EPISODE_FINISHED,
            component=EventComponent.HARNESS,
            status=EventStatus.SUCCEEDED,
            attempt=1,
            attributes={
                "task_status": "SUCCESS" if passed else "FAILURE",
                "execution_validity": "VALID",
                "verifier_status": "PASSED" if passed else "FAILED",
                "score": 1.0 if passed else 0.0,
                "termination_reason": "VERIFIER_PASSED" if passed else "VERIFIER_FAILED",
                "evidence_event_ids": [f"evt-{episode_id}-verif-end"],
            },
            artifact_refs=(),
        ),
    )


def run_apps_rollouts(
    *,
    manifest_path: Path,
    task_ids: list[str],
    model_path: str,
    adapter_path: str | None,
    output_dir: Path,
    group_size: int = 4,
    device: str = "cuda:0",
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 1536,
    python_bin: str = sys.executable,
) -> dict[str, Any]:
    sys.set_int_max_str_digits(0)
    output_dir.mkdir(parents=True, exist_ok=True)
    rollouts_dir = output_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    workspaces_dir = output_dir / "workspaces"
    workspaces_dir.mkdir(parents=True, exist_ok=True)

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest_data["tasks"]

    # Compute policy fingerprint
    policy_info = {"model": model_path, "adapter": adapter_path or "base"}
    policy_fingerprint = sha256_json(policy_info)
    sampling_info = {"temperature": temperature, "top_p": top_p, "max_new_tokens": max_new_tokens}
    sampling_fingerprint = sha256_json(sampling_info)

    print(f"[init] Loading tokenizer from {model_path}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[init] Loading base model {model_path} on {device}...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)

    if adapter_path and Path(adapter_path).exists():
        print(f"[init] Attaching policy adapter {adapter_path}...", flush=True)
        policy = PeftModel.from_pretrained(base, adapter_path).to(device)
    else:
        print("[init] Using base model as policy...", flush=True)
        policy = base

    policy.eval()

    run_id = f"rl-run-{int(time.time())}"
    assembler = EpisodeAssembler()

    artifacts_by_cs: dict[str, ProducerArtifact] = {}
    bundles_by_cs: dict[str, ExecutionBundle] = {}
    decisions_by_cs: dict[str, Any] = {}
    certified_entries: list[CertifiedArtifact] = []

    rollout_records: list[dict[str, Any]] = []

    for task_idx, task_id in enumerate(task_ids):
        if task_id not in tasks:
            print(f"[warn] task_id {task_id} not in manifest, skipping", flush=True)
            continue
        task_data = tasks[task_id]
        question = task_data["question"]
        group_id = f"group-{run_id}-{task_id}"

        prompt = (
            "<|im_start|>system\nYou are an expert competitive programmer. Implement the requested solution in python3. "
            "The program must read input from standard input (sys.stdin) and write the answer to standard output (sys.stdout). "
            "Output only the executable Python code inside a ```python ``` codeblock without unnecessary chatter.<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        print(f"\n[{task_idx+1}/{len(task_ids)}] Task {task_id} (G={group_size} rollouts):", flush=True)

        # Batched generation for all G attempts concurrently
        inputs = tok([prompt] * group_size, return_tensors="pt").to(device)
        prompt_len = inputs.input_ids.shape[1]
        prompt_ids = list(inputs.input_ids[0].cpu().numpy().tolist())

        t0 = time.time()
        with torch.no_grad():
            out = policy.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tok.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        gen_time = time.time() - t0
        scores = out.scores

        for attempt in range(1, group_size + 1):
            idx = attempt - 1
            episode_id = f"ep-{task_id}-{attempt}"
            trace_id = f"trace-{episode_id}"
            ws = workspaces_dir / f"{task_id}-attempt-{attempt}"
            ws.mkdir(parents=True, exist_ok=True)

            full_seq = out.sequences[idx][prompt_len:].cpu().numpy().tolist()
            if tok.eos_token_id in full_seq:
                eos_idx = full_seq.index(tok.eos_token_id)
                resp_ids = full_seq[: eos_idx + 1]
            else:
                resp_ids = full_seq

            logprobs = []
            for t_step, tid in enumerate(resp_ids):
                lp = torch.log_softmax(scores[t_step][idx].float(), dim=-1)[tid].item()
                logprobs.append(round(float(lp), 4))

            resp_text = tok.decode(resp_ids, skip_special_tokens=True)
            code = extract_python_code(resp_text)
            (ws / "solution.py").write_text(code, encoding="utf-8")

            # Run verifier
            verifier_out = ws / "verifier-output.json"
            verify_script = _REPO_ROOT / "scripts" / "verify_apps.py"
            verifier_cmd = [
                python_bin,
                str(verify_script),
                "--manifest",
                str(manifest_path),
                "--task-id",
                task_id,
                "--source-worktree",
                str(ws),
                "--python",
                python_bin,
                "--output",
                str(verifier_out),
            ]
            v_res = subprocess.run(verifier_cmd, capture_output=True, text=True)
            passed = False
            verifier_report: dict[str, Any] = {}
            if verifier_out.exists():
                try:
                    verifier_report = json.loads(verifier_out.read_text(encoding="utf-8"))
                    passed = bool(verifier_report.get("resolved", False))
                except Exception as exc:
                    print(f"    [error] reading verifier output: {exc}")

            reward = 1.0 if passed else 0.0
            print(
                f"  -> Attempt {attempt}/{group_size}: "
                f"tokens={len(resp_ids)} ({gen_time:.1f}s), "
                f"passed={passed} (reward={reward})",
                flush=True,
            )

            # Assemble execution artifacts
            identity = ExecutionIdentity(
                run_id=run_id,
                task_id=task_id,
                episode_id=episode_id,
                attempt_id=attempt,
                producer_id="pro6000_gpu",
                producer_version="pro6000-qwen14b/v1",
                group_id=group_id,
                policy_fingerprint=policy_fingerprint,
                sampling_fingerprint=sampling_fingerprint,
            )

            context = EpisodeContext(
                task_id=task_id,
                attempt=attempt,
                harness_manifest=HarnessManifest(
                    name="apps-harness",
                    version="1.0.0",
                    revision="apps-stdin/v1",
                    config_digest=sha256_json({"sampling": sampling_info}),
                ),
                model_manifest=ModelManifest(
                    provider="local-pytorch",
                    model_id="qwen2.5-coder-14b-instruct",
                    revision=policy_fingerprint[:16],
                    sampling_config=sampling_info,
                ),
                environment_manifest=EnvironmentManifest(
                    runtime_type="host-worktree",
                    revision="pro6000-blackwell",
                    image=None,
                    resource_limits={"timeout_seconds": 60},
                    network_policy="disabled",
                    task_snapshot=sha256_json(task_data),
                ),
                evaluator_manifest=EvaluatorManifest(
                    name="apps-verifier",
                    revision="v1",
                    config_digest=sha256_json({"task_id": task_id}),
                ),
                experiment_manifest_ref=f"apps:{manifest_path.name}",
                capabilities=frozenset({
                    CaptureCapability.MODEL_IO,
                    CaptureCapability.MODEL_TOKEN_USAGE,
                    CaptureCapability.TOOL_IO,
                    CaptureCapability.VERIFIER_EVIDENCE,
                }),
            )

            events = make_events_for_rollout(
                run_id=run_id,
                episode_id=episode_id,
                trace_id=trace_id,
                prompt=prompt,
                response_text=resp_text,
                passed=passed,
                input_tokens=prompt_len,
                output_tokens=len(resp_ids),
            )

            assembled_batch = assembler.assemble(events, contexts={episode_id: context})
            episode = assembled_batch.episodes[0]

            trace = {
                "prompt_ids": prompt_ids,
                "response_ids": resp_ids,
                "loss_mask": [1] * len(resp_ids),
                "response_logprobs": logprobs,
                "reward": reward,
            }

            artifact = ProducerArtifact(
                identity=identity,
                status=ProducerExecutionStatus.COMPLETED,
                capabilities=frozenset({
                    ProducerCapability.TOKEN_IDS,
                    ProducerCapability.ACTION_MASK,
                    ProducerCapability.BEHAVIOR_LOGPROBS,
                    ProducerCapability.POLICY_VERSION,
                    ProducerCapability.VERIFIER_EVIDENCE,
                }),
                payload={"trajectory": {"traces": [trace]}},
            )

            verifier_cs = _sha256(canonical_json_bytes(verifier_report).decode("utf-8"))
            bundle = assemble_execution_bundle(
                identity=identity,
                episode=episode,
                producer_artifacts=(artifact,),
                verifier_report_checksum=verifier_cs,
            )

            decision = certify_for(
                episode,
                ConsumerProfile.ON_POLICY_RL,
                execution_bundle=bundle,
                policy_artifact=artifact,
                target_policy_fingerprint=policy_fingerprint,
            )

            artifacts_by_cs[artifact.checksum] = artifact
            bundles_by_cs[bundle.checksum] = bundle
            decisions_by_cs[decision.checksum] = decision

            if decision.verdict is ConsumerVerdict.ELIGIBLE:
                certified_entries.append(
                    CertifiedArtifact(
                        identity=identity,
                        decision=decision,
                        artifact_checksum=artifact.checksum,
                        split=DatasetSplit.TRAIN,
                        role=DatasetRole.TRAJECTORY,
                    )
                )

            rollout_records.append({
                "task_id": task_id,
                "attempt": attempt,
                "passed": passed,
                "tokens": len(resp_ids),
                "decision": decision.verdict.name,
                "artifact_checksum": artifact.checksum[:16],
            })

    # Write out data plane artifacts
    with open(rollouts_dir / "producer-artifacts.jsonl", "w", encoding="utf-8") as f:
        for a in artifacts_by_cs.values():
            f.write(json.dumps(a.to_dict(), ensure_ascii=False) + "\n")

    with open(rollouts_dir / "execution-bundles.jsonl", "w", encoding="utf-8") as f:
        for b in bundles_by_cs.values():
            f.write(json.dumps(b.to_dict(), ensure_ascii=False) + "\n")

    with open(rollouts_dir / "eligibility-decisions.jsonl", "w", encoding="utf-8") as f:
        for d in decisions_by_cs.values():
            f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")

    manifest = compile_dataset(
        dataset_id=f"rl-apps-{run_id}",
        revision="r1",
        purpose=DatasetPurpose.ON_POLICY_RL,
        selection_policy_version="on-policy-rl/v1",
        artifacts=tuple(certified_entries),
    )
    (rollouts_dir / "dataset-manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n"
    )

    admission_batch = admit_on_policy_manifest(
        manifest,
        decisions_by_checksum=decisions_by_cs,
        bundles_by_checksum=bundles_by_cs,
        artifacts_by_checksum=artifacts_by_cs,
        minimum_group_size=min(2, group_size),
    )
    admission_file = rollouts_dir / "slime-admission.json"
    admission_file.write_text(
        json.dumps(admission_batch.to_dict(), indent=2, ensure_ascii=False) + "\n"
    )

    summary = {
        "run_id": run_id,
        "tasks": task_ids,
        "group_size": group_size,
        "total_rollouts": len(rollout_records),
        "certified_eligible": len(certified_entries),
        "admitted_traces": len(admission_batch.traces),
        "policy_fingerprint": policy_fingerprint,
        "admission_checksum": admission_batch.checksum,
        "admission_file": str(admission_file),
        "records": rollout_records,
    }
    (output_dir / "rollout-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"\n[done] Rollouts completed. Admitted traces={len(admission_batch.traces)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_apps_rollouts(
        manifest_path=args.manifest,
        task_ids=args.tasks,
        model_path=args.model,
        adapter_path=args.adapter,
        output_dir=args.output_dir,
        group_size=args.group_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
