"""ARCHIVED: harness self-evolution A/B prototype on the 15-task suite.

Diagnose-first design (NOT a rigged baseline):
  - baseline  : NEUTRAL harness. The prompt states the task and output contract
                only — no recovery guidance. We observe the model's *natural*
                behaviour when the requested file does not exist.
  - candidate : the SAME harness plus ONE targeted policy change. The change is
                chosen from the failure slice that attribution surfaces in the
                baseline episodes (reported in the run summary).

Same model / environment / evaluator across arms — only HarnessManifest
.policy_flags vary — so the frozen regression gate attributes any outcome
delta to the harness policy alone.

Run on the server with the local OpenAI server up (PYTHONPATH=/root/agentic-rl):
  python3 experiments/local_model/self_evolve_ab.py \
      --output /root/rivermind-data/self-evolve-v1
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping as AbcMapping
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.analysis.compare import compare_runs
from src.analysis.metrics import compute_episode_metrics
from src.analysis.regression_gate import GateConfig, evaluate_gate
from src.assembly.episode_assembler import EpisodeAssembler, EpisodeContext
from src.capture.pi_adapter import (
    PiAdapterResult,
    PiJsonAdapter,
    PiRunConfig,
    dump_pi_ndjson,
    read_pi_ndjson,
)
from src.capture.pi_runner import run_pi_process
from src.contracts._json import sha256_json
from src.contracts.agent_episode import CaptureCapability
from src.contracts.experiment import ExperimentManifest
from src.contracts.manifests import (
    EnvironmentManifest,
    EvaluatorManifest,
    HarnessManifest,
    ModelManifest,
)
from src.real_pi_v2 import verify_reference_answer
from src.task_suite import (
    ALL_TASKS,
    expected_counts_map,
    extra_files,
    task_file_payload,
)

PROVIDER = "local-qwen"
# Overridable so the same A/B/eval machinery can point at a different served
# model (e.g. the 14B server) without code changes.
MODEL = os.environ.get("LOCAL_MODEL_NAME", "qwen2.5-7b-instruct")
EXPECTED = expected_counts_map(ALL_TASKS)
ARMS = ("baseline", "candidate")
RUN_IDS = {"baseline": "run-se-baseline", "candidate": "run-se-candidate"}
EXPERIMENT_ID = "experiment-harness-self-evolve-15task"

# --- harness policies ---------------------------------------------------------
# baseline: neutral. No recovery instruction — we diagnose the natural failure.
BASELINE_POLICY_FLAGS: dict[str, Any] = {}
# candidate (iteration 2): ONE targeted change aimed at the diagnosed slice
# (WRONG_COUNT — the model reads the file fine but miscounts the FAILURE
# records). The harness nudges precise counting via a tool, with a concrete
# example, rather than handing over a finished answer command: the model still
# discovers the file, chooses to run it, and interprets the output.
CANDIDATE_TOOLCOUNT = (
    " Harness policy: after reading the task file, do NOT count the records in "
    "your head. Count them with the bash tool, for example run: python3 -c "
    "\"import json;print(sum(1 for r in json.load(open('TASKFILE'))['records'] "
    "if r['status']=='FAILURE'))\" with TASKFILE replaced by the task file you "
    "found, then use the number it prints as failure_count."
)
CANDIDATE_POLICY_FLAGS: dict[str, Any] = {
    "counting": "tool_exact_count_no_mental_math",
    "count_tool": "bash",
    "maximum_identical_retries": 0,
}
TOOL_BUDGET = 5


def build_prompt(task_index: int, arm: str) -> str:
    base = (
        f"Start by reading missing-{task_index}.json. "
        "Return only JSON (no markdown, no code fences) with fields "
        "task_id, status, failure_count. "
        "status must be exactly the string SUCCESS when you have successfully "
        "read the task file, regardless of the records inside it. "
        "failure_count is the number of records whose status is exactly FAILURE; "
        "ignore SUCCESS, ERROR, UNKNOWN. "
        f"Use at most {TOOL_BUDGET} tool calls total."
    )
    if arm == "candidate":
        base += CANDIDATE_TOOLCOUNT
    return base


def policy_flags(arm: str) -> dict[str, Any]:
    return dict(BASELINE_POLICY_FLAGS if arm == "baseline" else CANDIDATE_POLICY_FLAGS)


def setup_workspace(ws_root: Path, task_index: int) -> Path:
    cwd = ws_root / f"task-{task_index}"
    cwd.mkdir(parents=True, exist_ok=True)
    for f in cwd.glob("*.json"):
        f.unlink()
    (cwd / f"task-{task_index}.json").write_text(
        json.dumps(task_file_payload(task_index), ensure_ascii=False) + "\n"
    )
    for name, body in extra_files(task_index).items():
        (cwd / name).write_text(body)
    return cwd


def _read_task_file_ok(records, task_index: int) -> bool:
    """Did the agent successfully read task-N.json? Robust to records whose
    fields were frozen into mappingproxy by secret redaction."""
    marker = f'"task_id": "task-{task_index}"'
    for rec in records:
        if rec.get("type") != "tool_execution_end" or rec.get("isError"):
            continue
        if rec.get("toolName") != "read":
            continue
        res = rec.get("result")
        try:
            content = res.get("content") if hasattr(res, "get") else None
            if isinstance(content, (list, tuple)):
                for part in content:
                    text = part.get("text") if hasattr(part, "get") else None
                    if isinstance(text, str) and marker in text:
                        return True
        except Exception:  # noqa: BLE001
            continue
    return False


def _final_count(records) -> int | None:
    """The failure_count in the model's final answer. Pi streams messages as
    start/update/end; the final answer is the last message_end whose message
    has role=assistant and stopReason=='stop' (intermediate tool-call messages
    stop with 'toolUse')."""
    final_text: str | None = None
    for rec in records:
        if rec.get("type") != "message_end":
            continue
        m = rec.get("message")
        # redaction freezes dicts into mappingproxy -> test Mapping, not dict
        if not isinstance(m, AbcMapping) or m.get("role") != "assistant":
            continue
        content = m.get("content")
        if isinstance(content, (list, tuple)):
            text = "".join(
                p.get("text", "") for p in content
                if isinstance(p, AbcMapping) and p.get("type") == "text"
            )
        else:
            text = content if isinstance(content, str) else ""
        if m.get("stopReason") == "stop":
            final_text = text
    if final_text is None:
        return None
    m2 = re.search(r'failure_count"?\s*[:=]\s*(\d+)', final_text)
    return int(m2.group(1)) if m2 else None


def _attribute(records, task_index: int, declaration) -> tuple[str, ...]:
    """Experiment-specific first-failure attribution (deliberately kept OUT of
    the released failure-attribution/v1 engine so it stays stable; promotable
    later). WRONG_COUNT = read the task file fine + produced an answer, but the
    count differs from expected -> the model miscounted (the diagnosed slice)."""
    if float(getattr(declaration, "score", 0.0)) == 1.0:
        return ()
    read_ok = _read_task_file_ok(records, task_index)
    count = _final_count(records)
    if read_ok and count is not None:
        return ("WRONG_COUNT",) if count != EXPECTED[task_index] else ("UNATTRIBUTED_TASK_FAILURE",)
    if read_ok:
        return ("NO_FINAL_ANSWER",)
    return ("RECOVERY_FAILURE",)


def run_episode(
    *,
    task_index: int,
    arm: str,
    workspace_root: Path,
    timeout_seconds: float,
    raw_output_dir: Path,
) -> PiAdapterResult:
    cwd = setup_workspace(workspace_root, task_index)
    # bash is enabled for BOTH arms so the tool environment is identical across
    # arms (the gate holds environment constant); only the prompt policy differs.
    config = PiRunConfig(
        model=MODEL, provider=PROVIDER, thinking="minimal",
        tools=("read", "grep", "find", "ls", "bash"),
    )
    capture = run_pi_process(
        config=config,
        prompt=build_prompt(task_index, arm),
        cwd=cwd,
        timeout_seconds=timeout_seconds,
    )
    records, issues = read_pi_ndjson(capture.stdout)
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    (raw_output_dir / f"{arm}-{task_index}.ndjson").write_text(dump_pi_ndjson(records))
    declaration, _ev = verify_reference_answer(
        records, task_index=task_index, expected_counts=EXPECTED
    )
    adapter = PiJsonAdapter().convert(
        records,
        run_id=RUN_IDS[arm],
        episode_id=f"episode-se-{arm}-{task_index}",
        trace_id=f"trace-se-{arm}-{task_index}",
        config=config,
        declared_outcome=declaration,
        source_issues=issues,
        normalize_tool_errors=True,
    )
    return adapter, records, declaration


def build_context(*, task_index: int, arm: str, experiment: ExperimentManifest, adapter: PiAdapterResult) -> EpisodeContext:
    flags = policy_flags(arm)
    return EpisodeContext(
        task_id=f"se/task-{task_index}",
        attempt=1,
        harness_manifest=HarnessManifest(
            name="pi",
            version=f"0.84.2-{arm}",
            revision=f"self-evolve-policy/{arm}/r1",
            config_digest=sha256_json(flags),
            policy_flags=flags,
            hook_version=None,
        ),
        model_manifest=ModelManifest(
            provider=PROVIDER,
            model_id=MODEL,
            revision="NOT_OBSERVABLE",
            sampling_config={"thinking": "minimal", "seed": "NOT_OBSERVABLE"},
            tokenizer_revision=None,
        ),
        environment_manifest=EnvironmentManifest(
            runtime_type="pi-local-process",
            revision="pi-self-evolve-env/v1",
            image=None,
            resource_limits={"timeout_seconds": 300},
            network_policy="model-provider-only",
            task_snapshot=experiment.task_dataset_revision,
        ),
        evaluator_manifest=EvaluatorManifest(
            name="exact-json-verifier",
            revision="self-evolve-verifier/v1",
            config_digest=sha256_json({"strict_keys": True, "expected_counts": EXPECTED}),
        ),
        experiment_manifest_ref=experiment.experiment_id,
        capabilities=adapter.capabilities | frozenset({CaptureCapability.VERIFIER_EVIDENCE}),
    )


def _experiment() -> ExperimentManifest:
    return ExperimentManifest(
        experiment_id=EXPERIMENT_ID,
        revision="r1",
        task_dataset_revision="pi-v3-15-task-suite/r1",
        control_run_id=RUN_IDS["baseline"],
        candidate_run_id=RUN_IDS["candidate"],
        target_policy_flag="WRONG_COUNT",
        minimum_pairs=3,
        metadata={
            "model": MODEL,
            "provider": PROVIDER,
            "pi_version": "0.84.2",
            "design": "neutral baseline vs one targeted harness policy change",
            "track": "harness self-evolution (model held fixed)",
        },
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v, ensure_ascii=False, sort_keys=True) + "\n" for v in values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--workspace", default="/root/rivermind-data/self-evolve-ws")
    parser.add_argument("--tasks", default=",".join(str(t) for t in ALL_TASKS))
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    output = Path(args.output)
    ws_root = Path(args.workspace)
    tasks = [int(t) for t in args.tasks.split(",")]
    arms = tuple(a for a in args.arms.split(","))
    experiment = _experiment()

    adapters: dict[str, PiAdapterResult] = {}
    contexts: dict[str, EpisodeContext] = {}
    raw: dict[str, tuple[list, Any, int]] = {}  # eid -> (records, declaration, task_index)
    for task_index in tasks:
        for arm in arms:
            eid = f"episode-se-{arm}-{task_index}"
            try:
                adapter, records, declaration = run_episode(
                    task_index=task_index, arm=arm, workspace_root=ws_root,
                    timeout_seconds=args.timeout, raw_output_dir=output / "raw-ndjson",
                )
                adapters[eid] = adapter
                raw[eid] = (records, declaration, task_index)
                contexts[eid] = build_context(
                    task_index=task_index, arm=arm, experiment=experiment, adapter=adapter
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[run] {eid} ERROR {type(exc).__name__}: {exc}", flush=True)
            print(f"[run] done {arm} task-{task_index}", flush=True)

    all_events = tuple(e for a in adapters.values() for e in a.events)
    assembly = EpisodeAssembler().assemble(all_events, contexts=contexts)
    if assembly.warnings or assembly.quarantined_events:
        # weak-model traces can be messy; keep them for diagnosis rather than abort
        print(f"[assemble] warnings={len(assembly.warnings)} quarantined={len(assembly.quarantined_events)}", flush=True)

    def episodes_of(run_id: str):
        return tuple(e for e in assembly.episodes if e.run_id == run_id)

    # domain-specific first-failure attribution (precise WRONG_COUNT slice),
    # keyed by episode_id for the gate's target-slice accounting
    reason_codes: dict[str, dict[str, tuple[str, ...]]] = {RUN_IDS[arm]: {} for arm in arms}
    for eid, (records, declaration, task_index) in raw.items():
        arm = "baseline" if "-baseline-" in eid else "candidate"
        reason_codes[RUN_IDS[arm]][eid] = _attribute(records, task_index, declaration)

    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "model": f"{PROVIDER}/{MODEL}",
        "tasks": tasks,
        "arms": list(arms),
        "per_arm_successes": {
            arm: sum(e.outcome.task_status.value == "SUCCESS" for e in episodes_of(RUN_IDS[arm]))
            for arm in arms
        },
        "per_arm_by_task": {
            arm: {
                e.task_id.split("/")[-1]: e.outcome.task_status.value for e in episodes_of(RUN_IDS[arm])
            }
            for arm in arms
        },
        "reason_code_tally": {
            arm: _tally(reason_codes[RUN_IDS[arm]].values()) for arm in arms
        },
    }

    if "baseline" in arms and "candidate" in arms:
        base_eps = episodes_of(RUN_IDS["baseline"])
        cand_eps = episodes_of(RUN_IDS["candidate"])
        comparison = compare_runs(
            base_eps, cand_eps,
            control_metrics=tuple(compute_episode_metrics(e) for e in base_eps),
            candidate_metrics=tuple(compute_episode_metrics(e) for e in cand_eps),
            experiment=experiment,
            control_reason_codes=reason_codes[RUN_IDS["baseline"]],
            candidate_reason_codes=reason_codes[RUN_IDS["candidate"]],
        )
        gate = evaluate_gate(comparison, GateConfig())
        _write_json(output / "comparison.json", comparison.to_dict())
        _write_json(output / "gate.json", gate.to_dict())
        summary["gate"] = {
            "decision": gate.decision.value,
            "paired_coverage": comparison.paired_coverage,
            "checks": [c.to_dict() for c in gate.checks],
            "control_success_rate": comparison.aggregate.control_success_rate,
            "candidate_success_rate": comparison.aggregate.candidate_success_rate,
        }

    _write_json(output / "experiment-manifest.json", experiment.to_dict())
    for arm in arms:
        _write_jsonl(output / f"episodes-{RUN_IDS[arm]}.jsonl", (e.to_dict() for e in episodes_of(RUN_IDS[arm])))
    _write_json(output / "reason-codes.json", {k: {e: list(v) for e, v in vv.items()} for k, vv in reason_codes.items()})
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _tally(code_groups) -> dict[str, int]:
    tally: dict[str, int] = {}
    for group in code_groups:
        for code in group:
            tally[code] = tally.get(code, 0) + 1
    return tally


if __name__ == "__main__":
    main()
