#!/usr/bin/env python3
"""Single-GPU Time-sliced Agentic RL Cycle Orchestrator.

State Machine:
1. START_SERVING: Launch inference server with current policy adapter.
2. ROLLOUT: Collect G rollouts per task; certify into Slime admission batch.
3. STOP_SERVING: Terminate inference server; ensure GPU VRAM is released.
4. TRAIN: Execute GRPO update to produce updated policy adapter.
5. EVAL_AND_GATE: Evaluate new adapter against fixed DEV holdout gate; promote or reject.

Real mode: this orchestrator only wires the cycle end-to-end in mock mode. Real
training runs scripts/train_grpo_lora.py on CUDA; real gating runs
scripts/evaluate_apps_holdout.py against the fixed holdout. This orchestrator
never fabricates gate numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class RLCycleOrchestrator:
    def __init__(
        self,
        *,
        cycle_id: str,
        tasks: list[str],
        eval_tasks: list[str],
        group_size: int,
        output_dir: Path,
        current_adapter: str | None = None,
        mock: bool = False,
    ) -> None:
        self.cycle_id = cycle_id
        self.tasks = tasks
        self.eval_tasks = eval_tasks
        self.group_size = group_size
        self.output_dir = output_dir
        self.current_adapter = current_adapter
        self.mock = mock
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / "cycle.log"

    def log(self, message: str) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = f"[{ts}] {message}"
        print(line, flush=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def step_rollout(self, rollouts_dir: Path, policy_fingerprint: str) -> dict[str, Any]:
        self.log(f"[Phase 1: Rollout] Collecting G={self.group_size} rollouts on {len(self.tasks)} tasks...")
        from scripts.collect_rl_rollouts import collect_group_rollouts

        summary = collect_group_rollouts(
            tasks=self.tasks,
            group_size=self.group_size,
            output_dir=rollouts_dir,
            run_id=f"run-{self.cycle_id}",
            policy_fingerprint=policy_fingerprint,
            mock=self.mock,
        )
        self.log(
            f"[Phase 1: Rollout] Done. Admitted traces={summary['admitted_traces']}, "
            f"admission={summary['admission_checksum'][:16]}..."
        )
        return summary

    def step_train(self, admission_file: Path, adapter_out: Path) -> dict[str, Any]:
        self.log("[Phase 2: Train] Executing GRPO policy optimization...")
        from scripts.train_grpo_lora import mock_grpo_train, _verify_admission

        admission_data = _verify_admission(admission_file)
        if self.mock:
            record = mock_grpo_train(admission_data, adapter_out, epochs=1, lr=1e-5)
        else:
            raise NotImplementedError("Real GPU training requires running train_grpo_lora.py directly on CUDA.")

        self.log(
            f"[Phase 2: Train] Done. Loss={record['final_mean_loss']}, "
            f"New Fingerprint={record['policy_fingerprint_out'][:16]}..."
        )
        return record

    def step_eval_gate(self, candidate_adapter: Path) -> dict[str, Any]:
        self.log("[Phase 3: Eval & Gate] Evaluating candidate on fixed unseen DEV holdout...")
        if not self.mock:
            raise NotImplementedError(
                "Real eval/gate is not performed by this orchestrator: run "
                "scripts/evaluate_apps_holdout.py against the fixed holdout and judge "
                "from its summary. This orchestrator does not fabricate gate numbers."
            )
        # Mock wiring test only: explicitly simulated pass-rate comparison.
        base_pass_rate = 0.50
        candidate_pass_rate = 0.65  # Simulated improvement
        improved = candidate_pass_rate > base_pass_rate

        verdict = "PROMOTED" if improved else "REJECTED"
        gate_report = {
            "mode": "mock-simulated",
            "candidate_adapter": str(candidate_adapter),
            "dev_tasks_count": len(self.eval_tasks),
            "base_pass_rate": base_pass_rate,
            "candidate_pass_rate": candidate_pass_rate,
            "delta_pp": round((candidate_pass_rate - base_pass_rate) * 100, 2),
            "verdict": verdict,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (self.output_dir / "gate-report.json").write_text(
            json.dumps(gate_report, indent=2, ensure_ascii=False) + "\n"
        )
        self.log(f"[Phase 3: Eval & Gate] Result: {verdict} (delta: +{gate_report['delta_pp']} pp)")
        return gate_report

    def run_full_cycle(self) -> dict[str, Any]:
        self.log(f"=== Starting RL Cycle {self.cycle_id} ===")
        rollouts_dir = self.output_dir / "rollouts"
        candidate_adapter_dir = self.output_dir / "candidate_policy_adapter"

        initial_fingerprint = "0" * 64

        # 1. Rollout
        rollout_summary = self.step_rollout(rollouts_dir, initial_fingerprint)
        admission_file = rollouts_dir / "slime-admission.json"

        # 2. Train
        train_record = self.step_train(admission_file, candidate_adapter_dir)

        # 3. Gate
        gate_report = self.step_eval_gate(candidate_adapter_dir)

        cycle_summary = {
            "cycle_id": self.cycle_id,
            "tasks": self.tasks,
            "eval_tasks": self.eval_tasks,
            "group_size": self.group_size,
            "rollouts": rollout_summary,
            "train": train_record,
            "gate": gate_report,
            "status": "MOCK_COMPLETED" if self.mock else "COMPLETED",
        }
        (self.output_dir / "cycle-summary.json").write_text(
            json.dumps(cycle_summary, indent=2, ensure_ascii=False) + "\n"
        )
        mode_label = "completed (mock)" if self.mock else "completed"
        self.log(f"=== Cycle {self.cycle_id} {mode_label} with verdict {gate_report['verdict']} ===")
        return cycle_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-id", default="cycle-001", help="Cycle identifier")
    parser.add_argument("--tasks", nargs="+", default=["apps-train-1", "apps-train-2"], help="Train tasks")
    parser.add_argument("--eval-tasks", nargs="+", default=["dev-holdout-1", "dev-holdout-2"], help="DEV holdout tasks")
    parser.add_argument("--group-size", type=int, default=4, help="Rollout group size G")
    parser.add_argument("--output-dir", required=True, help="Cycle workspace root")
    parser.add_argument("--mock", action="store_true", help="Execute complete cycle in mock mode")
    args = parser.parse_args()

    orchestrator = RLCycleOrchestrator(
        cycle_id=args.cycle_id,
        tasks=args.tasks,
        eval_tasks=args.eval_tasks,
        group_size=args.group_size,
        output_dir=Path(args.output_dir),
        mock=args.mock,
    )
    result = orchestrator.run_full_cycle()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
