#!/usr/bin/env python3
"""Staged entry point for the verl integration closeout.

Stages: preflight / smoke / rollout / cycle / evaluate / report / cleanup.

Default is preflight only (CPU, no GPU). GPU stages (smoke/rollout/cycle/
evaluate) require explicit --frozen-input, --gpu-uuids, and --deadline, and
refuse to run without them. Nothing here occupies a GPU by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

VERL_PIN_TAG = "v0.7.1"
VERL_PIN_COMMIT = "bec9ef74768dd201881cd4e54cd0385e87caae27"

STAGES = ("preflight", "smoke", "rollout", "cycle", "evaluate", "report", "cleanup")
GPU_STAGES = ("smoke", "rollout", "cycle", "evaluate")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cmd_preflight(args: argparse.Namespace) -> int:
    """CPU-only checks: imports, frozen-input manifest, config loading."""
    from src.integrations import verl as verl_bridge
    from src.training.policy_fingerprint import POLICY_FINGERPRINT_VERSION

    manifest_path = Path(args.frozen_input) if args.frozen_input else None
    report: dict = {
        "stage": "preflight",
        "verl_pin": {"tag": VERL_PIN_TAG, "commit": VERL_PIN_COMMIT},
        "bridge_modules": sorted(verl_bridge.__all__),
        "policy_fingerprint_version": POLICY_FINGERPRINT_VERSION,
    }
    if manifest_path is not None:
        if not manifest_path.exists():
            print(f"frozen input manifest not found: {manifest_path}", file=sys.stderr)
            return 2
        manifest = json.loads(manifest_path.read_text())
        report["frozen_input_manifest"] = str(manifest_path)
        report["frozen_input_keys"] = sorted(manifest.keys())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"preflight ok -> {output}")
    return 0


def cmd_gpu_stage(stage: str, args: argparse.Namespace) -> int:
    """GPU stages are gated: frozen input + GPU UUIDs + deadline required."""
    missing = []
    if not args.frozen_input:
        missing.append("--frozen-input")
    if not args.gpu_uuids:
        missing.append("--gpu-uuids")
    if not args.deadline:
        missing.append("--deadline")
    if missing:
        print(
            f"stage {stage!r} requires explicit GPU inputs; missing: "
            + ", ".join(missing)
            + ". Refusing to occupy GPU.",
            file=sys.stderr,
        )
        return 2
    print(
        f"stage {stage!r} acknowledged (NOT_RUN in this delivery): "
        "GPU execution requires an explicit usage window; see acceptance.md.",
    )
    return 3


def cmd_report(args: argparse.Namespace) -> int:
    print("report: see docs/plans/verl-closeout/acceptance.md")
    return 0


def cmd_cleanup(args: argparse.Namespace) -> int:
    from src.integrations.verl import RunCleanup

    cleanup = RunCleanup()
    outcomes = cleanup.terminate_all()
    print(f"cleanup: no registered processes in CPU delivery; outcomes={outcomes}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES, nargs="?", default="preflight")
    parser.add_argument("--frozen-input", default=None)
    parser.add_argument("--gpu-uuids", default=None)
    parser.add_argument("--deadline", default=None)
    parser.add_argument("--output", default="docs/plans/verl-closeout/preflight.json")
    args = parser.parse_args(argv)
    if args.stage == "preflight":
        return cmd_preflight(args)
    if args.stage in GPU_STAGES:
        return cmd_gpu_stage(args.stage, args)
    if args.stage == "report":
        return cmd_report(args)
    if args.stage == "cleanup":
        return cmd_cleanup(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
