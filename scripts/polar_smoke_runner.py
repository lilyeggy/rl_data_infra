"""Run Polar's calculator driver against a tiny deterministic coding task."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    repo = Path(os.environ["AGENT_REPO"])
    polar_dir = Path(os.environ["POLAR_DIR"])
    topology = Path(os.environ["POLAR_TOPOLOGY"])
    smoke_assets = repo / "benchmarks" / "polar_smoke" / "assets"

    sys.path.insert(0, str(polar_dir / "examples" / "calculator"))
    import run  # type: ignore[import-not-found]

    # Fair warmed comparison: use the same prebuilt Pi image/version and the
    # same model identity as the Local Data Plane benchmark. Polar still owns
    # its gateway, rollout server, session lifecycle and result assembly.
    run.RUNTIME_IMAGE = os.environ.get("POLAR_RUNTIME_IMAGE", "agent-data-plane-pi:0.84.2")
    run.HARNESS_INSTALL["pi"] = os.environ.get(
        "POLAR_PI_INSTALL",
        "pi --version >/dev/null",
    )
    run.HARNESS_MODEL["pi"] = f"openai/{os.environ.get('POLAR_MODEL_ID', 'qwen2.5-coder-14b-instruct')}"

    original_builder = run.build_task_payload

    def build_task_payload(harness: str, batch_id: str, backend: str):
        payload = original_builder(harness, batch_id, backend)
        payload["task_id"] = f"polar-smoke-{harness}-{batch_id}"
        payload["instruction"] = (
            "The complete initial contents of target.py are exactly: "
            "def answer():\\n    return 0\\n. Edit only target.py so answer() returns 42; "
            "use edit oldText `return 0` and newText `return 42`. Do not change test_target.py. "
            "Dependency policy: in your first tool-call turn issue only the edit call; after its successful "
            "result, issue exactly one bash call `python3 test_target.py` in a later turn."
        )
        payload["agent"]["settings"] = {
            "thinking": "minimal",
            "context_window": 32768,
            "max_tokens": 8192,
            "api_type": "openai-completions",
            "compat": {
                "supportsStrictMode": False,
                "supportsStore": False,
                "maxTokensField": "max_tokens",
            },
        }
        prepare = payload["runtime"]["prepare"]
        prepare[1]["source"] = str(smoke_assets / "test_target.py")
        prepare[1]["target"] = "/polar/session/workspace/test_target.py"
        prepare[2]["source"] = str(smoke_assets / "target.py")
        prepare[2]["target"] = "/polar/session/workspace/target.py"
        evaluator = payload["evaluator"]["config"]
        evaluator["test_command"] = (
            "cd /polar/session/workspace && python3 test_target.py && "
            "echo 'PASSED test_target'"
        )
        evaluator["expected_output_json"] = {"test_target": "PASSED"}
        # This deterministic smoke verifier checks the final workspace state.
        # Reusing the isolated agent runtime avoids Polar's second npm-based
        # eval-runtime prewarm, which is both redundant here and network-fragile.
        payload["evaluator"]["refresh_runtime"] = False
        return payload

    run.build_task_payload = build_task_payload
    run.DEFAULT_TOPOLOGY = topology
    run.NUM_SAMPLES = int(os.environ["POLAR_NUM_SAMPLES"])
    run.POLL_INTERVAL_SECONDS = float(os.environ.get("POLAR_POLL_INTERVAL_SECONDS", "1"))
    sys.argv = ["run.py", "--harness", "pi", "--backend", "docker"]
    return run.main()


if __name__ == "__main__":
    raise SystemExit(main())
