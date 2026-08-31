#!/usr/bin/env python3
"""Run the real Pi Harness through the evidence-capturing model proxy.

This is the integration gate before a SWE-bench batch: Pi retains its native
SSE protocol while the proxy turns the controlled model call into durable,
RL-usable evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.capture import (
    EventWriter,
    ModelEndpointKind,
    ModelEvidenceJsonlWriter,
    ModelProxyHttpServer,
    ModelProxyService,
    TraceRecorder,
    read_pi_ndjson,
)
from src.contracts.execution_identity import ExecutionIdentity


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi", required=True, help="absolute path to Pi executable")
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument("--model", default="qwen2.5-coder-14b-instruct")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-fingerprint", required=True)
    parser.add_argument("--sampling-fingerprint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--tools",
        default="read,bash,write,edit,grep,find,ls,glob",
        help="explicit Pi tool allowlist for the bounded run",
    )
    parser.add_argument("--timeout", type=float, default=180)
    return parser.parse_args()


def _models_config(*, base_url: str) -> dict[str, object]:
    return {
        "providers": {
            "local-qwen-proxy": {
                "baseUrl": f"{base_url}/v1",
                "api": "openai-completions",
                "apiKey": "$AGENT_MODEL_PROXY_API_KEY",
                "compat": {
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                    "supportsUsageInStreaming": True,
                    "supportsStore": False,
                    "maxTokensField": "max_tokens",
                    "supportsStrictMode": False,
                },
                "models": [
                    {
                        "id": "qwen2.5-coder-14b-instruct",
                        "name": "Qwen2.5 Coder 14B via Data Plane",
                        "reasoning": False,
                        "input": ["text"],
                        "contextWindow": 32768,
                        "maxTokens": 4096,
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    }
                ],
            }
        }
    }


def main() -> int:
    args = _arguments()
    args.workspace.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=False)
    run_id = f"pi-proxy-{secrets.token_hex(6)}"
    identity = ExecutionIdentity(
        run_id=run_id,
        task_id="pi-proxy-integration",
        episode_id=f"episode-{run_id}",
        attempt_id=1,
        producer_id="pi-direct",
        producer_version="pi-direct/v1",
        policy_fingerprint=args.policy_fingerprint,
        sampling_fingerprint=args.sampling_fingerprint,
    )
    events_path = args.output_dir / "proxy-events.jsonl"
    evidence_path = args.output_dir / "model-evidence.jsonl"
    recorder = TraceRecorder(
        EventWriter(events_path),
        run_id=identity.run_id,
        episode_id=identity.episode_id,
        trace_id=f"trace-{run_id}",
    )
    access_token = secrets.token_urlsafe(32)
    server = ModelProxyHttpServer(
        ModelProxyService(
            identity=identity,
            endpoint_kind=ModelEndpointKind.CONTROLLED,
            upstream_chat_completions_url=args.upstream_url,
            evidence_writer=ModelEvidenceJsonlWriter(evidence_path),
            recorder=recorder,
            access_token=access_token,
            timeout_seconds=args.timeout,
        ),
        host="127.0.0.1",
        port=0,
    )
    server.start_in_thread()
    pi_home = Path(tempfile.mkdtemp(prefix="pi-data-plane-"))
    try:
        models_path = pi_home / ".pi" / "agent" / "models.json"
        models_path.parent.mkdir(parents=True)
        models_path.write_text(json.dumps(_models_config(base_url=f"http://127.0.0.1:{server.address[1]}")))
        environment = os.environ | {
            "HOME": str(pi_home),
            "AGENT_MODEL_PROXY_API_KEY": access_token,
            "PATH": f"{Path(args.pi).parent}:{os.environ.get('PATH', '')}",
        }
        command = [
            args.pi,
            "--provider",
            "local-qwen-proxy",
            "--model",
            args.model,
            "--mode",
            "json",
            "--print",
            "--no-session",
            "--no-context-files",
            "--no-extensions",
            "--no-skills",
            "--tools",
            args.tools,
            "--thinking",
            "minimal",
            args.prompt,
        ]
        result = subprocess.run(
            command,
            cwd=args.workspace,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    finally:
        server.close()
        shutil.rmtree(pi_home, ignore_errors=True)
    raw_path = args.output_dir / "pi.ndjson"
    raw_path.write_text(result.stdout)
    (args.output_dir / "pi.stderr.txt").write_text(result.stderr)
    records, issues = read_pi_ndjson(result.stdout)
    evidence = (
        [json.loads(line) for line in evidence_path.read_text().splitlines() if line]
        if evidence_path.exists()
        else []
    )
    rl_usable = sum(
        {
            "TOKEN_IDS",
            "BEHAVIOR_LOGPROBS",
            "POLICY_VERSION",
        }.issubset(item["capabilities"])
        for item in evidence
    )
    summary = {
        "run_id": run_id,
        "returncode": result.returncode,
        "pi_record_count": len(records),
        "pi_parse_issue_count": len(issues),
        "model_call_count": len(evidence),
        "rl_usable_model_call_count": rl_usable,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if result.returncode != 0 or not evidence or rl_usable != len(evidence):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
