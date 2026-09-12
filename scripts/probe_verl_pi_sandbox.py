"""Real sandboxed Pi + scripted token backend; CPU diagnostic, never training data."""
import argparse
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer
from verl.experimental.agent_loop.tool_parser import ToolParser
from src.capture import EventWriter, TraceRecorder, ModelEvidenceJsonlWriter, ModelEndpointKind
from src.capture.model_proxy_http import ModelProxyHttpServer, ModelProxyService
from src.contracts.execution_identity import ExecutionIdentity
from src.contracts._json import sha256_json
from src.integrations.verl.native_transport import NativeTokenTransport
from src.integrations.verl.sandbox import pi_bwrap_command
from src.orchestration.pi_host_execution import _models_config, _run_owned_process


async def main(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    workspace = root / "workspace"
    home = root / "home"
    workspace.mkdir()
    (workspace / "task").write_text("Diagnostic: return the word observed.")
    config = home / ".pi/agent"
    config.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    class ScriptedServer:
        calls = 0

        async def generate(self, **kwargs):
            self.calls += 1
            text = ('<tool_call>{"name":"read","arguments":{"path":"task"}}</tool_call>'
                    if self.calls == 1 else "observed") + "<|im_end|>"
            ids = tokenizer.encode(text, add_special_tokens=False)
            return SimpleNamespace(token_ids=ids, log_probs=[-0.1] * len(ids),
                                   extra_fields={"global_steps": 0})

    transport = NativeTokenTransport(
        loop=asyncio.get_running_loop(), server_manager=ScriptedServer(), tokenizer=tokenizer,
        parser=ToolParser.get_tool_parser("hermes", tokenizer), episode_id="cpu-real-pi",
        policy_revision="SCRIPTED_CPU_ONLY", sampling_params={"temperature": 1.0},
        max_prompt=4096, max_response=4096, max_tokens=1024, max_requests=8, timeout=30,
    )
    server = ModelProxyHttpServer(ModelProxyService(
        identity=ExecutionIdentity(run_id="cpu-probe", task_id="diagnostic", episode_id="cpu-real-pi",
            attempt_id=1, producer_id="pi", producer_version="0.84.2", policy_fingerprint="f" * 64),
        endpoint_kind=ModelEndpointKind.CONTROLLED, upstream_chat_completions_url="http://unused",
        evidence_writer=ModelEvidenceJsonlWriter(root / "model-evidence.jsonl"),
        recorder=TraceRecorder(EventWriter(root / "events.jsonl"), run_id="cpu-probe",
            episode_id="cpu-real-pi", trace_id="cpu-probe"), access_token="cpu-probe",
        upstream_transport=transport,
    ), host="127.0.0.1", port=0)
    server.start_in_thread()
    (config / "models.json").write_text(json.dumps(_models_config(
        base_url=f"http://127.0.0.1:{server.address[1]}", model="probe", max_tokens=1024)))
    (config / "settings.json").write_text(json.dumps({"compaction": {"enabled": False},
                                                     "retry": {"enabled": False}}))
    command = [args.pi, "--provider", "local-qwen-proxy", "--model", "probe", "--mode", "json",
        "--print", "--no-session", "--no-context-files", "--no-extensions", "--no-skills",
        "--tools", "read,bash,write,edit,ls", "--thinking", "minimal", "Read task then report the result."]
    command = pi_bwrap_command(command, workspace=workspace, home=home, pi=args.pi, proxy_token="cpu-probe")
    try:
        result = await asyncio.to_thread(_run_owned_process, command, cwd=workspace, env=os.environ, timeout=60)
    finally:
        server.close()
    (root / "pi.ndjson").write_text(result.stdout)
    (root / "stderr.txt").write_text(result.stderr)
    report = {"evidence_kind": "REAL_PI_SCRIPTED_MODEL_CPU_ONLY", "returncode": result.returncode,
              "calls": transport.calls, "bridge_error": transport.error}
    report["status"] = "PASSED" if result.returncode == 0 and transport.calls == 2 and not transport.error else "FAILED"
    if transport.context.tools:
        (root / "tool-schema.json").write_text(json.dumps(transport.context.tools, indent=2))
        report["tool_schema_checksum"] = sha256_json(transport.context.tools)
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    if report["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pi", required=True)
    parser.add_argument("--output", required=True)
    asyncio.run(main(parser.parse_args()))
