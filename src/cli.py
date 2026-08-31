"""Dependency-free CLI for current data-plane operations."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from src.assembly import finalize_local_run
from src.capture import (
    ArtifactStore,
    EventWriter,
    ModelEndpointKind,
    ModelEvidenceJsonlWriter,
    ModelProxyHttpServer,
    ModelProxyService,
    TraceRecorder,
)
from src.capture.pi_adapter import dump_pi_ndjson, read_pi_ndjson
from src.certification import EligibilityDecision
from src.contracts.agent_episode import AgentEpisode
from src.contracts.dataset import DatasetManifest
from src.contracts.execution_bundle import ExecutionBundle
from src.contracts.execution_identity import ExecutionIdentity
from src.contracts.run_manifest import ExecutionRunManifest
from src.evaluation import build_data_quality_report
from src.integrations.slime import admit_on_policy_manifest
from src.launchers import (
    DockerLocalLauncher,
    LocalHarnessRequest,
    SandboxLimits,
    SandboxNetworkPolicy,
    SandboxProfile,
    SandboxRuntime,
    build_docker_launch_plan,
)
from src.orchestration import (
    LocalExecutionOrchestrator,
    LocalExecutionSpec,
    prepare_local_execution,
)
from src.producers import ProducerArtifact
from src.storage.http_service import StorageHttpServer, StorageIngestionService


def _read_episodes(path: Path) -> tuple[AgentEpisode, ...]:
    episodes: list[AgentEpisode] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                episodes.append(AgentEpisode.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid episode at {path}:{line_number}: {exc}") from exc
    return tuple(episodes)


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _serve_storage(args: argparse.Namespace) -> int:
    service = StorageIngestionService(
        args.root,
        max_pending_events=args.max_pending_events,
    )
    server = StorageHttpServer(service, host=args.host, port=args.port)
    print(json.dumps({"address": server.address, "root": args.root}, sort_keys=True))
    try:
        server.httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.close()
    return 0


def _serve_model_proxy(args: argparse.Namespace) -> int:
    identity = ExecutionIdentity(
        run_id=args.run_id,
        task_id=args.task_id,
        episode_id=args.episode_id,
        attempt_id=args.attempt_id,
        producer_id="local-docker",
        producer_version="local-docker-launcher/v1",
        group_id=args.group_id,
        policy_fingerprint=args.policy_fingerprint,
        sampling_fingerprint=args.sampling_fingerprint,
    )
    recorder = TraceRecorder(
        EventWriter(args.events),
        run_id=identity.run_id,
        episode_id=identity.episode_id,
        trace_id=args.trace_id,
    )
    authorization = os.environ.get(args.upstream_auth_env) if args.upstream_auth_env else None
    access_token = os.environ.get(args.access_token_env)
    if not access_token:
        raise SystemExit(f"proxy access token env {args.access_token_env!r} is missing")
    service = ModelProxyService(
        identity=identity,
        endpoint_kind=ModelEndpointKind(args.endpoint_kind),
        upstream_chat_completions_url=args.upstream_url,
        evidence_writer=ModelEvidenceJsonlWriter(args.evidence),
        recorder=recorder,
        upstream_authorization=authorization,
        access_token=access_token,
        timeout_seconds=args.timeout_seconds,
    )
    server = ModelProxyHttpServer(service, host=args.host, port=args.port)
    _print(
        {
            "address": server.address,
            "endpoint": "/v1/chat/completions",
            "run_id": identity.run_id,
            "episode_id": identity.episode_id,
            "events": args.events,
            "evidence": args.evidence,
        }
    )
    try:
        server.httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.httpd.server_close()
    return 0


def _sanitize_pi(args: argparse.Namespace) -> int:
    source = Path(args.input)
    destination = Path(args.output)
    records, issues = read_pi_ndjson(source.read_text())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(dump_pi_ndjson(records))
    _print(
        {
            "input": str(source),
            "output": str(destination),
            "record_count": len(records),
            "issue_count": len(issues),
        }
    )
    return 0


def _inspect(args: argparse.Namespace) -> int:
    episodes = _read_episodes(Path(args.episodes))
    matches = [episode for episode in episodes if episode.episode_id == args.episode_id]
    if not matches:
        raise SystemExit(f"episode_id {args.episode_id!r} not found")
    episode = matches[0]
    _print(
        {
            "episode_id": episode.episode_id,
            "task_id": episode.task_id,
            "harness": episode.harness_manifest.to_dict(),
            "model": episode.model_manifest.to_dict(),
            "outcome": episode.outcome.to_dict(),
            "integrity": episode.integrity.to_dict(),
            "capabilities": sorted(item.value for item in episode.capabilities),
            "timeline": [
                {
                    "sequence": event.sequence,
                    "timestamp": event.timestamp,
                    "event_id": event.event_id,
                    "span_id": event.span_id,
                    "parent_span_id": event.parent_span_id,
                    "type": event.event_type.value,
                    "component": event.component.value,
                    "status": event.status.value,
                    "attempt": event.attempt,
                    "artifact_refs": list(event.artifact_refs),
                }
                for event in episode.events
            ],
        }
    )
    return 0


def _read_jsonl(path: Path, parser: Any) -> tuple[Any, ...]:
    values = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                values.append(parser(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid record at {path}:{line_number}: {exc}") from exc
    return tuple(values)


def _quality_report(args: argparse.Namespace) -> int:
    artifacts = _read_jsonl(Path(args.artifacts), ProducerArtifact.from_dict)
    decisions = _read_jsonl(Path(args.decisions), EligibilityDecision.from_dict)
    report = build_data_quality_report(artifacts, decisions)
    payload = report.to_dict() | {"checksum": report.checksum}
    if args.output:
        _atomic_write_json(Path(args.output), payload)
    _print(payload)
    return 0


def _admit_slime(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    try:
        manifest = DatasetManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise ValueError(f"invalid manifest at {manifest_path}: {exc}") from exc
    decisions = _read_jsonl(Path(args.decisions), EligibilityDecision.from_dict)
    bundles = _read_jsonl(Path(args.bundles), ExecutionBundle.from_dict)
    artifacts = _read_jsonl(Path(args.artifacts), ProducerArtifact.from_dict)
    admission = admit_on_policy_manifest(
        manifest,
        decisions_by_checksum={item.checksum: item for item in decisions},
        bundles_by_checksum={item.checksum: item for item in bundles},
        artifacts_by_checksum={item.checksum: item for item in artifacts},
        minimum_group_size=args.minimum_group_size,
    )
    payload = admission.to_dict() | {"checksum": admission.checksum}
    _atomic_write_json(Path(args.output), payload)
    _print(
        {
            "output": args.output,
            "checksum": admission.checksum,
            "trajectory_count": admission.trajectory_count,
            "trace_count": len(admission.traces),
        }
    )
    return 0


def _local_request(args: argparse.Namespace) -> LocalHarnessRequest:
    command = tuple(args.harness_command)
    if command[:1] == ("--",):
        command = command[1:]
    return LocalHarnessRequest(
        identity=ExecutionIdentity(
            run_id=args.run_id,
            task_id=args.task_id,
            episode_id=args.episode_id,
            attempt_id=args.attempt_id,
            producer_id="local-docker",
            producer_version="local-docker-launcher/v1",
            group_id=args.group_id,
            policy_fingerprint=args.policy_fingerprint,
            sampling_fingerprint=args.sampling_fingerprint,
        ),
        image=args.image,
        image_digest=args.image_digest,
        harness_argv=command,
        host_workspace=args.workspace,
        timeout_seconds=args.timeout_seconds,
        platform=args.platform,
        model_proxy_url=args.model_proxy_url,
        model_proxy_api_key=(
            os.environ.get(args.model_proxy_api_key_env) if args.model_proxy_api_key_env else None
        ),
        network_policy=SandboxNetworkPolicy(args.network_policy),
        sandbox_profile=SandboxProfile(args.sandbox_profile),
        sandbox_runtime=SandboxRuntime(args.sandbox_runtime),
        limits=SandboxLimits(
            cpus=args.cpus,
            memory_mb=args.memory_mb,
            pids=args.pids,
            tmpfs_mb=args.tmpfs_mb,
        ),
    )


def _plan_local(args: argparse.Namespace) -> int:
    plan = build_docker_launch_plan(_local_request(args))
    _print(plan.to_dict() | {"checksum": plan.checksum})
    return 0


def _run_local(args: argparse.Namespace) -> int:
    artifact_root = (
        Path(args.artifact_root)
        if args.artifact_root
        else Path(args.artifact_output).parent / "objects"
    )
    artifact = DockerLocalLauncher(artifact_store=ArtifactStore(artifact_root)).run(
        _local_request(args)
    )
    _atomic_write_json(Path(args.artifact_output), artifact.to_dict())
    _print(
        {
            "artifact_output": args.artifact_output,
            "artifact_checksum": artifact.checksum,
            "status": artifact.status.value,
        }
    )
    return 0 if artifact.status.value == "COMPLETED" else 1


def _finalize_local(args: argparse.Namespace) -> int:
    manifest = ExecutionRunManifest.from_dict(
        json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    )
    artifact = ProducerArtifact.from_dict(
        json.loads(Path(args.producer_artifact).read_text(encoding="utf-8"))
    )
    result = finalize_local_run(
        manifest=manifest,
        producer_artifact=artifact,
        events_path=args.events,
        model_evidence_path=args.model_evidence,
        artifacts_path=args.artifacts,
    )
    output = Path(args.output_dir)
    _atomic_write_json(output / "episode.json", result.episode.to_dict())
    _atomic_write_json(output / "execution-bundle.json", result.execution_bundle.to_dict())
    _atomic_write_json(
        output / "finalization.json",
        {
            "manifest_checksum": manifest.checksum,
            "producer_artifact_checksum": artifact.checksum,
            "episode_checksum": result.episode.checksum,
            "execution_bundle_checksum": result.execution_bundle.checksum,
            "model_evidence_checksums": [item.checksum for item in result.model_evidence],
            "warnings": list(result.warnings),
        },
    )
    _print(
        {
            "output_dir": str(output),
            "episode_checksum": result.episode.checksum,
            "execution_bundle_checksum": result.execution_bundle.checksum,
            "integrity": result.episode.integrity.state.value,
        }
    )
    return 0


def _read_local_execution_spec(path: str) -> LocalExecutionSpec:
    source = Path(path)
    try:
        return LocalExecutionSpec.from_dict(json.loads(source.read_text(encoding="utf-8")))
    except Exception as exc:
        raise ValueError(f"invalid local execution spec at {source}: {exc}") from exc


def _prepare_local(args: argparse.Namespace) -> int:
    spec = _read_local_execution_spec(args.spec)
    prepared = prepare_local_execution(
        spec,
        proxy_base_url=args.proxy_base_url,
        access_token="prepare-only-secret",
    )
    output = Path(args.output_dir)
    _atomic_write_json(output / "execution-run-manifest.json", prepared.manifest.to_dict())
    _atomic_write_json(
        output / "launch-plan.json",
        prepared.launch_plan.to_dict() | {"checksum": prepared.launch_plan.checksum},
    )
    _print(
        {
            "output_dir": str(output),
            "manifest_checksum": prepared.manifest.checksum,
            "launch_plan_checksum": prepared.launch_plan.checksum,
        }
    )
    return 0


def _execute_local(args: argparse.Namespace) -> int:
    spec = _read_local_execution_spec(args.spec)
    authorization = os.environ.get(args.upstream_auth_env) if args.upstream_auth_env else None
    result = LocalExecutionOrchestrator().run(
        spec,
        output_dir=args.output_dir,
        upstream_authorization=authorization,
    )
    _print(
        {
            "output_dir": str(result.output_dir),
            "producer_status": result.producer_artifact.status.value,
            "episode_checksum": result.finalization.episode.checksum,
            "execution_bundle_checksum": result.finalization.execution_bundle.checksum,
            "integrity": result.finalization.episode.integrity.state.value,
        }
    )
    return (
        0
        if result.producer_artifact.status.value == "COMPLETED"
        and result.finalization.episode.integrity.state.value == "COMPLETE"
        else 1
    )


def _add_local_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--attempt-id", type=int, default=1)
    parser.add_argument("--group-id")
    parser.add_argument("--policy-fingerprint")
    parser.add_argument("--sampling-fingerprint")
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument(
        "--platform",
        choices=("linux/amd64", "linux/arm64"),
        default="linux/amd64",
    )
    parser.add_argument("--model-proxy-url")
    parser.add_argument("--model-proxy-api-key-env")
    parser.add_argument(
        "--network-policy",
        choices=[item.value for item in SandboxNetworkPolicy],
        default=SandboxNetworkPolicy.NONE.value,
    )
    parser.add_argument("--cpus", type=float, default=2.0)
    parser.add_argument("--memory-mb", type=int, default=4096)
    parser.add_argument("--pids", type=int, default=256)
    parser.add_argument("--tmpfs-mb", type=int, default=512)
    parser.add_argument(
        "--sandbox-profile",
        choices=[item.value for item in SandboxProfile],
        default=SandboxProfile.FAST.value,
    )
    parser.add_argument(
        "--sandbox-runtime",
        choices=[item.value for item in SandboxRuntime],
        default=SandboxRuntime.RUNC.value,
    )
    parser.add_argument("harness_command", nargs=argparse.REMAINDER)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-data-plane")
    commands = parser.add_subparsers(dest="command", required=True)
    serve_storage = commands.add_parser(
        "serve-storage",
        help="run the bounded V2.3 HTTP ingestion baseline",
    )
    serve_storage.add_argument("--root", default="artifacts/v2.3-http-storage")
    serve_storage.add_argument("--host", default="127.0.0.1")
    serve_storage.add_argument("--port", type=int, default=8080)
    serve_storage.add_argument("--max-pending-events", type=int, default=10000)
    serve_storage.set_defaults(func=_serve_storage)
    serve_proxy = commands.add_parser(
        "serve-model-proxy",
        help="run an execution-bound OpenAI-compatible evidence proxy",
    )
    serve_proxy.add_argument("--run-id", required=True)
    serve_proxy.add_argument("--task-id", required=True)
    serve_proxy.add_argument("--episode-id", required=True)
    serve_proxy.add_argument("--trace-id", required=True)
    serve_proxy.add_argument("--attempt-id", type=int, default=1)
    serve_proxy.add_argument("--group-id")
    serve_proxy.add_argument("--policy-fingerprint")
    serve_proxy.add_argument("--sampling-fingerprint")
    serve_proxy.add_argument(
        "--endpoint-kind",
        choices=[item.value for item in ModelEndpointKind],
        required=True,
    )
    serve_proxy.add_argument("--upstream-url", required=True)
    serve_proxy.add_argument("--upstream-auth-env")
    serve_proxy.add_argument("--access-token-env", required=True)
    serve_proxy.add_argument("--events", required=True)
    serve_proxy.add_argument("--evidence", required=True)
    serve_proxy.add_argument("--host", default="127.0.0.1")
    serve_proxy.add_argument("--port", type=int, default=8088)
    serve_proxy.add_argument("--timeout-seconds", type=float, default=300)
    serve_proxy.set_defaults(func=_serve_model_proxy)
    sanitize_pi = commands.add_parser(
        "sanitize-pi", help="redact and normalize a Pi NDJSON trace for fixtures"
    )
    sanitize_pi.add_argument("--input", required=True)
    sanitize_pi.add_argument("--output", required=True)
    sanitize_pi.set_defaults(func=_sanitize_pi)
    inspect = commands.add_parser("inspect", help="inspect one canonical Episode timeline")
    inspect.add_argument("--episodes", required=True)
    inspect.add_argument("--episode-id", required=True)
    inspect.set_defaults(func=_inspect)
    quality = commands.add_parser(
        "quality-report",
        help="summarize persisted producer artifacts and certification decisions",
    )
    quality.add_argument("--artifacts", required=True, help="ProducerArtifact JSONL")
    quality.add_argument("--decisions", required=True, help="EligibilityDecision JSONL")
    quality.add_argument("--output")
    quality.set_defaults(func=_quality_report)
    admission = commands.add_parser(
        "admit-slime",
        help="strictly rebuild a Slime admission envelope from persisted evidence",
    )
    admission.add_argument("--manifest", required=True)
    admission.add_argument("--decisions", required=True)
    admission.add_argument("--bundles", required=True)
    admission.add_argument("--artifacts", required=True)
    admission.add_argument("--output", required=True)
    admission.add_argument("--minimum-group-size", type=int, default=2)
    admission.set_defaults(func=_admit_slime)
    plan_local = commands.add_parser(
        "plan-local",
        help="render a non-executing, identity-bound local Docker launch plan",
    )
    _add_local_arguments(plan_local)
    plan_local.set_defaults(func=_plan_local)
    run_local = commands.add_parser(
        "run-local",
        help="run one Harness in a constrained local Docker container",
    )
    _add_local_arguments(run_local)
    run_local.add_argument("--artifact-output", required=True)
    run_local.add_argument("--artifact-root")
    run_local.set_defaults(func=_run_local)
    finalize_local = commands.add_parser(
        "finalize-local",
        help="strictly join one persisted local run into Episode and ExecutionBundle",
    )
    finalize_local.add_argument("--manifest", required=True)
    finalize_local.add_argument("--producer-artifact", required=True)
    finalize_local.add_argument("--events", required=True)
    finalize_local.add_argument("--model-evidence", required=True)
    finalize_local.add_argument("--artifacts")
    finalize_local.add_argument("--output-dir", required=True)
    finalize_local.set_defaults(func=_finalize_local)
    prepare_local = commands.add_parser(
        "prepare-local",
        help="freeze a secret-free local execution spec into manifest and launch plan",
    )
    prepare_local.add_argument("--spec", required=True)
    prepare_local.add_argument("--output-dir", required=True)
    prepare_local.add_argument(
        "--proxy-base-url",
        default="http://host.docker.internal:8088",
    )
    prepare_local.set_defaults(func=_prepare_local)
    execute_local = commands.add_parser(
        "execute-local",
        help="run proxy, Harness, verifier and finalizer as one local transaction",
    )
    execute_local.add_argument("--spec", required=True)
    execute_local.add_argument("--output-dir", required=True)
    execute_local.add_argument("--upstream-auth-env")
    execute_local.set_defaults(func=_execute_local)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
