"""Local-only operations dashboard for the Agent Improvement Data Plane.

The dashboard deliberately exposes a small allowlist of operations.  It is not
a remote shell: model restart and smoke execution use fixed, auditable argv.
Run it on the GPU host and access it through an SSH local-forward.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run(argv: list[str], *, timeout: float = 8) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-8_000:],
        "stderr": completed.stderr[-8_000:],
    }


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    repo_root: Path
    model_url: str
    model_id: str
    model_launcher: Path
    python_executable: str
    image_tag: str
    host: str = "127.0.0.1"
    port: int = 8788

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_root", self.repo_root.resolve())
        object.__setattr__(self, "model_launcher", self.model_launcher.resolve())
        if not self.repo_root.is_dir():
            raise ValueError("repo_root must be a directory")
        if not self.model_launcher.is_file():
            raise ValueError("model_launcher must be a file")
        if not self.model_url.startswith("http://"):
            raise ValueError("model_url must use http")
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not self.image_tag:
            raise ValueError("image_tag must be non-empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be valid")

    @property
    def runs_dir(self) -> Path:
        return self.repo_root / "runs"


@dataclass(slots=True)
class DashboardJob:
    job_id: str
    kind: str
    argv: tuple[str, ...]
    log_path: Path
    process: subprocess.Popen[str]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        status = "RUNNING" if self.process.poll() is None else "FINISHED"
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": status,
            "returncode": self.process.poll(),
            "created_at": self.created_at,
            "log_path": str(self.log_path),
            "argv": list(self.argv),
        }


@dataclass(slots=True)
class DashboardController:
    config: DashboardConfig
    _jobs: dict[str, DashboardJob] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def status(self) -> dict[str, Any]:
        return {
            "model": self._model_status(),
            "gpu": _run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]
            ),
            "docker_image": _run(
                ["docker", "image", "inspect", self.config.image_tag, "--format", "{{.Id}}"]
            ),
            "recent_runs": self._recent_runs(),
            "jobs": [job.to_dict() for job in self._jobs.values()],
        }

    def restart_model(self) -> dict[str, Any]:
        return self._start_job("model-restart", ["bash", str(self.config.model_launcher)])

    def run_smoke(self) -> dict[str, Any]:
        image = _run(["docker", "image", "inspect", self.config.image_tag, "--format", "{{.Id}}"])
        image_id = image.get("stdout", "").strip()
        if not image.get("ok") or not image_id.startswith("sha256:"):
            raise RuntimeError("configured Docker image is unavailable")
        digest = image_id.removeprefix("sha256:")
        run_name = f"live-ui-{_utc_slug()}"
        workspace = self.config.runs_dir / "live-ui-workspace"
        output = self.config.runs_dir / run_name
        workspace.mkdir(parents=True, exist_ok=True)
        return self._start_job(
            "live-smoke",
            [
                self.config.python_executable,
                "scripts/a6000_live_smoke.py",
                "--image",
                image_id,
                "--image-digest",
                digest,
                "--upstream-url",
                f"{self.config.model_url.rstrip('/')}/v1/chat/completions",
                "--model-id",
                self.config.model_id,
                "--workspace",
                str(workspace),
                "--output-dir",
                str(output),
            ],
        )

    def _start_job(self, kind: str, argv: list[str]) -> dict[str, Any]:
        with self._lock:
            if any(job.to_dict()["status"] == "RUNNING" for job in self._jobs.values()):
                raise RuntimeError("another dashboard operation is still running")
            job_id = f"{kind}-{_utc_slug()}"
            log_dir = self.config.runs_dir / "dashboard-jobs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{job_id}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                argv,
                cwd=self.config.repo_root,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log_handle.close()
            job = DashboardJob(
                job_id=job_id,
                kind=kind,
                argv=tuple(argv),
                log_path=log_path,
                process=process,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self._jobs[job_id] = job
            return job.to_dict()

    def job_log(self, job_id: str) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job.to_dict() | {"log": job.log_path.read_text(errors="replace")[-32_000:]}

    def _model_status(self) -> dict[str, Any]:
        try:
            health_url = f"{self.config.model_url.rstrip('/')}/health"
            with urllib.request.urlopen(health_url, timeout=3) as response:
                return {"ok": response.status == 200, "response": json.loads(response.read())}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return {"ok": False, "error": str(exc)}

    def _recent_runs(self) -> list[dict[str, Any]]:
        if not self.config.runs_dir.is_dir():
            return []
        results: list[dict[str, Any]] = []
        for directory in sorted(self.config.runs_dir.iterdir(), reverse=True):
            episode_path = directory / "finalized" / "episode.json"
            if not episode_path.is_file():
                continue
            try:
                episode = json.loads(episode_path.read_text())
                results.append(
                    {
                        "run_directory": directory.name,
                        "integrity": episode["integrity"]["state"],
                        "verifier_status": episode["outcome"]["verifier_status"],
                        "task_status": episode["outcome"]["task_status"],
                    }
                )
            except (KeyError, OSError, json.JSONDecodeError):
                results.append({"run_directory": directory.name, "integrity": "UNREADABLE"})
            if len(results) == 12:
                break
        return results


def make_handler(controller: DashboardController) -> type[BaseHTTPRequestHandler]:
    static_html = (Path(__file__).with_name("dashboard.html")).read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def _send(self, status: HTTPStatus, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(static_html)))
                self.end_headers()
                self.wfile.write(static_html)
                return
            if self.path == "/api/status":
                self._send(HTTPStatus.OK, controller.status())
                return
            if self.path.startswith("/api/jobs/"):
                try:
                    self._send(HTTPStatus.OK, controller.job_log(self.path.rsplit("/", 1)[-1]))
                except KeyError:
                    self._send(HTTPStatus.NOT_FOUND, {"error": "job_not_found"})
                return
            self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:
            try:
                if self.path == "/api/model/restart":
                    self._send(HTTPStatus.ACCEPTED, controller.restart_model())
                    return
                if self.path == "/api/smoke/run":
                    self._send(HTTPStatus.ACCEPTED, controller.run_smoke())
                    return
                self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            except RuntimeError as exc:
                self._send(HTTPStatus.CONFLICT, {"error": str(exc)})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-id", default="Qwen2.5-Coder-14B")
    parser.add_argument("--model-launcher", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--image-tag", default="agent-data-plane:20260823")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args()
    config = DashboardConfig(
        repo_root=args.repo_root,
        model_url=args.model_url,
        model_id=args.model_id,
        model_launcher=args.model_launcher,
        python_executable=args.python,
        image_tag=args.image_tag,
        host=args.host,
        port=args.port,
    )
    server = ThreadingHTTPServer(
        (config.host, config.port), make_handler(DashboardController(config))
    )
    print(f"dashboard listening on http://{config.host}:{config.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
