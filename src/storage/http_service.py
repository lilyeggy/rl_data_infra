"""Dependency-free HTTP ingestion boundary for the V2.3 container baseline."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from src.storage.event_store import PartitionedEventStore
from src.storage.queue import BackpressureError, BoundedEventQueue
from src.storage.schema import StorageSchemaRegistry


HTTP_SERVICE_VERSION = "storage-http-service/v1"


class StorageIngestionService:
    """Bounded HTTP-to-storage worker with explicit no-auth baseline semantics."""

    def __init__(self, root: str, *, max_pending_events: int = 10_000) -> None:
        self.store = PartitionedEventStore(root, durable=True)
        self.queue = BoundedEventQueue(max_events=max_pending_events)
        self.schema_registry = StorageSchemaRegistry()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._ingested_batches = 0
        self._ingested_events = 0
        self._failed_batches = 0

    def start_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def stop_worker(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=5)

    def submit_payload(self, payload: bytes, *, content_type: str) -> int:
        events = self._parse_payload(payload, content_type=content_type)
        self.queue.put(events, block=False)
        return len(events)

    def health(self) -> dict[str, Any]:
        snapshot = self.queue.snapshot()
        return {
            "service_version": HTTP_SERVICE_VERSION,
            "status": "ok" if not self._stop.is_set() else "stopping",
            "queue": {
                "pending_events": snapshot.pending_events,
                "pending_batches": snapshot.pending_batches,
                "max_events": snapshot.max_events,
            },
        }

    def metrics(self) -> dict[str, Any]:
        snapshot = self.queue.snapshot()
        with self._lock:
            return {
                "service_version": HTTP_SERVICE_VERSION,
                "ingested_batches": self._ingested_batches,
                "ingested_events": self._ingested_events,
                "failed_batches": self._failed_batches,
                "queue": {
                    "pending_events": snapshot.pending_events,
                    "enqueued_events": snapshot.enqueued_events,
                    "dequeued_events": snapshot.dequeued_events,
                    "rejected_batches": snapshot.rejected_batches,
                },
            }

    def _run_worker(self) -> None:
        while not self._stop.is_set():
            try:
                batch = self.queue.get(block=True, timeout_seconds=0.25)
            except BackpressureError:
                continue
            try:
                self.store.ingest(batch)
            except Exception:
                with self._lock:
                    self._failed_batches += 1
                continue
            with self._lock:
                self._ingested_batches += 1
                self._ingested_events += len(batch)

    def _parse_payload(self, payload: bytes, *, content_type: str):
        if len(payload) > 10 * 1024 * 1024:
            raise ValueError("request body exceeds 10 MiB limit")
        try:
            if "ndjson" in content_type:
                raw_values = [
                    json.loads(line)
                    for line in payload.decode().splitlines()
                    if line.strip()
                ]
            else:
                value = json.loads(payload.decode())
                raw_values = value if isinstance(value, list) else [value]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON event payload: {exc}") from exc
        events = []
        for value in raw_values:
            issue = self.schema_registry.validate_event_payload(value)
            if issue is not None:
                raise ValueError(issue.message)
            events.append(self.schema_registry.parse_trace_event(value))
        if not events:
            raise ValueError("request contains no events")
        return tuple(events)


class StorageHttpServer:
    """Threaded HTTP server wrapper for container smoke/integration use."""

    def __init__(self, service: StorageIngestionService, *, host: str = "127.0.0.1", port: int = 0):
        self.service = service
        service_ref = service

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/healthz":
                    self._send(200, service_ref.health())
                elif self.path == "/metrics":
                    self._send(200, service_ref.metrics())
                else:
                    self._send(404, {"error": "not_found"})

            def do_POST(self) -> None:
                if self.path != "/v1/events":
                    self._send(404, {"error": "not_found"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                payload = self.rfile.read(length)
                try:
                    count = service_ref.submit_payload(
                        payload,
                        content_type=self.headers.get("Content-Type", "application/json"),
                    )
                except BackpressureError as exc:
                    self._send(429, {"error": "backpressure", "message": str(exc)})
                    return
                except (TypeError, ValueError) as exc:
                    self._send(400, {"error": "invalid_payload", "message": str(exc)})
                    return
                self._send(202, {"accepted_events": count})

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.service.start_worker()
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.httpd.server_address[:2]
        return str(host), int(port)

    def start_in_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.service.stop_worker()
