from __future__ import annotations

import json
import tempfile
import time
import unittest
import urllib.error
import urllib.request

from src.storage.http_service import StorageHttpServer, StorageIngestionService
from tests.execution_fixtures import make_trace_event


class StorageHttpServiceTest(unittest.TestCase):
    def test_http_ingestion_health_metrics_and_schema_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = StorageIngestionService(temporary, max_pending_events=10)
            server = StorageHttpServer(service, port=0)
            server.start_in_thread()
            url = f"http://127.0.0.1:{server.address[1]}"
            event = make_trace_event(
                event_id="evt-http-1",
                run_id="run-http",
                episode_id="episode-http",
            ).to_dict()
            try:
                request = urllib.request.Request(
                    f"{url}/v1/events",
                    data=json.dumps([event]).encode(),
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    accepted = json.loads(response.read())
                self.assertEqual(accepted["accepted_events"], 1)

                deadline = time.time() + 5
                while time.time() < deadline:
                    if service.metrics()["ingested_events"] == 1:
                        break
                    time.sleep(0.01)
                self.assertEqual(service.metrics()["ingested_events"], 1)
                with urllib.request.urlopen(f"{url}/healthz", timeout=5) as response:
                    self.assertEqual(json.loads(response.read())["status"], "ok")
                with urllib.request.urlopen(f"{url}/metrics", timeout=5) as response:
                    self.assertEqual(json.loads(response.read())["ingested_events"], 1)

                invalid = dict(event, schema_version="trace-event/v999")
                bad_request = urllib.request.Request(
                    f"{url}/v1/events",
                    data=json.dumps(invalid).encode(),
                    method="POST",
                    headers={"Content-Type": "application/json"},
                )
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(bad_request, timeout=5)
                self.assertEqual(raised.exception.code, 400)
            finally:
                server.close()


if __name__ == "__main__":
    unittest.main()
