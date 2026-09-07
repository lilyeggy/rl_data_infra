#!/usr/bin/env python3
"""Temporary diagnostic proxy for comparing Pi and vLLM request payloads."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
import json
import sys

UPSTREAM = sys.argv[1]
LOG = sys.argv[2]

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        with open(LOG, "ab") as f:
            f.write(body + b"\n")
        req = Request(UPSTREAM, data=body, method="POST", headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
        try:
            with urlopen(req, timeout=900) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.end_headers()
                self.wfile.write(data)
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(str(e).encode())
    def log_message(self, *args):
        pass

ThreadingHTTPServer(("127.0.0.1", 18000), Handler).serve_forever()
