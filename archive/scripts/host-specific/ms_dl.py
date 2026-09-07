#!/usr/bin/env python3
"""ARCHIVED host-specific Qwen2.5-Coder-14B downloader.

Download Qwen2.5-Coder-14B-Instruct from modelscope CDN with curl (resumable),
bypassing the buggy modelscope snapshot_download API."""
import json, os, subprocess, sys

MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
DEST = "/home/f630/homePLUS/agentic/models/qwen2.5-coder-14b-instruct"
API = f"https://www.modelscope.cn/api/v1/models/{MODEL}/repo/files?Revision=master&Recursive=true"

d = json.loads(subprocess.run(["curl", "-s", "-m", "30", API], capture_output=True, text=True).stdout)
files = [f["Path"] for f in d["Data"]["Files"] if f["Name"] not in (".gitattributes",)]
print("FILES_TO_DL:", len(files), flush=True)

os.makedirs(DEST, exist_ok=True)
for f in files:
    url = f"https://www.modelscope.cn/models/{MODEL}/resolve/master/{f}"
    out = os.path.join(DEST, f)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # resumable curl; retry a few times per file
    for attempt in range(5):
        r = subprocess.run(["curl", "-sL", "-C", "-", "-o", out, url])
        if r.returncode == 0 and os.path.getsize(out) > 0:
            # verify not an error page (<1KB usually means 404/error)
            if os.path.getsize(out) > 1000:
                break
        print("  retry", f, "attempt", attempt, flush=True)
    print("DL", f, os.path.getsize(out), flush=True)
print("MS_DL_DONE", flush=True)
