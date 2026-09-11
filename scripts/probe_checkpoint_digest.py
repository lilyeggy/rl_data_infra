"""Recompute the manager's checkpoint digest over global_step_1/actor.

Mirrors CertifiedVerlAgentLoopManager.generate_sequences: relative path, a NUL
separator, then the file bytes in sorted order.
"""

import hashlib
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
files = sorted(p for p in checkpoint.rglob("*") if p.is_file())
digest = hashlib.sha256()
for path in files:
    digest.update(str(path.relative_to(checkpoint)).encode() + b"\0")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
print(f"files={len(files)}")
print(f"digest={digest.hexdigest()}")
