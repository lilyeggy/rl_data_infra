"""Assemble the smoke27 evidence from the run's own durable artifacts.

Usage: python3 probe_smoke27_bundle.py <run-root> <log> <out-dir>
"""

import hashlib
import json
import re
import sys
from pathlib import Path

RUN = Path(sys.argv[1])
LOG = Path(sys.argv[2])
OUT = Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
PREFIX = re.compile(r"^\(?[A-Za-z_]+ pid=\d+\)?\s*")
SESSION = next(p for p in (RUN / "episodes").iterdir() if p.is_dir())


def clean_lines() -> list[str]:
    raw = LOG.read_text(encoding="utf-8", errors="replace")
    return [PREFIX.sub("", line) for line in ANSI.sub("", raw).splitlines()]


lines = clean_lines()

# 1. what the manager said, in order
(OUT / "certify-lines.txt").write_text(
    "\n".join(line[line.index("[pi-manager]") :] for line in lines if "pi-manager" in line) + "\n"
)

# 2. per-step metrics, exactly as verl reported them
step_lines = [line for line in lines if re.match(r"^step:\d+ - ", line)]
timings = [line.strip() for line in lines if "timing_s/" in line and "step:" not in line]
timings = sorted(set(timings))
(OUT / "step-metrics.txt").write_text("\n".join(step_lines) + "\n\nstep timings:\n" + "\n".join(timings) + "\n")

# 3. checkpoints, adapter hashes, and the digest the second rollout claimed
checkpoints = RUN / "checkpoints"
report: list[str] = ["checkpoint files written by the framework's own FSDP2 workers:"]
for step in (1, 2):
    for path in sorted((checkpoints / f"global_step_{step}" / "actor").rglob("*")):
        if path.is_file():
            report.append(str(path))

report.append("")
report.append("adapter hashes (written by RayPPOTrainer vs the frozen P0):")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


for label, path in (
    ("P1 global_step_1", checkpoints / "global_step_1/actor/lora_adapter/adapter_model.safetensors"),
    ("P2 global_step_2", checkpoints / "global_step_2/actor/lora_adapter/adapter_model.safetensors"),
    ("P0 frozen SFT", Path("/home/cxr/agentic/checkpoints/sft-runs/qwen14b-base-apps-clean-v2/epoch1/adapter_model.safetensors")),
):
    report.append(f"{sha256(path)}  {label}  {path}")

report.append("")
report.append("policy identity each generation's rollout ran under:")


def checkpoint_digest(step: int) -> str:
    root = checkpoints / f"global_step_{step}" / "actor"
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


for generation, step in (("gen-000", None), ("gen-001", 1)):
    policy = json.loads((SESSION / generation / "round-policy.json").read_text())
    report.append(
        f"{generation}: policy_generation={policy['policy_generation']} "
        f"adapter_revision={policy['adapter_revision']}"
    )
    if step is not None:
        recomputed = checkpoint_digest(step)
        report.append(
            f"  recomputed digest of global_step_{step}/actor = {recomputed} "
            f"-> {'MATCH' if recomputed == policy['adapter_revision'] else 'MISMATCH'}"
        )
report.append("")
report.append("episode-level behaviour fingerprint (one episode per generation):")
for generation in ("gen-000", "gen-001"):
    attempt = sorted(p for p in (SESSION / generation).glob("attempt-*") if p.is_dir())[-1]
    episode = sorted(p for p in attempt.iterdir() if p.is_dir())[0]
    sequence = json.loads((episode / "admitted-sequence.json").read_text())
    report.append(f"{generation}: {sequence['episode_id']} policy_fingerprint={sequence['policy_fingerprint']}")

report.append("")
report.append("episode shape per attempt (episodes / multi-turn / solved):")
for generation in ("gen-000", "gen-001"):
    for attempt in sorted(p for p in (SESSION / generation).glob("attempt-*") if p.is_dir()):
        episodes = sorted(p for p in attempt.iterdir() if p.is_dir())
        multi = 0
        solved = 0
        for episode in episodes:
            summary = json.loads((episode / "summary.json").read_text())
            if summary["model_call_count"] >= 2:
                multi += 1
            if json.loads((episode / "verifier-output.json").read_text()).get("resolved"):
                solved += 1
        report.append(
            f"{generation}/{attempt.name}: episodes={len(episodes)} multi_turn={multi} solved={solved}"
        )

(OUT / "checkpoints.txt").write_text("\n".join(report) + "\n")

# 4. the refused draw, as it was written at the time
redraw = [line for line in lines if "carries no learning signal" in line]
rejections = sorted(SESSION.glob("gen-*/rejected-attempt*.json"))
(OUT / "resample.txt").write_text(
    "\n".join(redraw)
    + "\n\n"
    + "\n\n".join(path.read_text().strip() for path in rejections)
    + "\n"
)

print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")
