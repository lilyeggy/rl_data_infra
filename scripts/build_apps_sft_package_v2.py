#!/usr/bin/env python3
"""Rebuild the APPS token-level SFT package with the 260905 data-quality fixes.

Fixes applied (see docs/experiments/sft-apps-260905-data-quality-postmortem.md):
1. Real teacher thinking recovered from the Pi session log (pi-ndjson artifact);
   redacted "[REDACTED]" placeholders never reach the training target.
2. Host absolute paths (/home/f630/...) normalized: episode workspaces -> <workspace>,
   other host paths -> <host-path>, in BOTH prompt and completion.
3. Samples are rebuilt from the Pi session log directly, so the assistant history
   in the prompt carries real thinking too (no redaction leak into context).

Fail-closed: any sample whose serialized text still contains "[REDACTED]" or a
host path is rejected and counted, never silently kept.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

WS_RE = re.compile(r"/home/f630/homePLUS/agent-data-plane/apps-workspaces/[^\s\"'\\,)]*")
HOST_RE = re.compile(r"/home/f630/[^\s\"'\\,)]*")


def canon(x) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize(text: str) -> tuple[str, int]:
    """Normalize host paths; returns (text, number of replacements)."""
    n = len(WS_RE.findall(text))
    text = WS_RE.sub("<workspace>", text)
    n2 = len(HOST_RE.findall(text))
    text = HOST_RE.sub("<host-path>", text)
    return text, n + n2


def load_session(episode_dir: Path) -> list[dict]:
    """Ordered [{role, content}] from the Pi session log artifact."""
    refs = json.loads((episode_dir / "artifact-refs.json").read_text())
    nd = [r for r in refs if r.get("kind") == "pi-ndjson"]
    if not nd:
        raise FileNotFoundError(f"no pi-ndjson artifact in {episode_dir}")
    path = episode_dir / "objects" / nd[0]["sha256"]
    messages = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") != "message_end":
            continue
        m = rec.get("message") or {}
        role = m.get("role")
        content = m.get("content")
        if role and isinstance(content, list):
            messages.append({"role": role, "content": content})
    return messages


def clean_completion_blocks(content: list) -> list:
    """Completion blocks for training: keep as-is (thinking is real in the
    session log).  Drop thinkingSignature (provider metadata, not model output)."""
    out = []
    for block in content:
        if not isinstance(block, dict):
            continue
        b = dict(block)
        b.pop("thinkingSignature", None)
        out.append(b)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-length", type=int, default=4096)
    a = ap.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)

    rows: list[dict] = []
    tasks: set[str] = set()
    rejected_eligibility = 0
    rejected_failclosed = 0
    kept = 0
    thinking_real = 0
    host_paths_normalized = 0
    episodes_used = 0

    for root in a.roots:
        for elig_path in sorted(Path(root).glob("*/finalized/sft-eligibility.json")):
            try:
                elig = json.loads(elig_path.read_text())
            except Exception:
                continue
            if elig.get("verdict") != "ELIGIBLE":
                rejected_eligibility += 1
                continue
            episode_dir = elig_path.parent.parent
            ep_path = episode_dir / "finalized" / "episode.json"
            if not ep_path.exists():
                continue
            try:
                ep = json.loads(ep_path.read_text())
                messages = load_session(episode_dir)
            except Exception as exc:
                print(f"[skip] {episode_dir.name}: {exc}", file=sys.stderr, flush=True)
                continue
            episodes_used += 1
            task_id = ep.get("task_id")
            episode_id = ep.get("episode_id")

            for idx, msg in enumerate(messages):
                if msg["role"] != "assistant":
                    continue
                blocks = clean_completion_blocks(msg["content"])
                trainable = any(
                    b.get("type") in ("text", "toolCall") for b in blocks
                )
                if not trainable:
                    continue  # pure-thinking turn with no action: nothing to teach
                prompt = "".join(
                    f"[{m['role']}] {normalize(canon(m['content']))[0]}\n"
                    for m in messages[:idx]
                )
                completion = canon(blocks)
                prompt, n1 = normalize(prompt)
                completion, n2 = normalize(completion)
                host_paths_normalized += n1 + n2
                if any(b.get("type") == "thinking" for b in blocks):
                    thinking_real += 1
                # fail-closed checks
                joined = prompt + completion
                if "[REDACTED]" in joined or "/home/f630" in joined:
                    rejected_failclosed += 1
                    continue
                p_ids = tok.encode(prompt, add_special_tokens=True)
                c_ids = tok.encode(completion + tok.eos_token, add_special_tokens=False)
                if not p_ids or not c_ids:
                    rejected_failclosed += 1
                    continue
                if len(p_ids) + len(c_ids) > a.max_length:
                    overflow = len(p_ids) + len(c_ids) - a.max_length
                    p_ids = p_ids[max(0, overflow):] if overflow < len(p_ids) else []
                    if not p_ids:
                        rejected_failclosed += 1
                        continue
                rows.append(
                    {
                        "task_id": task_id,
                        "episode_id": episode_id,
                        "prompt_ids": p_ids,
                        "completion_ids": c_ids,
                        "prompt_text": prompt,
                        "completion_text": completion,
                    }
                )
                tasks.add(task_id)
                kept += 1

    (out / "sft_tokens.jsonl").write_text(
        "\n".join(canon(x) for x in rows) + "\n"
    )
    (out / "examples.jsonl").write_text(
        "\n".join(
            canon({k: v for k, v in x.items() if not k.endswith("_ids")}) for x in rows
        )
        + "\n"
    )
    report = {
        "format": "apps-token-sft/v1",
        "builder": "build_apps_sft_package_v2 (thinking-recovery + path-normalization)",
        "examples": len(rows),
        "tasks": len(tasks),
        "rejected_eligibility": rejected_eligibility,
        "rejected_failclosed": rejected_failclosed,
        "episodes_used": episodes_used,
        "assistant_turns_with_real_thinking": thinking_real,
        "host_path_replacements": host_paths_normalized,
        "max_length": a.max_length,
        "model": a.model,
        "source_roots": a.roots,
    }
    (out / "quality-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    manifest = dict(report)
    manifest["files"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.iterdir())
        if p.is_file()
    }
    (out / "package-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
