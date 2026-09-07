"""ARCHIVED: build an SFT dataset from old SWE-bench teacher trajectories.

Source: package-v3/episodes.jsonl (canonical AgentEpisode, Pi message format).
Only SUCCESS episodes (teacher actually resolved the instance) are used — these
are the verified trajectories our data plane produced.

Output format matches experiments/local_model/harness.render_and_mask: a
{"messages":[...]} list where the system content already embeds Pi's tool
schemas (captured from a real local-qwen prompt), assistant turns are
`<tool_call>` blocks / final text (trainable), and tool results are user turns
`<tool_response>...</tool_response>` (masked). Tool results are truncated to
keep sequences bounded.

Writes swebench/swe-sft-dataset.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/agentic-rl")

EPISODES = Path("/root/rivermind-data/swebench/package-v3/episodes.jsonl")
OUT = Path("/root/rivermind-data/swebench/swe-sft-dataset.jsonl")

# Compact system prompt: the full Pi prompt (~2100 tokens, mostly tool-schema
# JSON + docs) both bloats every sequence and OOMs the LM head. The system turn
# is masked (never trained), and the base model already parses the full prompt
# at inference, so we keep only the tool roster + tool_call format — the SFT
# signal is the WORKFLOW (which tools, what order, to find & fix the bug).
SYSTEM = (
    "You are an expert coding assistant operating inside pi, a coding agent harness.\n"
    "Available tools: read, bash, write, edit, grep, find, ls, glob.\n"
    "grep/find may be unavailable; prefer `bash` (e.g. grep/rg/find via shell) and "
    "`read` to explore, then `edit`/`write` to fix, then validate with `bash` tests.\n"
    "For each function call, return a json object within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
)

TOOL_RESULT_MAX = 400  # chars; file dumps are the bloat, keep them bounded
KEEP_LAST_TURNS = 14   # keep the tail (exploration→edit→validate); the fix is at the end


def _text_of(content) -> str:
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def convert(messages: list[dict], system: str) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        role = m.get("role")
        content = m.get("content") or []
        if role == "user" and not m.get("tool_name"):
            out.append({"role": "user", "content": _text_of(content)})
        elif role == "assistant":
            parts = [p for p in content if isinstance(p, dict)]
            toolcalls = [p for p in parts if p.get("type") == "toolCall"]
            texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
            if toolcalls:
                blocks = [
                    "<tool_call>\n"
                    + json.dumps({"name": p.get("name"), "arguments": p.get("arguments")}, ensure_ascii=False)
                    + "\n</tool_call>"
                    for p in toolcalls
                ]
                out.append({"role": "assistant", "content": "\n".join(blocks)})
            elif any(t.strip() for t in texts):
                out.append({"role": "assistant", "content": " ".join(texts).strip()})
        elif role == "toolResult":
            text = _text_of(content)
            if len(text) > TOOL_RESULT_MAX:
                text = text[:TOOL_RESULT_MAX] + "\n...[truncated]"
            out.append({"role": "user", "content": "<tool_response>\n" + text + "\n</tool_response>"})
    return out


def main() -> None:
    system = SYSTEM
    episodes = [json.loads(l) for l in EPISODES.read_text().splitlines() if l.strip()]
    n_written = 0
    with OUT.open("w") as f:
        for ep in episodes:
            if ep["outcome"]["task_status"] != "SUCCESS":
                continue
            iid = ep["task_id"].split("/")[-1]
            # the message-carrying event with the most messages = full trajectory
            msg_evs = [e for e in ep["events"] if isinstance(e.get("attributes", {}).get("messages"), list)]
            if not msg_evs:
                continue
            msgs = max(msg_evs, key=lambda e: len(e["attributes"]["messages"]))["attributes"]["messages"]
            conv = convert(msgs, system)
            # keep system + first user (the problem) + the LAST few turns (the fix)
            if len(conv) > KEEP_LAST_TURNS + 2:
                conv = conv[:2] + conv[-KEEP_LAST_TURNS:]
            n_assistant = sum(1 for m in conv if m["role"] == "assistant")
            if n_assistant < 2:
                continue
            f.write(json.dumps({"instance": iid, "messages": conv}, ensure_ascii=False) + "\n")
            n_written += 1
            print(f"  {iid}: {len(conv)} msgs, {n_assistant} assistant turns", flush=True)
    print(f"WROTE {n_written} examples -> {OUT}")


if __name__ == "__main__":
    main()
