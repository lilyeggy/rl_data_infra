import argparse, hashlib, json
from pathlib import Path
VERSION = "sft-messages/v1"

# comment line
def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def sha_json(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def to_message(role, blocks, pending):
    role = {"toolResult": "tool", "tool_result": "tool"}.get(role, role)
    texts = []
    calls = []
    if not isinstance(blocks, list):
        return None
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif kind in ("toolCall", "tool_call"):
            name = block.get("name")
            args = block.get("arguments")
            if not isinstance(name, str) or not isinstance(args, dict):
                continue
            call_id = block.get("id")
            if not isinstance(call_id, str) or not call_id:
                call_id = "call_" + sha_json([name, args])[:20]
            calls.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False, sort_keys=True)}})
    content = "".join(texts)
    message = {"role": role, "content": content}
    if calls:
        message["tool_calls"] = calls
        pending[:] = [call["id"] for call in calls]
    if role == "tool" and not content:
        return None
    if role == "tool":
        message["tool_call_id"] = pending.pop(0) if pending else "call_" + sha_json(content)[:20]
    return message

def split_prompt(text):
    messages = []
    pending = []
    for line in text.splitlines():
        if not line.startswith("["):
            continue
        end = line.find("] ")
        if end < 0:
            continue
        role = line[1:end]
        body = line[end+2:]
        try:
            blocks = json.loads(body)
        except json.JSONDecodeError:
            continue
        message = to_message(role, blocks, pending)
        if message is not None:
            messages.append(message)
    return messages

def parse_completion(text, pending):
    try:
        blocks = json.loads(text)
    except json.JSONDecodeError:
        return None
    return to_message("assistant", blocks, pending)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--tools", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--text-only", action="store_true")
    args = parser.parse_args()
    tools = json.loads(args.tools.read_text(encoding="utf-8"))
    if not isinstance(tools, list) or not tools:
        raise SystemExit("tools must be a non-empty JSON list")
    rows = []
    skipped = {}
    def skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1
    with args.examples.open("r", encoding="utf-8") as handle:
        for line in handle:
            if args.limit and len(rows) >= args.limit:
                break
            if not line.strip():
                continue
            try:
                source = json.loads(line)
            except json.JSONDecodeError:
                skip("source_json_decode")
                continue
            prompt = source.get("prompt_text")
            completion_text = source.get("completion_text")
            if not isinstance(prompt, str) or not isinstance(completion_text, str):
                skip("missing_fields")
                continue
            messages = split_prompt(prompt)
            if not (messages and messages[0].get("role") == "user"):
                skip("not_user_first")
                continue
            pending = []
            for message in messages:
                if message.get("tool_calls"):
                    pending = [call["id"] for call in message["tool_calls"]]
                elif message.get("role") == "tool" and pending:
                    pending.pop(0)
            completion = parse_completion(completion_text, pending)
            if completion is None:
                skip("invalid_completion")
                continue
            has_content = bool(completion.get("content"))
            has_calls = bool(completion.get("tool_calls"))
            if not has_content and not has_calls:
                skip("empty_completion")
                continue
            if args.text_only and (not has_content or has_calls):
                skip("text_only")
                continue
            rows.append({"task_id": source.get("task_id"), "episode_id": source.get("episode_id"), "messages": messages + [completion], "prompt": messages, "completion": [completion], "tools": tools, "source_row_sha256": sha_json(source)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + chr(10))
    manifest = {"format": VERSION, "source": str(args.examples), "source_sha256": sha_file(args.examples), "tool_schema_sha256": sha_json(tools), "output": str(args.output), "output_sha256": sha_file(args.output), "rows": len(rows), "skipped": skipped}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
