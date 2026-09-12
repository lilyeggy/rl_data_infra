#!/usr/bin/env python3
"""Write the round's PolicyFingerprint so the loop and the manager agree on it.

Both the per-sample agent loop and the batch-certifying manager must bind the
same behaviour policy, and they run in different processes. Persisting the
fingerprint once per round and having both read it is what makes that
agreement checkable rather than assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--policy-generation", required=True, help="P0 / P1 / P2 ...")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tool-schema-json", required=True,
                        help="Tool schema captured from a real Pi CPU protocol probe")
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.training.policy_fingerprint import PolicyFingerprint
    from src.contracts._json import sha256_json

    adapter = Path(args.adapter)
    base = Path(args.base_model)
    base_files = sorted(base.glob("*.safetensors"))
    if not base_files:
        raise SystemExit("base model has no safetensors weights")
    base_manifest = {p.name: sha256_file(p) for p in base_files + [base / "config.json"]}
    # The model worker reads the base tokenizer. Bind that actual input, not a
    # possibly different tokenizer sitting next to the adapter.
    tokenizer_config = json.loads((base / "tokenizer_config.json").read_text())
    template = tokenizer_config.get("chat_template")
    if not isinstance(template, str) or not template:
        raise SystemExit("frozen base tokenizer has no traceable chat template")
    adapter_template = adapter / "chat_template.jinja"
    if not adapter_template.is_file() or adapter_template.read_text() != template:
        raise SystemExit("base template differs from the frozen SFT adapter template")
    tools = json.loads(Path(args.tool_schema_json).read_text())
    if not isinstance(tools, list) or len(tools) != 5:
        raise SystemExit("expected the five real Pi tools")
    fingerprint = PolicyFingerprint(
        provider="verl-vllm",
        model_id="qwen2.5-coder-14b",
        base_model_revision=sha256_json(base_manifest),
        adapter_revision=sha256_file(adapter / "adapter_model.safetensors"),
        tokenizer_revision=sha256_json({p.name: sha256_file(p) for p in
            (base / "tokenizer.json", base / "tokenizer_config.json")}),
        chat_template_checksum=hashlib.sha256(template.encode()).hexdigest(),
        tool_schema_checksum=sha256_json(tools),
        sampling_config={"temperature": 1.0, "top_p": 1.0},
        temperature=1.0,
        top_p=1.0,
        policy_generation=args.policy_generation,
    )
    if not fingerprint.is_complete():
        raise SystemExit(f"incomplete fingerprint: {fingerprint.missing_fields()}")
    Path(args.output).write_text(json.dumps(fingerprint.to_dict(), indent=2) + "\n")
    print(json.dumps({
        "output": args.output,
        "policy_generation": args.policy_generation,
        "checksum": fingerprint.checksum(),
        "adapter_revision": fingerprint.adapter_revision,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
