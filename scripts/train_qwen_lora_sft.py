#!/usr/bin/env python3
"""Train Qwen LoRA from a certified turn-level package, or run CPU preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.contracts._json import sha256_json


def _load_verified_package(package: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest_path = package / "training-package-manifest.json"
    config_path = package / "lora-config.json"
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(config_path.read_text())
    if sha256_json(config) != manifest.get("lora_config_checksum"):
        raise ValueError("LoRA config checksum does not match package manifest")
    if manifest.get("schema_version") == "teacher-sft-training-package/v6":
        target_modules = config.get("lora", {}).get("target_modules", [])
        if "lm_head" not in target_modules:
            raise ValueError(
                "v6 native tool-call training requires lm_head in LoRA target modules"
            )
        output_head_policy = manifest.get("output_head_policy", {})
        if output_head_policy.get("name") != "lora-lm-head/v1":
            raise ValueError("v6 package has no recognized output-head policy")
    dataset_path = package / config["dataset"]
    dataset_bytes = dataset_path.read_bytes()
    if hashlib.sha256(dataset_bytes).hexdigest() != manifest.get("train_turns_sha256"):
        raise ValueError("training dataset checksum does not match package manifest")
    rows = [json.loads(line) for line in dataset_bytes.splitlines() if line]
    if len(rows) != manifest.get("turn_example_count"):
        raise ValueError("training example count does not match package manifest")
    expected_tools = manifest.get("tools_checksum")
    if not isinstance(expected_tools, str):
        raise ValueError("package manifest has no tool contract checksum")
    for row in rows:
        tools = row.get("tools")
        if sha256_json(tools) != expected_tools:
            raise ValueError(f"tool contract checksum mismatch in turn {row.get('turn_id')}")
        if row.get("tools_checksum") != expected_tools:
            raise ValueError(f"persisted tool checksum mismatch in turn {row.get('turn_id')}")
    return manifest, config, rows


def _render(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_length: int,
) -> tuple[list[int], list[int]]:
    target = messages[-1]
    if target.get("role") != "assistant":
        raise ValueError("turn target must be the final assistant message")
    initial = messages[0]
    middle = list(messages[1:-1])
    while True:
        candidate = [initial, *middle, target]
        prompt = tokenizer.apply_chat_template(
            candidate[:-1], tools=tools, tokenize=True, add_generation_prompt=True
        )
        full = tokenizer.apply_chat_template(
            candidate, tools=tools, tokenize=True, add_generation_prompt=False
        )
        if full[: len(prompt)] != prompt:
            raise ValueError("chat template prompt is not a prefix of target sequence")
        if len(full) <= max_length:
            labels = [-100] * len(prompt) + full[len(prompt):]
            if not any(item != -100 for item in labels):
                raise ValueError("assistant target has no trainable tokens")
            return full, labels
        if not middle:
            raise ValueError("one turn exceeds max_length after context compression")
        middle.pop(0)
        while middle and middle[0].get("role") == "tool":
            middle.pop(0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--smoke-longest", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int)
    args = parser.parse_args()
    package_manifest, config, rows = _load_verified_package(args.package)

    from transformers import AutoConfig, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["student_model"], trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered = [
        _render(tokenizer, row["messages"], row["tools"], config["max_length"])
        for row in rows
    ]
    model_config = AutoConfig.from_pretrained(config["student_model"], trust_remote_code=False)
    maximum_token_id = max(max(item[0]) for item in rendered)
    stats = {
        "example_count": len(rendered),
        "min_tokens": min(len(item[0]) for item in rendered),
        "max_tokens": max(len(item[0]) for item in rendered),
        "mean_tokens": round(sum(len(item[0]) for item in rendered) / len(rendered), 2),
        "trainable_tokens": sum(sum(label != -100 for label in item[1]) for item in rendered),
        "tokenizer_size": len(tokenizer),
        "model_vocab_size": model_config.vocab_size,
        "maximum_token_id": maximum_token_id,
        "embedding_resize_required": len(tokenizer) > model_config.vocab_size,
    }
    if maximum_token_id >= len(tokenizer):
        raise ValueError("rendered token id exceeds tokenizer vocabulary")
    print(json.dumps({"preflight": "PASSED", **stats}, indent=2))
    if args.preflight_only:
        return 0
    if args.output is None:
        raise ValueError("--output is required unless --preflight-only is used")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output: {args.output}")
    if args.smoke_longest:
        rendered = [max(rendered, key=lambda item: len(item[0]))]
    effective_accumulation = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else config["gradient_accumulation_steps"]
    )
    if args.max_steps < 0 and len(rendered) % effective_accumulation:
        raise ValueError(
            "full training requires example_count divisible by gradient accumulation; "
            f"got {len(rendered)} and {effective_accumulation}"
        )

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    class Dataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(rendered)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            ids, labels = rendered[index]
            return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in batch)
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in batch:
            pad = width - len(item["input_ids"])
            result["input_ids"].append(item["input_ids"] + [tokenizer.pad_token_id] * pad)
            result["attention_mask"].append(item["attention_mask"] + [0] * pad)
            result["labels"].append(item["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}

    model = AutoModelForCausalLM.from_pretrained(
        config["student_model"], torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation="eager"
    )
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    lora = config["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM"
    ))
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
    trainer = Trainer(
        model=model,
        train_dataset=Dataset(),
        data_collator=collate,
        args=TrainingArguments(
            output_dir=str(args.output / "checkpoints"),
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=(
                effective_accumulation
            ),
            num_train_epochs=config["epochs"], learning_rate=config["learning_rate"],
            max_steps=args.max_steps,
            bf16=config["bf16"], fp16=False, gradient_checkpointing=config["gradient_checkpointing"],
            logging_steps=1,
            save_strategy="no" if args.max_steps > 0 else "epoch",
            report_to=[], dataloader_num_workers=0,
            remove_unused_columns=False,
            seed=42,
            data_seed=42,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        ),
    )
    training = trainer.train()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    output_files = {}
    for path in sorted(args.output.iterdir()):
        if path.is_file():
            output_files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    run_manifest = {
        "schema_version": "qwen-lora-sft-run/v1",
        "base_model": config["student_model"],
        "training_package": str(args.package.resolve()),
        "training_package_manifest_checksum": hashlib.sha256(
            (args.package / "training-package-manifest.json").read_bytes()
        ).hexdigest(),
        "train_turns_sha256": package_manifest["train_turns_sha256"],
        "lora_config": config,
        "preflight": stats,
        "max_steps_override": args.max_steps,
        "smoke_longest": args.smoke_longest,
        "seed": 42,
        "trainer_metrics": training.metrics,
        "output_files": output_files,
    }
    (args.output / "training-run.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
