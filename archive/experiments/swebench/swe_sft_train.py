"""ARCHIVED: LoRA SFT on old SWE-bench teacher trajectories.

Same training skeleton as experiments/local_model/sft_train.py, but with a
render that (a) uses a large max_len for long multi-turn SWE trajectories and
(b) masks everything except assistant turns. The dataset system content already
embeds Pi's tool schemas, so we render as plain <|im_start|>role pieces (tools
are NOT re-injected) to stay byte-identical to local-qwen inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

DEFAULT_MODEL = "/root/rivermind-data/models/qwen2.5-7b-instruct"
DEFAULT_DATA = "/root/rivermind-data/swebench/swe-sft-dataset.jsonl"
DEFAULT_OUT = "/root/rivermind-data/models/qwen-swe-adapter-7b"
MAX_LEN = 3072


def render_and_mask(tokenizer, messages, max_len: int = MAX_LEN):
    input_ids: list[int] = []
    trainable: list[bool] = []
    for message in messages:
        piece = "<|im_start|>" + message["role"] + "\n" + message["content"] + "<|im_end|>\n"
        toks = tokenizer(piece, add_special_tokens=False)["input_ids"]
        input_ids.extend(toks)
        trainable.extend([message["role"] == "assistant"] * len(toks))
    input_ids = input_ids[:max_len]
    trainable = trainable[:max_len]
    return input_ids, trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    dataset = []
    for ex in examples:
        ids, trainable = render_and_mask(tokenizer, ex["messages"], args.max_len)
        labels = [ids[i] if trainable[i] else -100 for i in range(len(ids))]
        dataset.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    print(f"dataset: {len(dataset)} examples, lens={[len(d['input_ids']) for d in dataset]}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager"
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        ),
    )
    model.cuda()
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/swe-sft-run",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            max_grad_norm=1.0,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            dataloader_num_workers=0,
            fp16=False,
            bf16=True,
            gradient_checkpointing=True,
        ),
        train_dataset=dataset,
    )
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("ADAPTER_SAVED=", args.output)


if __name__ == "__main__":
    main()
