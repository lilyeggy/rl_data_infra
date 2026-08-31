"""ARCHIVED: LoRA SFT training on the retired local-model chat dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

from experiments.local_model.harness import device_and_dtype, render_and_mask

DEFAULT_MODEL = "/root/models/qwen2.5-1.5b-instruct"
DEFAULT_DATA = Path(__file__).parent / "sft-dataset.jsonl"
DEFAULT_OUT = "/root/models/qwen-sft-adapter"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--no-bf16", action="store_true", help="force fp32 training")
    args = parser.parse_args()

    device, dtype = device_and_dtype()
    if args.no_bf16:
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = [
        json.loads(line)
        for line in Path(args.data).read_text().splitlines()
        if line.strip()
    ]
    dataset = []
    for ex in examples:
        ids, _positions, trainable = render_and_mask(tokenizer, ex["messages"])
        # Trainer shifts internally: CE(logits[:, :-1], labels[:, 1:]), so
        # labels[i] must be the token AT position i (masked for non-trainable).
        labels = [ids[i] if trainable[i] else -100 for i in range(len(ids))]
        dataset.append(
            {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "labels": labels,
            }
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, low_cpu_mem_usage=True, attn_implementation="eager"
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        ),
    )
    model.to(device)
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/sft-run",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            max_grad_norm=1.0,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            dataloader_num_workers=0,
            fp16=False,
            bf16=(device == "cuda" and not args.no_bf16),
        ),
        train_dataset=dataset,
    )
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("ADAPTER_SAVED=", args.output)


if __name__ == "__main__":
    main()
