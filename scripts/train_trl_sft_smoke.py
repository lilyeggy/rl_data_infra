import argparse, hashlib, json
from pathlib import Path
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Config
from trl import SFTConfig, SFTTrainer

def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--tiny-random-model", action="store_true")
    parser.add_argument("--use-cpu", action="store_true")
    args = parser.parse_args()
    ds = datasets.load_dataset("json", data_files=args.dataset, split="train")
    ds = ds.select_columns(["prompt", "completion", "tools"])
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if args.tiny_random_model:
        config = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192, tie_word_embeddings=False)
        model = AutoModelForCausalLM.from_config(config); peft_config = None
    else:
        from peft import LoraConfig; model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="sdpa"); peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    training_args = SFTConfig(output_dir=args.output_dir, max_steps=args.max_steps, per_device_train_batch_size=args.batch_size, learning_rate=args.learning_rate, logging_steps=1, save_steps=args.max_steps, save_total_limit=1, save_strategy="no", report_to="none", use_cpu=args.use_cpu, bf16=(not args.use_cpu), completion_only_loss=True, assistant_only_loss=False, max_length=args.max_length, gradient_checkpointing=(not args.use_cpu), packing=False)
    trainer = SFTTrainer(model=model, args=training_args, train_dataset=ds, processing_class=tok, peft_config=peft_config)
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    losses = [entry for entry in trainer.state.log_history if "loss" in entry]
    record = {"format": "trl-sft-smoke/v1", "model": args.model, "dataset": args.dataset, "dataset_sha256": sha_file(args.dataset), "rows": len(ds), "max_steps": args.max_steps, "loss_history": losses, "output_dir": args.output_dir}
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
