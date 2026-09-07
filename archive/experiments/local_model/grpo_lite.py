"""ARCHIVED: GRPO-lite smoke; not certified as the production RL path.

- policy: SFT-merged weights + fresh trainable LoRA (starts at SFT)
- reference: same model with LoRA adapter disabled (KL anchor, single model)
- reward: verifier pass=1.0; recovery path executed=0.5; else 0
- advantage: per-task group normalization
- loss: -min(ratio, clip(ratio, 1±eps)) * adv + beta * KL, per token
- memory: per-rollout gradient accumulation (safe under small cgroup limits)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model

from experiments.local_model.harness import (
    EVAL_TASKS,
    TRAIN_TASKS,
    MAX_LEN,
    device_and_dtype,
    load_model,
    render_and_mask,
    rollout,
)

DEFAULT_BASE = "/root/models/qwen2.5-1.5b-instruct"
DEFAULT_SFT = "/root/models/qwen-sft-adapter"
DEFAULT_OUT = "/root/models/qwen-grpo-adapter"
G = 3
K = 3
TEMP = 0.4
TOP_P = 0.9
CLIP = 0.2
BETA = 0.05
LR = 1e-5


def logps_at(model, ids, positions, device, disable_adapter=False):
    with torch.no_grad():
        if disable_adapter:
            with model.disable_adapter():
                logits = model(torch.tensor([ids], dtype=torch.long).to(device)).logits[0].float()
        else:
            logits = model(torch.tensor([ids], dtype=torch.long).to(device)).logits[0].float()
    logp = F.log_softmax(logits, dim=-1)
    return [logp[t, ids[t + 1]].item() for t in positions]


def answer_positions(tokenizer, messages, max_len):
    """Token offsets that predict the content of the FINAL assistant message
    only (the answer). Tool-protocol assistant tokens are excluded so RL does
    not reward-hack / unlearn the learned tool protocol (credit assignment:
    the verifier outcome is determined by the final answer).
    """
    last_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_idx = i
            break
    input_ids = []
    start = None
    for i, m in enumerate(messages):
        piece = "<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n"
        toks = tokenizer(piece, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
        if i == last_idx:
            start = len(input_ids)
        input_ids.extend(toks)
        if len(input_ids) >= max_len:
            break
    input_ids = input_ids[:max_len]
    if start is None:
        return input_ids, []
    positions = [t for t in range(start, len(input_ids) - 1)]
    return input_ids, positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_BASE)
    parser.add_argument("--sft", default=DEFAULT_SFT)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--workspace", default="/tmp/local-harness-workspace")
    parser.add_argument(
        "--train-tasks",
        default=",".join(str(t) for t in TRAIN_TASKS),
        help="task ids for on-policy rollouts",
    )
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--clip", type=float, default=CLIP)
    parser.add_argument("--group", type=int, default=G)
    parser.add_argument("--steps", type=int, default=K)
    parser.add_argument("--temp", type=float, default=TEMP)
    parser.add_argument(
        "--answer-only",
        action="store_true",
        help="apply RL objective only to final-answer tokens (freeze tool protocol)",
    )
    args = parser.parse_args()
    train_tasks = tuple(int(t) for t in args.train_tasks.split(","))
    G_, K_, LR_, BETA_, CLIP_, TEMP_ = (
        args.group, args.steps, args.lr, args.beta, args.clip, args.temp,
    )

    device, dtype = device_and_dtype()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    merged = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, low_cpu_mem_usage=True),
        args.sft,
    ).merge_and_unload()
    policy = get_peft_model(
        merged,
        LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        ),
    ).to(device)
    optimizer = AdamW(policy.parameters(), lr=LR_)
    ws_root = Path(args.workspace)

    for step in range(K_):
        print("STEP", step, flush=True)
        data = []
        policy.eval()
        for task_index in train_tasks:
            for _ in range(G_):
                messages, reward, status = rollout(
                    policy, tokenizer, ws_root, task_index, sample=True, device=device,
                    temperature=TEMP_, top_p=TOP_P,
                )
                ids, positions, _trainable = render_and_mask(tokenizer, messages)
                if args.answer_only:
                    ids, positions = answer_positions(tokenizer, messages, MAX_LEN)
                old = logps_at(policy, ids, positions, device)
                ref = logps_at(policy, ids, positions, device, disable_adapter=True)
                data.append({"task": task_index, "ids": ids, "positions": positions,
                             "old": old, "ref": ref, "reward": reward})
                print("  rollout task", task_index, "reward=", reward, "status=", status, flush=True)
        groups = {}
        for d in data:
            groups.setdefault(d["task"], []).append(d["reward"])
        mean = {t: sum(v) / len(v) for t, v in groups.items()}
        std = {t: (sum((x - mean[t]) ** 2 for x in v) / len(v)) ** 0.5 for t, v in groups.items()}
        for d in data:
            s = std[d["task"]]
            d["adv"] = (d["reward"] - mean[d["task"]]) / (s + 1e-4) if s > 1e-6 else 0.0

        optimizer.zero_grad()
        policy.train()
        for d in data:
            ids = d["ids"]
            logits = policy(torch.tensor([ids], dtype=torch.long).to(device)).logits[0].float()
            logp = F.log_softmax(logits, dim=-1)
            loss = torch.tensor(0.0)
            n = 0
            for i, t in enumerate(d["positions"]):
                new_lp = logp[t, ids[t + 1]]
                old_lp = d["old"][i]
                ref_lp = d["ref"][i]
                ratio = torch.exp(new_lp - old_lp)
                clipped = torch.clamp(ratio, 1 - CLIP_, 1 + CLIP_)
                kl = torch.exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1.0
                loss = loss + (-torch.min(ratio, clipped) * d["adv"] + BETA_ * kl)
                n += 1
            if n:
                (loss / n).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        print("  step done", flush=True)

    policy.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("GRPO_ADAPTER_SAVED=", args.output, flush=True)

    policy.eval()
    results = []
    for task_index in EVAL_TASKS:
        messages, reward, status = rollout(
            policy, tokenizer, ws_root, task_index, sample=False, device=device
        )
        ok = reward == 1.0
        results.append(ok)
        print("EVAL task", task_index, "success=", ok, "status=", status, flush=True)
    print("GRPO_EVAL_SUMMARY success=", sum(results), "/", len(results), results, flush=True)


if __name__ == "__main__":
    main()
