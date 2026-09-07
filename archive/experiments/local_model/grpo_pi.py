"""ARCHIVED: experimental GRPO through the real Pi harness; not production RL.

The local student model is served by ``openai_server.py``; Pi drives it and the
server logs each turn's ``prompt_ids`` / ``completion_ids`` / behavior logprobs.
The trainer reads those rollouts, scores them with the strict verifier, and does
a PPO-style GRPO update on the LoRA adapter.

A single RTX 4090 (24GB) cannot hold the 7B inference server AND the 7B training
process at once, so the loop ALTERNATES: server up (collect rollouts) -> server
down (train) -> server up with the new adapter. This is the standard decoupled
inference/training + weight-sync pattern of an agentic-RL stack.

Run on the server: PYTHONPATH=/root/agentic-rl python3 experiments/local_model/grpo_pi.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from experiments.local_model.pi_local import (
    MODEL,
    PROVIDER,
    run_task,
)
from src.task_suite import ALL_TASKS

MODEL_PATH = os.environ.get("LOCAL_MODEL", "/root/rivermind-data/models/qwen2.5-7b-instruct")
ROLLOUT_LOG = os.environ.get("ROLLOUT_LOG", "/root/rivermind-data/qwen-rollout-log.jsonl")
START_SCRIPT = "/root/start-qwen-server.sh"
SERVER = "http://127.0.0.1:8000"


# ---------- server lifecycle ----------
def _post(path, payload):
    req = urllib.request.Request(
        SERVER + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception:
        return {}


def start_server(adapter: str):
    subprocess.run(["pkill", "-f", "openai_server.py"], check=False)
    time.sleep(3)
    env = dict(os.environ)
    env["LOCAL_ADAPTER"] = adapter
    env["PORT"] = "8000"
    env["QWEN_SERV_TEMPERATURE"] = "0.0"
    subprocess.Popen(
        ["setsid", "nohup", "python3", "experiments/local_model/openai_server.py"],
        cwd="/root/agentic-rl",
        env=env,
        stdout=open("/root/rivermind-data/qwen-server.log", "a"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )
    for _ in range(60):
        try:
            with urllib.request.urlopen(SERVER + "/health", timeout=5) as r:
                if json.loads(r.read()).get("ok"):
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def stop_server():
    # SIGKILL and wait until GPU memory is actually released before training.
    subprocess.run(["pkill", "-9", "-f", "openai_server.py"], check=False)
    for _ in range(40):
        time.sleep(2)
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False,
            ).stdout.strip().splitlines()
            used = int(out[0]) if out and out[0] else 0
            if used < 2000:  # server freed its ~15GB
                break
        except Exception:
            break
    torch.cuda.empty_cache()


def set_sampling(temp: float):
    _post("/v1/set_sampling", {"temperature": temp})


# ---------- rollout collection (server UP) ----------
def _log_offset() -> int:
    if not Path(ROLLOUT_LOG).exists():
        return 0
    with open(ROLLOUT_LOG) as f:
        return sum(1 for _ in f)


def _read_log_from(offset: int):
    entries = []
    if not Path(ROLLOUT_LOG).exists():
        return entries
    with open(ROLLOUT_LOG) as f:
        for i, line in enumerate(f):
            if i >= offset and line.strip():
                entries.append(json.loads(line))
    return entries


def collect_rollouts(tasks, group, ws, adapter, raw_dir):
    """Run each task `group` times through real Pi (server sampling on).
    Returns list of {task, reward, turns:[{prompt_ids,completion_ids,old_logprobs}]}."""
    assert start_server(adapter), "server failed to start"
    set_sampling(0.8)
    rollouts = []
    for t in tasks:
        for g in range(group):
            for attempt in (0, 1):
                off = _log_offset()
                try:
                    records, reward, decl = run_task(
                        t, ws, timeout_seconds=240, raw_output_dir=raw_dir, tag=f"grpo-t{t}-g{g}"
                    )
                except Exception as exc:  # noqa: BLE001
                    # Fail closed: an infra exception is never reward 0.
                    print(f"  rollout t{t} g{g}: ERROR {exc}", flush=True)
                    reward, records = None, []
                turns = _read_log_from(off)
                if turns or attempt == 1:
                    break
                # 0 turns => server likely died mid-rollout; restart and retry once
                if reward is not None:
                    print(f"  rollout t{t} g{g}: 0 turns, restarting server", flush=True)
                    start_server(adapter)
                    set_sampling(0.8)
            rollouts.append({"task": t, "reward": reward, "turns": turns})
            print(f"  rollout t{t} g{g}: reward={reward} turns={len(turns)}", flush=True)
    stop_server()
    return rollouts


# ---------- GRPO update (server DOWN) ----------
def _logprobs(model, prompt_ids, completion_ids, device):
    """Per-token logprobs of completion_ids. Selects only the completion
    positions' logits BEFORE the float32 softmax to avoid materialising a
    [seq, vocab] float32 tensor (the OOM source on long multi-turn ctx)."""
    full = torch.tensor([prompt_ids + completion_ids], device=device)
    n_in = len(prompt_ids)
    n_out = len(completion_ids)
    logits = model(full).logits[0]  # [L, V] bf16
    pos = torch.arange(n_in, full.shape[1], device=device)
    sel = logits[pos - 1]  # [n_out, V] bf16 — only completion positions
    logp = torch.log_softmax(sel.float(), dim=-1)  # [n_out, V] float32 (small)
    tgt = torch.tensor(completion_ids, device=device)
    return logp[torch.arange(n_out, device=device), tgt]  # [n_out]


def grpo_train(rollouts, sft_adapter, out_adapter, *, lr, clip, beta, epochs, device):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager"
    )
    # policy = base + trainable LoRA initialised from the current (SFT) adapter
    policy = PeftModel.from_pretrained(base, sft_adapter, is_trainable=True).to(device)
    policy.train()
    if hasattr(policy, "gradient_checkpointing_enable"):
        policy.gradient_checkpointing_enable()
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)

    # per-task group baseline (GRPO advantage), computed only over VALID
    # rollouts.  Infra-invalid rollouts (reward=None) are excluded entirely.
    from collections import defaultdict

    valid_rollouts = [r for r in rollouts if r.get("reward") is not None]
    by_task = defaultdict(list)
    for r in valid_rollouts:
        by_task[r["task"]].append(r["reward"])
    stats = {t: (sum(v) / len(v), (torch.tensor(v).std().item() or 0.0)) for t, v in by_task.items() if len(v) > 0}
    print(
        f"  GRPO: {len(valid_rollouts)}/{len(rollouts)} rollouts valid; "
        f"excluded {len(rollouts) - len(valid_rollouts)} infra-invalid", flush=True
    )

    # Cache old (behavior-policy) logprobs once, before any update. The policy
    # starts at the same adapter that generated the rollouts, so old == the
    # collecting distribution at the natural temp=1.
    for r in rollouts:
        for turn in r["turns"]:
            with torch.no_grad():
                turn["old"] = _logprobs(
                    policy, turn["prompt_ids"], turn["completion_ids"], device
                ).detach()
            torch.cuda.empty_cache()

    for epoch in range(epochs):
        total = 0.0
        ntok = 0
        # group rollouts by task; step per task to bound gradient-accumulation memory
        from collections import defaultdict as _dd
        per_task = _dd(list)
        for r in rollouts:
            per_task[r["task"]].append(r)
        for t, rs in per_task.items():
            if t not in stats:
                # all rollouts for this task were infra-invalid; nothing to train
                continue
            mean, std = stats[t]
            opt.zero_grad()
            for r in rs:
                if r.get("reward") is None:
                    # infra-invalid rollout: excluded from gradient/advantage
                    continue
                adv = (r["reward"] - mean) / (std + 1e-6) if std > 1e-6 else 0.0
                if not r["turns"]:
                    continue
                for turn in r["turns"]:
                    old = turn["old"]
                    new = _logprobs(policy, turn["prompt_ids"], turn["completion_ids"], device)
                    logratio = new - old
                    ratio = torch.exp(logratio)
                    s1 = ratio * adv
                    s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
                    pg = -torch.mean(torch.min(s1, s2))
                    kl = torch.mean(torch.exp(-logratio) + logratio - 1)  # k3, KL(new||old)
                    loss = pg + beta * kl
                    (loss / max(1, len(r["turns"]))).backward()
                    total += loss.item()
                    ntok += int(new.shape[0])
                    del new, old, ratio, loss
                    torch.cuda.empty_cache()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        print(f"  epoch {epoch}: mean_loss={total / max(1, len(rollouts)):.4f} tokens={ntok}", flush=True)

    out = Path(out_adapter)
    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out)
    del policy, base, opt
    torch.cuda.empty_cache()
    return str(out)


def evaluate(adapter, ws, device):
    assert start_server(adapter), "server failed to start (eval)"
    set_sampling(0.0)  # greedy
    res = []
    for t in ALL_TASKS:
        try:
            _, reward, _ = run_task(t, ws, timeout_seconds=240)
        except Exception:  # noqa: BLE001
            reward = 0.0
        res.append((t, reward))
        print(f"  eval t{t}: reward={reward}", flush=True)
    stop_server()
    full = sum(1 for _, r in res if r == 1.0)
    print(f"GRPO_PI_EVAL full={full}/{len(res)} -> {json.dumps({t: r for t, r in res})}", flush=True)
    return full, res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft", default="/root/rivermind-data/models/qwen-sft-adapter-7b-v3")
    p.add_argument("--output", default="/root/rivermind-data/models/qwen-grpo-pi-7b")
    p.add_argument("--workspace", default="/root/rivermind-data/pi-grpo-ws")
    p.add_argument("--raw-dir", default="/root/rivermind-data/pi-grpo-raw")
    p.add_argument("--tasks", default="6,7,10,11,12,13,14,15")
    p.add_argument("--group", type=int, default=6)
    p.add_argument("--outer", type=int, default=3)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--probe", action="store_true", help="only collect + report reward distribution, no training")
    args = p.parse_args()

    ws = Path(args.workspace)
    raw = Path(args.raw_dir)
    tasks = [int(t) for t in args.tasks.split(",")]
    device = "cuda"

    if args.probe:
        Path(ROLLOUT_LOG).unlink(missing_ok=True)
        rollouts = collect_rollouts(tasks, args.group, ws, args.sft, raw)
        from experiments.local_model.rollout_trust import evaluate_on_policy_certification

        cert = evaluate_on_policy_certification(rollouts, exact_behavior_logprobs=False)
        print("ON_POLICY_CERTIFICATION", cert.on_policy_certification, cert.reasons, flush=True)
        from collections import defaultdict
        by_task = defaultdict(list)
        for r in rollouts:
            by_task[r["task"]].append(r["reward"])
        print("PROBE reward distribution per task:")
        for t in tasks:
            v = by_task.get(t, [])
            full = sum(1 for x in v if x == 1.0)
            print(f"  task {t}: n={len(v)} full={full} rewards={v}")
        return

    current = args.sft
    for outer in range(args.outer):
        print(f"=== OUTER {outer}: collect (adapter={current}) ===", flush=True)
        rollouts = collect_rollouts(tasks, args.group, ws, current, raw)
        from experiments.local_model.rollout_trust import evaluate_on_policy_certification

        cert = evaluate_on_policy_certification(rollouts, exact_behavior_logprobs=False)
        print(
            f"  ON_POLICY_CERTIFICATION={cert.on_policy_certification} "
            f"valid={cert.valid}/{cert.total} reasons={cert.reasons}",
            flush=True,
        )
        mean_r = sum(r["reward"] for r in rollouts if r["reward"] is not None) / max(1, cert.valid)
        print(f"  collected {len(rollouts)} rollouts, mean_reward={mean_r:.3f}", flush=True)
        print(f"=== OUTER {outer}: train ===", flush=True)
        nxt = f"{args.output}-o{outer}"
        grpo_train(
            rollouts, current, nxt,
            lr=args.lr, clip=args.clip, beta=args.beta, epochs=args.epochs, device=device,
        )
        current = nxt

    print("=== FINAL EVAL ===", flush=True)
    evaluate(current, ws, device)
    print("GRPO_PI_ADAPTER=" + current, flush=True)


if __name__ == "__main__":
    main()
