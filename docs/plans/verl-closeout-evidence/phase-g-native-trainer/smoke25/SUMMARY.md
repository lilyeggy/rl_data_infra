# smoke25 — the framework's own trainer ran one full GRPO step (P0 → P1)

`verlpi-smoke25`, 2026-09-11 14:34–14:45 UTC, `RayPPOTrainer` via `main_ppo`, 2×RTX PRO 6000.
Entry point: `scripts/run_verl_pi_train.sh verlpi-smoke25 <P0 adapter> P0 train-mbpp118.jsonl
actor_rollout_ref.rollout.n=16`.

## What ran

`verl.trainer.main_ppo` started the Ray cluster, the FSDP2 `AsyncActorRolloutRefWorker` (bf16, LoRA
r=8), the colocated vLLM engine (TP=2), and our `CertifiedVerlAgentLoopManager` +
`PiAgentLoopWorker` + `PiAgentLoop`. 16 real Pi episodes ran under bwrap against the engine through
our evidence-capturing proxy, the batch gate certified them, `RayPPOTrainer.fit()` computed GRPO
advantages, updated the actor, synced the weights into the live engine, and wrote a checkpoint.

## Evidence

| Claim | Evidence |
|---|---|
| Our loop ran inside the framework's trainer | `[pi-manager] … generation=0 prompt_rows=16` (`certify-lines.txt` shows per-episode certification) |
| 16 real multi-turn episodes, real tool rounds | `certify-lines.txt`: `calls` 1–5, `tool_rounds` 0–4, `seq_len` 736–5524 |
| Intra-group reward variance, no fabrication | `critic/score/mean: 0.0625` = 1/16 PASSED; `critic/score/min: 0.0`, `max: 1.0` |
| GRPO advantages were computed | `critic/advantages/mean: 0.1779`, `max: 3.75`, `min: -0.25` — impossible without variance |
| The trainer performed the update | `timing_s/update_actor: 117.57`, `actor/grad_norm: 0.1114` |
| The framework synced weights in place | `timing_s/update_weights: 2.98` (verl's `checkpoint_manager.update_weights`) |
| Checkpoint written by the framework's workers | `checkpoints.txt`: `global_step_1/actor/{model,optim,extra_state}_world_size_2_rank_*.pt`, `data.pt`, `already_checkpointed_iteration.txt` |
| **A real parameter change, not a no-op** | P1 adapter sha256 `f0ddcac8321b38712eb9b75c57ef375e0aa133e740fbfe12e5a928a1501776af` (137 714 872 B) vs frozen P0 `6db6a40c881b1896a061a0a51c36112a482f7dd8a659294c2471c248eb2458ae` (137 714 904 B) |
| Native logprobs match the trainer's recomputation | `training/rollout_probs_diff_valid: 1`, `training/rollout_actor_probs_pearson_corr: 0.99951` |
| Rollout length/shape came from our contract | `num_turns/mean: 2.31`, `max 5`; `response_length/mean: 2447`, `max: 5524`; `response_length/clip_ratio: 0.0` |

Run log: `/home/cxr/verl-closeout/verl-pi/smoke25.log`, sha256
`8b838424433e7f70107643ebbec87f3c367f4c5cd5e414180ee9009967b86bba`, 493 672 B.
It was not copied into the repo (mostly vLLM engine noise); the extracted lines above are the record.

## What this does *not* prove

* **§5.3 item 9 is still partial.** `update_weights` executed and is measured, but every rollout in
  this run happened *before* the sync, so no Pi episode consumed the updated weights. Proving that
  needs a second step, whose rollouts are bound to the checkpoint the framework just wrote.
* **The engine never confirms its own step.** With the shipped `checkpoint_engine.backend=naive`,
  `fsdp_workers.py:1735` discards `global_steps`, so every `TokenOutput` reports `None`. The
  transport records this per call instead of demanding it (29 such notes in the earlier smoke24 run),
  so policy identity rests on the trainer's step plus the on-disk checkpoint digest.
* **No effectiveness claim.** One step, one task (Mbpp/118), 1/16 solved — this shows the machinery,
  not learning.

## Non-fatal log artefact

The log ends with a `Traceback … RuntimeError: DataLoader worker (pid 436069) is killed by signal:
Killed` raised from `torchdata`'s `__del__` during interpreter shutdown, after
`Training Progress: 100%` and `Final validation metrics: None`. The step had already completed and
the checkpoint was written; this is a teardown message, not a training failure.
