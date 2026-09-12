# smoke27 — the framework's trainer ran two steps, and step 2's rollout used the weights step 1 synced

`verlpi-smoke27`, 2026-09-11 16:15–16:38 CST (22 min 34 s wall), `RayPPOTrainer` via `main_ppo`,
2×RTX PRO 6000. Entry point:

```
scripts/run_verl_pi_train.sh verlpi-smoke27 <P0 adapter> P0 train-mbpp118.jsonl
```

with the launcher's own budgets: `rollout.n=32`, `trainer.total_epochs=2`,
`pi_certification.max_attempts=3`. Log: `/home/cxr/verl-closeout/verl-pi/smoke27.log`
sha256 `886ad180609c8332a2ed4d0c165ce01930a24c6eee45fb8a9df84001554bd219`.

## Why this run exists

smoke25 proved one framework-owned GRPO step (acceptance item 2). It could not prove item 9 —
"新权重经框架同步后被 Pi 使用" — because with `total_epochs=1` every rollout preceded the single
weight sync, so no Pi episode ever ran against synced weights. Reaching step 2 needed a batch that
certifies at step 1, and two things stood in the way:

1. `max_attempts` was configuration that could not take effect. The attempt loop re-raised inside
   its own `except`, so attempt 1 was unreachable and `last_error` was dead code. One variance-free
   draw ended the run — measured in smoke26 (0/16 solved, whole round spent). Fixed in `bc06a5f`:
   only a flat reward group is redrawn, every other violation still fails on its first occurrence.
2. `rollout.n=4` was sized from the closed-out 4-episode scope, not from the measured solve rate.
   The frozen P0 solves Mbpp/118 about once in 16 draws (3/48 across smoke24–26), so n=4 certified
   ~22% of the time. n=32 puts one attempt at ~87%.

## What ran

Engine, workers and loop as in smoke25. Generation 0 rolled out 32 episodes under **P0**, the gate
refused the first draw and the manager redrew, the second draw certified, `RayPPOTrainer.fit()`
computed GRPO advantages, updated the actor, **synced the weights in place**, and wrote
`global_step_1`. Generation 1 then rolled out 32 fresh episodes under **P1** — the weights the
framework had just synced — certified, updated, and wrote `global_step_2`.

## Evidence

| Claim | Evidence |
|---|---|
| The redraw is real, not a longer first try | `resample.txt`: `attempt 0 carries no learning signal, redrawing: group '53ca36e3…' has no intra-group reward variance`, plus `rejected-attempt0.json` (`reason: DegenerateBatchError`). `gen-000` has two attempt directories; `gen-001` certified on its first |
| A refused draw costs a draw, not the run | smoke26 died on exactly this condition; here the run continued to a second step |
| Both steps are the framework's own loop | `step-metrics.txt`: `step:1` … `step:2`, `local_global_step_folder: …/global_step_1` and `…/global_step_2`, `Training Progress: 100%\|██\| 2/2 [22:34]` |
| The update is the trainer's | `timing_s/update_actor` 186.81 s (step 1) / 188.06 s (step 2); `actor/grad_norm` 0.1194 / 0.1009 |
| **The framework synced weights in place** | `timing_s/update_weights` 2.90 s (step 1) and 2.83 s (step 2) — `checkpoint_manager.update_weights`, i.e. the LoRA delta pushed into the live engine |
| **A later rollout consumed the synced weights** | `gen-001` ran under `policy_generation: P1` with `adapter_revision: 13cc8199…`, and `checkpoints.txt` shows the recomputed digest of `global_step_1/actor` is `13cc8199…` → **MATCH**. `gen-000` by contrast ran under `P0` / `6db6a40c…`, the frozen adapter handed to the launcher |
| The two rollouts really had different behaviour policies | episode fingerprints `e3eca6fe…` (gen-000) vs `aed2a953…` (gen-001), each equal to its round policy's checksum — enforced by the gate, which rejects a batch whose samples carry another fingerprint |
| Step 2's gradient came from P1 rollouts | `global_step_2`'s adapter sha256 `30e3f3ad…` differs from `global_step_1`'s `15a741e7…`, and step 2's only rollout was `gen-001` |
| Two genuinely real parameter changes | `checkpoints.txt`: P1 `15a741e266048f11…`, P2 `30e3f3ad35c3c554…`, both ≠ frozen P0 `6db6a40c881b1896…`; all three are 137 714 872 B |
| Real Pi, real tool rounds, under the framework | `certify-lines.txt`: `calls` 1–5, `tool_rounds` 0–4; `timing_s/agent_loop/tool_calls/mean` 0.8125 (step 1) / 0.875 (step 2); `num_turns/mean` 1.8125 / 1.875, max 5 |
| Intra-group variance, no fabrication | `critic/score/mean` 0.0625 = 2/32 solved (step 1), 0.03125 = 1/32 (step 2); `critic/score/min` 0.0, `max` 1.0 both steps; `critic/advantages/mean` 0.3520 / 0.1148 |
| Native logprobs survive the framework round trip | `training/rollout_probs_diff_valid: 1`, `training/rollout_actor_probs_pearson_corr` 0.99957 (step 1) / 0.99959 (step 2) |
| Nothing was clipped or aborted | `response_length/clip_ratio: 0.0`, `response/aborted_ratio: 0.0` both steps |
| Episode shape per draw (for the yield argument) | `gen-000/attempt-0` 32 / 15 multi-turn / 0 solved; `gen-000/attempt-1` 32 / 14 / 2; `gen-001/attempt-0` 32 / 16 / 1 |

## Reading of the result

Item 9 is satisfied in the form the framework permits. The engine still cannot self-report which
step it serves — with `checkpoint_engine.backend=naive` (the shipped default), `fsdp_workers.py:1735`
discards `global_steps`, so `set_global_steps` never runs and every `TokenOutput` reports
`global_steps: None`; that gap is recorded in `engine-notes/<episode>.step.jsonl` rather than
assumed away. What binds the identity is the trainer's own artifact: the manager computes the
digest of the checkpoint **the framework wrote**, and step 2's rollout is stamped with it, at every
level (round policy, per-episode fingerprint, and the batch gate that refuses any other). Step 2's
optimizer step then demonstrably depended on those rollouts.

Not claimed: that the policy improved. Two solves in 32 draws then one in 32 is noise, the holdout
evaluation is F's job, and this run is a pipeline result.

## Teardown

`Training Progress: 100%` and `Final validation metrics: None` both precede a
`torchdata` `__del__` `RuntimeError: DataLoader worker … killed by signal: Killed` under
`Exception ignored in:` — the same teardown noise recorded on smoke25 and smoke26, after the run
finished. Verified after exit: no `cxr`-owned `main_ppo`/`raylet`/`EngineCore`/`vLLMHttpServer`
processes, no GPU compute allocations, Ray ports free. The GGUF-free host's foreign job was absent
throughout this run (both GPUs showed no compute processes before launch).
