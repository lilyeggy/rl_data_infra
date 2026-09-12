# smoke26 — the variance gate refused a degenerate batch (P0 only, no update)

`verlpi-smoke26`, 2026-09-11 14:48 UTC, same command as smoke25 plus `trainer.total_epochs=2`
(`Total training steps: 2` was announced), `actor_rollout_ref.rollout.n=16`.

**Outcome: refused, correctly, and no weights were touched.**

All 16 episodes were certified individually (`certify-and-refusal.txt`, one line each: `calls` 1–2,
`seq_len` 1024–2301) and every one failed the task, so the group had a single reward value:

```
ContractValidationError: group 'd3fa719d-8b26-4ff0-8a8a-acf19c22c020' has no intra-group reward variance
verlpi-smoke26: batch rejected without resampling: group '…' has no intra-group reward variance
```

This is the gate doing its job: GRPO's advantage is `(r - mean)/std` within the group, and a group
with one distinct reward carries no learning signal, so it must not reach the optimizer. It is
§5.3 item 6 ("certification runs before training and failure cannot be bypassed") exercised on the
framework path, and it cost nothing: no actor update, no weight sync, no checkpoint
(`checkpoints/` stayed empty), and the process exited.

## What it says about the remaining gap

The binding constraint for a *second* framework step is now the policy's solve rate on this task,
not the integration. smoke25 (same command, `n=16`) drew 1 PASSED of 16 and therefore certified;
smoke26 drew 0 of 16 and therefore did not. At the observed rate (~1/16) a 16-sample group has
roughly a 60 % chance of containing a solve, so a run that must reach a *second* step — which is what
proves §5.3 item 9 — is a coin flip per attempt, and needs either more samples per group or a task
the policy solves more often. Both are experiment-design choices, not code changes.

Run log: `/home/cxr/verl-closeout/verl-pi/smoke26.log`, sha256
`017214aaec3141532bd6b6f0a7bda92e8814a08579f20bf5bfe7fef90410141a`, 493 445 B.
