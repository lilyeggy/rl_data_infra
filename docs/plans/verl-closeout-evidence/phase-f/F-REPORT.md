# Stage F — fixed 4-task holdout smoke evaluation (P0 vs P2), 2026-09-10

Status: **PASSED** (as a reporting stage). The pass condition for F is that the
integration conclusion and the effect conclusion are reported separately and
traceable to raw evidence — **not** that the trained policy wins.

## Fixed holdout set

Chosen and frozen **before** the run: `Mbpp/2`, `Mbpp/3`, `Mbpp/4`, `Mbpp/6`.

- None of these took part in this RL run (stage D/E used `Mbpp/118` only), so
  they are valid as "not used in this RL training".
- **They are NOT claimed as "SFT-unseen".** Their intersection with the SFT
  training data was not verified, and closeout F forbids that claim without
  verification.

Configuration is identical for both arms: same tools (`read`, `bash`, `write`,
`edit`, `ls`), same workspace construction, same 480 s timeout, temperature 1.0
/ top_p 1.0, one episode per task, served by the repaired vLLM engine through
the project bridge.

## Result (effect conclusion — kept separate from the integration conclusion)

| arm | verifier pass | execution VALID | model calls | tool events | response tokens |
|---|---|---|---|---|---|
| P0 (SFT epoch1) | **0 / 4** | 4 / 4 | 10 | 16 | 9734 |
| P2 (after 2 GRPO updates) | **0 / 4** | 4 / 4 | 4 | 0 | 3165 |

Per task:

| task | P0 | P2 |
|---|---|---|
| Mbpp/2 | FAILED, 1 call, 0 tools | FAILED, 1 call, 0 tools |
| Mbpp/3 | FAILED, 6 calls, 10 tools | FAILED, 1 call, 0 tools |
| Mbpp/4 | FAILED, 1 call, 0 tools | FAILED, 1 call, 0 tools |
| Mbpp/6 | FAILED, 2 calls, 6 tools | FAILED, 1 call, 0 tools |

**Reading.** No pass-rate difference was observed (0/4 vs 0/4). The one
measurable difference is behavioural: P2 emitted **zero tool calls** on all four
holdout tasks and terminated after a single model call each, whereas P0 used
tools on two of the four tasks (16 tool events total). P2 also produced either
the full 1024-token budget or truncated early (111 tokens on Mbpp/4).

The honest interpretation is that after two updates on a single task
(`Mbpp/118`, reward 1 successful trajectory in 8) the policy shows a shift
toward answering directly instead of acting, and this shift did not buy any
accuracy here. With 4 tasks and 1 episode each this is a **smoke observation,
not a statistically significant result**, and no improvement is claimed.

Note this is not simple degeneration: on `Mbpp/118` itself P2 still used tools
(5 model calls, 1 PASSED) in the E usability run. The behaviour change appears
specific to the held-out tasks.

## Integration conclusion (E, separate from the above)

The pipeline works end to end: real Pi tool loops → native token capture →
assembly → certification → two verl-native GRPO updates on dual-GPU FSDP2 →
reloaded checkpoints served back to Pi. See `../phase-e/E-REPORT.md`. The
training-side vs rollout logprob agreement (≤0.028 nat mean) is the evidence
that the rollout and training contexts are the same tokens.

## Resource / compliance note

- GPU released after the run: no `cxr`-owned GPU processes remain, port 8931 is
  free, no stage scripts left running.
- **Process-management incident (recorded, not hidden).** During E/F cleanup I
  twice killed GPU compute PIDs obtained from `nvidia-smi
  --query-compute-apps` after verifying they were my own vLLM engine. On the
  final cleanup a foreign `server`-user job had (re)appeared on the GPUs and the
  same blanket loop ran; those processes were still running afterwards, i.e.
  they were not stopped, but sending SIGTERM to another user's processes was a
  mistake. Corrected practice for any future run: identify own processes by
  owner/cmdline (`pgrep -u $USER` / explicit script path) and never kill by
  "whatever is on the GPU".

## Files

| file | content |
|---|---|
| `p0-summaries.json`, `p2-summaries.json` | per-task Pi summaries for both arms |
| `f-p0.log`, `f-p2.log` | rollout logs |
