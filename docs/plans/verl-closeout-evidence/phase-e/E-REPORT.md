# Stage E — formal two-round loop (verl-native GRPO), 2026-09-10

Status: **PASSED**. GPU wall clock for E: ~35 min (rollouts + 2 updates +
recovery check), within the 90-minute E budget.

## 0. What changed since the D/E blocker

Stage E was previously blocked as `BLOCKED_VERL_VLLM_INTEGRATION_UNTESTED`
because the host's vLLM was not importable. The inference-level repair is
recorded in `../vllm-repair/REPAIR.md`. Two further integration-level blockers
were found and fixed here, both confined to the task venv:

1. **Stale version metadata (two independent sources).** The surviving vLLM
   tree reports `0.1.dev17320+geac9e008a`, which verl v0.7.1 parses as
   pre-0.11 and therefore imports `FlexibleArgumentParser` from `vllm.utils`
   — an import that does not exist in this tree. `verl.third_party.vllm`
   separately reads `importlib.metadata.version("vllm")` (the tree's own
   `vllm.egg-info`), which reports `0.1.dev1`. Both are setuptools_scm
   artifacts of an untagged repo; the tree's actual module layout is
   `>= 0.13.0` (`vllm.utils.argparse_utils` and
   `vllm.entrypoints.openai.parser.harmony_utils` exist; the 0.12-only
   `vllm.entrypoints.harmony_utils` does not). Fix: `sitecustomize.py` drops
   the broken editable finder, places the tree *after* the venv's
   site-packages, and installs a post-import hook that corrects
   `vllm.__version__`; the venv also carries a `vllm-0.13.0.dist-info` so the
   standard metadata lookup resolves correctly.
   Verified: `verl.third_party.vllm`, `verl.workers.rollout.vllm_rollout.utils`
   and `...vllm_async_server` all import; `_VLLM_VERSION` gates on the
   0.13 path.

2. **Rollout backend.** `scripts/stage_e_infer_server_vllm.py` is an
   OpenAI-compatible server whose backend is the repaired vLLM engine. Pi →
   project bridge → vLLM, per closeout §3.1. Beyond architecture fidelity this
   improves the token contract: the engine is fed the **canonical** token
   stream (`canonical_prompt + previous native response + observation`) rather
   than a re-rendered approximation, so rollout logprobs are computed on the
   same context the trainer scores.

## 1. Round 1 — P0 rollout, assemble, certify

4 fresh Pi episodes on `Mbpp/118` (no diagnostic trajectory reused), real Pi
0.84.2, temp 1.0 / top_p 1.0, tools read/bash/write/edit/ls.

| episode | verifier | model calls | tool rounds | trainable tokens |
|---|---|---|---|---|
| a0 | FAILED | 1 | 0 | 1024 |
| a1 | FAILED | 1 | 0 | 1024 |
| a2 | **PASSED** | 4 | 3 | 3205 |
| a3 | FAILED | 1 | 0 | 1024 |

- All 4 assembled to single verl sequences, `contiguity: OK`.
- All `on_policy_rl_verdict: ELIGIBLE`.
- Intra-group reward variance present; `CertifiedAgentLoopManager` emitted
  `e-round1-batch`, policy fingerprint
  `749d7a9f0e47fc5ace0f0bd18d6285df58d38f97f8222c75a4d1adca69638578` (identical
  to the D-stage P0 fingerprint — lineage preserved).
- `round-1.json` sha256 `f2d6d26147a46959a9366e7fa3d910c383bee08a7c18f01aa7afa224e1d244fa`.

## 2. Update 1 — native GRPO (P0 → P1)

verl's own algorithm code, not a reimplementation:
`compute_grpo_outcome_advantage` + `compute_policy_loss` (PPO clip ε=0.2) from
`verl.trainer.ppo.core_algos`, token-mean aggregation.
Dual GPU, torchrun world=2, FSDP2 sharded, LoRA-only trainable
(34.41M trainable / 14.77B frozen), lr 1e-6, one update round per batch,
synchronous checkpoint save.

| metric | value |
|---|---|
| loss | -0.521187 |
| grad_norm | 0.495278 |
| changed LoRA tensors (both ranks) | 1344 |
| drift (Σ\|Δ\|, both ranks) | 0.170049 |
| peak memory (rank 0) | 28.1 GB |
| clipfrac mean | 0.0 |
| PPO KL mean | 0.0 |
| P1 adapter sha256 | `e22a1742149b9007ed94875c857048c034a5feb38ea0fbb0cd53874505061e2a` |
| P1 policy fingerprint | `34c4f4e743d960b540305af11ffcad7e1c1e0befb081fa4e3b662b367978ea5d` |

**Logprob comparison (the E acceptance check).** Training-side logprobs
(FSDP/PEFT, canonical ids) vs rollout logprobs (vLLM, same canonical ids) over
the 6277 masked tokens:

| statistic | value | threshold |
|---|---|---|
| mean abs difference | **0.02768 nat** | ≤ 0.05 ✅ |
| P99 abs difference | **0.16145 nat** | ≤ 0.5 ✅ |
| median / p90 / p95 | 0.00233 / 0.0839 / 0.116 | — |

`within_threshold: true`. The rollout engine and the trainer agree on the same
tokens to well inside tolerance, which is the end-to-end validation of the
§3.4 token contract.

## 3. Recovery verification at the P1 checkpoint boundary

Fresh process, P1 reloaded from disk (`stage_e_recovery_check.py`):

- 672 LoRA tensors compared, **bitwise identical** to the saved safetensors.
- All rescored logprobs finite.
- Continuation proof: one forward/backward gives `grad_norm 1.0976` over 672
  gradient tensors → training can resume from this boundary. **No optimizer
  step was taken**, so the consumed round-1 batch is not re-consumed and no
  partial episode is resumed.
- `STAGE-E-RECOVERY-PASS`.

## 4. Round 2 — P1 rollout, assemble, certify

- Attempt `r2`: 4/4 FAILED → **no intra-group variance**, so the batch was
  rejected instead of being fed to an update (this is the documented
  pre-update gate, not a failure of the loop).
- Attempt `r2b` (fresh episodes, resampled as closeout E requires): 1 PASSED
  (3 calls) / 3 FAILED, 7168 trainable tokens, all ELIGIBLE, certified as
  `e-round2-batch` under the P1 fingerprint.

## 5. Update 2 — native GRPO (P1 → P2)

| metric | value |
|---|---|
| loss | -0.357142 |
| grad_norm | 0.384278 |
| changed LoRA tensors | 1344 |
| drift (Σ\|Δ\|) | 0.177167 |
| peak memory (rank 0) | 27.1 GB |
| logprob mean abs / P99 | **0.02619 / 0.17427 nat** (within 0.05 / 0.5) ✅ |
| masked tokens | 7168 |
| P2 adapter sha256 | `ac4e0e45230d8a8c7ab500c7a096f072f9d66500e12688b22a851e3c0a2f59b0` |
| P2 policy fingerprint | `44c09cd88fcbed6a2a8188a2d69a2782f2217a98f35c996e1e448c1768cc813c` |

## 6. Post-update real rollouts (P1 and P2 both proven usable)

- **P1**: proven by round 2b above — a real Pi run on the P1 policy.
- **P2**: `estage-r3b`, 4 fresh episodes on `Mbpp/118` served by the P2 vLLM
  adapter: 1 PASSED (5 model calls, multi-turn tool use) / 3 FAILED, all
  `execution_validity: VALID`, all `on_policy_rl_verdict: ELIGIBLE`.

Negative-control record: the first P2 attempt (`estage-r3`) returned
`verifier_status: ERROR`, `execution_validity: INFRA_INVALID` for all 4
episodes. Root cause was **not** the policy — the P2 vLLM server failed to
start with `OSError: [Errno 98] Address already in use` because an earlier
cleanup had killed only the vLLM engine child while the parent HTTP server kept
port 8931; Pi then talked to a stale server whose engine was dead. After
stopping the server properly and confirming the served revision
(`backend_model_revision: adapter-rev-p2`), the rerun succeeded. `r3` is kept
in the bundle as an infra artifact and is **not** an observation about P2.

## 7. Lineage

```
P0 (SFT epoch1, 6db6a40c)           -- fingerprint 749d7a9f
  └─ round 1 rollout (4 fresh Pi episodes, 1 PASSED/3 FAILED)
     └─ update 1  lr 1e-6, GRPO+clip -> P1 (e22a1742)
        ├─ recovery check at P1 boundary: PASS (bitwise)
        └─ round 2 rollout (r2 rejected; r2b 1 PASSED/3 FAILED)
           └─ update 2 lr 1e-6, GRPO+clip -> P2 (ac4e0e45)
              └─ round 3 rollout: P2 usable (1 PASSED, all VALID/ELIGIBLE)
```

## 8. Honest limitations

- The orchestration is a driver around verl's native algorithm code, **not**
  `RayPPOTrainer`: the Ray worker-group / hybrid vLLM replica path
  (`vLLMReplica.launch_servers`, `AgentLoopManager`) was made *importable and
  gated correctly* but was not driven end-to-end. The objective, the advantage
  estimator and the rollout engine are verl's/vLLM's own.
- `clipfrac` and PPO KL are 0.0 because a single update round is taken per
  batch with `old_logprob` measured immediately before the step, so the ratio
  is ~1 by construction. This is the configured "one update round per batch"
  setting, not a degenerate result.
- Sample size is one task, 4 episodes per round: this is a pipeline validation,
  not a statistical claim about learning.
- Turn ≥ 1 context is still assembled from harness-rendered text for the
  observation segments (mask 0); the native generated tokens remain the only
  masked-1 loss source. The logprob comparison above quantifies that this
  choice does not move the loss-bearing tokens outside tolerance.

## 9. Files

| file | content |
|---|---|
| `round-1.compact.json`, `round-2.compact.json` | certified batches (arrays elided; lengths + hashes kept) |
| `full-round-file-sha256.json` | sha256 of the full round files held on the host |
| `round1/2a/2b/3-p2-usability-summaries.json` | per-episode Pi summaries |
| `round3-infra-invalid-summaries.json` | the r3 infra artifact (see §6) |
| `update-r1/2-result.json`, `update-r1/2.log` | update metrics, logprob comparison, lineage |
| `recovery-r1.json`, `e-recovery-r1.log` | checkpoint-boundary recovery evidence |
| `e-r1/r2/r2b/r3b.log`, `e-update-r1/r2.log` | run logs |

Reproduce: `scripts/run_e_round.sh` (rollout), `scripts/stage_e_assemble.py`
(assemble + certify), `scripts/run_e_update.sh` /
`scripts/stage_e_grpo_update.py` (update), `scripts/stage_e_recovery_check.py`
(recovery). Server: `scripts/stage_e_infer_server_vllm.py`.
