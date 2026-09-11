# Stage G — driving the Pi loop through verl's own trainer (status)

Date: 2026-09-10/11. **Not complete.** This records exactly how far the
`RayPPOTrainer` integration got, so a later session does not re-derive it.

The framework path now runs real Pi episodes through the framework's own vLLM
engine and gets them certified. It stops one step short of a framework-owned
GRPO update, for a reason that is architectural rather than mechanical; that is
described under "The remaining blocker".

## What was built (tests green, not yet committed)

Note on provenance: every file below is **untracked in git**. `git ls-files`
returns nothing for `src/integrations/verl/`, `src/capture/tool_calls.py`,
`scripts/run_verl_pi_train.sh` or `docs/plans/`. The whole closeout deliverable —
code, evidence and `acceptance.*` — lives only in the working tree, so it must
be committed before this work can be handed off or referenced.

| file | role |
|---|---|
| `src/integrations/verl/pi_loop.py` | `PiAgentLoop(AgentLoopBase)` — runs the real Pi harness through our existing orchestrator and returns the certified episode as `AgentLoopOutput` |
| `src/integrations/verl/verl_manager.py` | `PiServerManager` (publishes the engine HTTP address), `PiAgentLoopWorker`, `CertifiedVerlAgentLoopManager` (batch gate + resample on zero variance) |
| `src/integrations/verl/admission.py` | added fail-closed `AdmittedVerlSequence.from_dict` so the gate can certify from on-disk evidence |
| `src/capture/tool_calls.py` | recovery of tool calls from generated text (port of the stage D/E extractor) |
| `src/capture/model_proxy_http.py` | asks vLLM for native ids/logprobs, reads them from a stock OpenAI response, and recovers tool calls the engine's parser missed |
| `src/orchestration/pi_host_execution.py` | optional `max_tokens_per_generation` and `backend_model_revision` on the spec; the generated Pi config honours the token cap |
| `scripts/run_verl_pi_train.sh` | one GRPO round through `verl.trainer.main_ppo` |
| `scripts/build_verl_pi_dataset.py`, `scripts/write_round_policy.py` | prompt set and per-round `PolicyFingerprint` |
| `scripts/venv_sitecustomize.py` | venv-local post-import hooks (vLLM version metadata, trl shim) |
| `tests/integrations/test_verl_pi_loop.py`, `tests/capture/test_tool_calls.py`, `tests/orchestration/test_pi_host_execution.py` | contract tests, launcher regression guards, proxy recovery tests |

Local suite: **262 passed, 9 skipped, 28 subtests**. On the host: 26 passed for
the three touched test modules.

Verified by construction: `PiAgentLoop` is a subclass of the real pinned
`AgentLoopBase` with no abstract methods left; the manager subclasses the real
`AgentLoopManager`; the worker subclasses `AgentLoopWorker`; `PiServerManager`
subclasses `AsyncLLMServerManager`. The framework path reproduced the **exact
P0 fingerprint `749d7a9f…`** that the earlier driver path produced.

## How far the run gets now

`verlpi-smoke16` and `verlpi-smoke17` reach, in order:

1. Hydra composes and validates the full override set.
2. Ray starts (capped object store) and `TaskRunner` runs.
3. Both FSDP actor workers load the 14B base in bfloat16, apply LoRA, load P0.
4. The vLLM engine starts and serves the model over HTTP.
5. The pre-rollout weight sync completes and the LoRA lands in the engine.
6. **Real Pi episodes run against the framework's engine.** Each episode
   captures `prompt_token_ids`, `response_token_ids` and per-token
   `response_logprobs` from the engine's own response, and the episode verdict
   is `execution_validity=VALID`, `on_policy_rl_verdict=ELIGIBLE`.
7. The verifier runs and produces real PASSED/FAILED verdicts — `smoke16`
   produced a batch with rewards FAILED/FAILED/FAILED/PASSED.
8. The batch gate certifies, resamples, or refuses.

What has **not** happened: a certified batch reaching the advantage/update step.
So no framework weight sync after an update, and no checkpoint.

## Blockers found and what fixed them

| # | blocker | resolution |
|---|---|---|
| 1 | `torchdata` missing (imported by `ray_trainer`) | installed into the task venv |
| 2 | `agent_loop_manager_class` not in the YAML struct | needs `+` |
| 3 | `ppo_mini_batch_size` is counted in *prompts*, not samples | set to 1; framework multiplies by `rollout.n` |
| 4 | comment placed after a trailing backslash swallowed the whole override list | moved comments out; **regression guard added** (this bit twice) |
| 5 | actor wanted FlashAttention2 (not installed) | `model.override_config.attn_implementation=sdpa` |
| 6 | trl 1.4 deleted `AutoModelForCausalLMWithValueHead`, which verl imports whenever trl is importable | venv post-import shim; the placeholder raises if actually used |
| 7 | `pi_certification` inside `rollout.agent` was rejected by the typed `AgentLoopConfig` | moved to a top-level config key |
| 8 | vLLM's vendored `custom_all_reduce` kernel faults on SM 12.0 (`custom_all_reduce.cuh:455 'invalid argument'`) and the platform still reports it as supported | `VLLM_BATCH_INVARIANT=1` forces it off |
| 9 | critic config instantiated with a placeholder model path | `critic.model.path` set |
| 10 | FSDP1 held **81.5 GiB in a single rank** and OOM'd on the GPU | switch to FSDP2 + explicit `wrap_policy.transformer_layer_cls_to_wrap` |
| 11 | Ray's node-memory OOM killer killed 8–12 workers **at startup**, with no rollout ever run | `FSDPEngineConfig.model_dtype` defaults to `"fp32"` and `fsdp_workers.py:387` feeds it straight to `from_pretrained`. Measured on this host: **55.8 GiB anonymous per rank in fp32 vs 0.8 GiB in bfloat16** (bf16 weights stay reclaimable mmap views). Two fp32 ranks plus the foreign job crossed Ray's 95 % threshold. Set `actor.fsdp_config.model_dtype=bfloat16`, which is also what the stage E driver used |
| 12 | OOM again, now inside `actor_rollout_update_weights`, ramping 19.7 → 132.5 GiB per rank | the *pre-rollout* sync (`ray_trainer.py:1252`) gathers the whole base to CPU because `fsdp_workers.py:742` sets `base_sync_done = ("dummy" not in load_format)` and the shipped default is `dummy`; `sleep_level=2` then gathers it a second time. Fixed with `rollout.load_format=safetensors` + `rollout.layered_summon=true` (`fsdp_utils.py:591` handles the fsdp2 prefix layout and empties the cache per layer; it also pins `sleep_level` to 1, where vLLM offloads weights instead of discarding them) |
| 13 | `TypeError: 'coroutine' object is not subscriptable` in `_validate` | my `generate_sequences` override dropped the base method's `@auto_await` bridge. The trainer is synchronous and calls it without awaiting (`ray_trainer.py:546`, `:1321`). **Regression guard added** |
| 14 | `FileExistsError` on the episode directory, reported against an arbitrary sample | `_execute_episode` created `output_dir/workspace` *before* calling the orchestrator, and `parents=True` created the episode root as a side effect; `pi_host_execution.py:113` then fails closed on it. The workspace now lives outside the episode tree — and outside the attempt directory, which the batch gate enumerates |
| 15 | the same episode id twice in one dispatch | `generate_sequences` is not called once per run: validation runs first and then every step, and padding a one-row batch to the worker count repeats that row. Episodes are now keyed by session id + generation (`generation_root`), and the loop reads the root stamped on the batch. **Regression guard added** |
| 16 | validation re-ran the same tasks with `do_sample=false` | `trainer.val_before_train=false`: it contradicts the frozen temperature/top_p fingerprint and costs a second set of episodes. Evaluation is the separate F holdout run |
| 17 | vLLM 400 `The "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set` | `rollout.engine_kwargs.vllm.enable_auto_tool_choice=true` + `tool_call_parser=hermes` (`vllm_async_server.py:215,326` splats these into the engine args; this build registers `hermes` but no `qwen25`) |
| 18 | vLLM 404 `The model 'pi-rollout' does not exist` | the engine serves the model under its own id; `pi_loop._served_model_id` asks `/v1/models` instead of assuming |
| 19 | generation was truncated at the engine's context | Pi's own config asks for 8192 tokens and the loop's `max_tokens_per_generation` was declared but never applied; it is now threaded into the spec and Pi's generated config |
| 20 | `tool_calls: []` on every response, so Pi executed nothing and every episode died on its first turn | stock vLLM tool parsers match `<tool_call>` tags only, and this policy emits bare JSON — which is why the stage D/E servers did their own extraction. Added the same recovery to the proxy. Two bugs in my port: the tagged form was counted twice (tag regex plus brace scan), and the `dict`-only checks rejected the proxy's frozen `mappingproxy` tools; the second one silently disabled recovery until an end-to-end proxy test caught it |
| 21 | every episode rejected as lacking a policy identity | stock vLLM responses carry no revision, so `backend_model_revision` was null and the `POLICY_VERSION` capability failed. The loop now passes the engine's served model id |

## The remaining blocker

The assembler requires **prompt_{i+1} = prompt_i + generation_i + observation**
(`src/integrations/verl/bridge.py`), which is what makes the assembled sequence
exactly the context the model saw. Pi does not satisfy it: instead of echoing
the model's raw generation into the next turn, it re-renders the assistant turn
from its structured tool calls. Measured on a 3-call episode
(`verlpi-smoke16-Mbpp-118-a0s3`):

```
call 0: prompt 1693, response 1024
call 1: prompt 1816, expected 2717, common prefix 1693   <- diverges exactly where the generation should be
```

The prefix is preserved; the divergence starts at the first generated token. So
the harness discards the generation — including the ~900 tokens of base-model
rambling that follow the tool call — and substitutes its own rendering.

Consequences, all of them load-bearing:

* **Multi-call episodes are refused**, correctly: assembling them would train on
  a context the policy never saw. The gate reports
  `records[1] rewrote generation history; episode is not trainable`.
* **Only single-call episodes certify.** That is also the only shape stage E
  ever certified — its episodes made exactly one model call each, carrying
  several tool calls in that one response. Multi-turn was never exercised
  before, so this is a pre-existing property of the data plane that the
  framework path is the first to surface.
* **Single-call episodes do not reliably solve the task**, so a four-episode
  GRPO group tends to have no reward variance, and the gate then refuses the
  batch after its three resample attempts. `smoke16` is the closest so far: it
  did produce variance (FAILED/FAILED/FAILED/PASSED), but three of the four
  episodes made three calls each.

This is a contract decision, not a bug to fix quietly. Either

* the assembler is relaxed to stage E's looser construction (keep the prompt
  prefix, then the generation, then treat the harness's re-rendered turn as
  observation) — which is what the project's already-accepted E/F evidence
  uses, but which trains on a context the policy did not literally see; or
* the strict contract stands, and the certified path is documented as
  single-call only.

Both are defensible. Changing it unilaterally would alter what the project
claims about its own training data, so it is recorded here rather than decided
in this session.

## Why I stopped

Each attempt costs ~5–7 minutes of GPU bring-up plus the episodes, and the last
three failures are the same architectural check rather than new information.
The remaining work needs the contract decision above; further re-runs would
spend GPU time on a settled question.

## Cleanup

No processes of this run remain: `pgrep` for `main_ppo`/`raylet`/`gcs_server`/
`vLLMHttpServer`/`EngineCore`/`memwatch.py` is empty, Ray ports 8265/6379 are
free, GPU memory is back to the foreign job only (~5 GB per card), and host RAM
is at its normal ~217 GB available. The foreign `server`-user job was never
touched.

## What this does and does not change

- §5.3 items 2 ("training actually executed by verl") and 9 ("new weights
  framework-synced then used by Pi") remain **not satisfied**: no certified
  batch has reached the framework's advantage/update step, so
  `checkpoint_manager.update_weights` has never run *after* an update.
- The project claim must stay as it is: do not write "已接入 verl trainer" or
  "完整 Agentic RL 闭环".
- The Stage E driver path and its evidence are untouched and remain the
  documented, passing result for the two-round cycle.
- New and independently useful: the framework path proves the rollout half
  works — real episodes, native token ids and logprobs from the framework's own
  engine, real verifier verdicts, and a batch gate that refuses both
  zero-variance and untrainable batches.
