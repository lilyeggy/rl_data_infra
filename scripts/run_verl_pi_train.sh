#!/usr/bin/env bash
# Stage G: run one GRPO round through verl's own trainer (RayPPOTrainer).
#
# Everything framework-side is stock verl: Ray placement groups, the
# AsyncActorRolloutRefWorker (FSDP2 + LoRA), the hybrid vLLM replica, the
# advantage/update math, the checkpoint and the weight sync. Our part is only
# the agent loop (real Pi + our certified data plane) and the batch gate.
#
# Usage:
#   ./run_verl_pi_train.sh <round-name> <adapter-dir> <policy-generation> <dataset.jsonl>
#
# Environment (deployment paths; override as needed):
#   VERL_PI_ROOT   run root (default /home/cxr/verl-closeout/verl-pi)
#   VERL_PYTHON    interpreter (default the task venv)
#   BASE_MODEL     HF base model
#   PI_BINARY      real Pi CLI
#   REPO           deployed repo root
set -euo pipefail

ROUND=${1:?round name, e.g. verlpi-r1}
ADAPTER=${2:?adapter dir (P0/P1/...)}
GENERATION=${3:?policy generation label, e.g. P0}
DATASET=${4:?dataset jsonl built by build_verl_pi_dataset.py}
shift 4 || true

VERL_PI_ROOT=${VERL_PI_ROOT:-/home/cxr/verl-closeout/verl-pi}
VERL_PYTHON=${VERL_PYTHON:-/home/cxr/verl-closeout/venv/bin/python}
BASE_MODEL=${BASE_MODEL:-/home/cxr/agentic/models/qwen2.5-coder-14b-base}
PI_BINARY=${PI_BINARY:-/home/cxr/verl-closeout/pi-global/bin/pi}
REPO=${REPO:-/home/cxr/agentic/code}
VERIFIER=${VERIFIER:-/home/cxr/verl-closeout/smoke/verify_mbpp_src.py}
TOOL_SCHEMA_JSON=${TOOL_SCHEMA_JSON:?real Pi CPU probe tool-schema.json is required}

RUN_DIR=$VERL_PI_ROOT/$ROUND
mkdir -p "$RUN_DIR"
AGENT_LOOP_YAML=$RUN_DIR/agent_loop.yaml
ROUND_POLICY=$RUN_DIR/round-policy.json

# vLLM needs ninja on PATH and a CUDA_HOME; Ray must not adopt a foreign cluster.
export PATH=/home/cxr/miniconda3/envs/vllm/bin:$PATH
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export RAY_ADDRESS=local
export PYTHONPATH=$REPO:$VERL_PI_ROOT
export AGENT_VERIFIER_PYTHON=$VERL_PYTHON
# RTX PRO 6000 is SM 12.x: vLLM's FlashInfer path refuses it, so pin the
# engine to the FlashAttention backend that works on this deployment.
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
# This vLLM build ships a custom all-reduce CUDA kernel that faults on
# SM 12.0 (custom_all_reduce.cuh:455, "invalid argument") and the platform
# still reports custom all-reduce as supported, so vLLM does not disable it
# itself. Batch invariance is the supported switch that forces it off; as a
# side effect the kernels become deterministic, which suits our
# train-vs-rollout logprob comparison.
export VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-1}
# A 14B actor plus a colocated vLLM engine on one card needs the allocator
# to hand back freed blocks instead of fragmenting.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

cp "$VERIFIER" "$RUN_DIR/verify_mbpp_src.py"

# 1. Persist the round policy so the per-sample loop and the batch gate bind
#    the same behaviour policy across processes.
"$VERL_PYTHON" "$REPO/scripts/write_round_policy.py" \
  --adapter "$ADAPTER" --base-model "$BASE_MODEL" --tool-schema-json "$TOOL_SCHEMA_JSON" \
  --policy-generation "$GENERATION" --output "$ROUND_POLICY"

# 2. Generate the agent-loop registry entry the framework loads.
"$VERL_PYTHON" - "$AGENT_LOOP_YAML" "$ROUND_POLICY" \
  "$PI_BINARY" "$RUN_DIR" "$ROUND" <<'PY'
import json, sys
from pathlib import Path

import yaml

loop_path, policy_path, pi_binary, run_dir, round_name = sys.argv[1:6]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.training.policy_fingerprint import PolicyFingerprint  # noqa: E402

fingerprint = PolicyFingerprint.from_dict(json.loads(Path(policy_path).read_text()))
entry = {
    "name": "pi_agent",
    "_target_": "src.integrations.verl.pi_loop.PiAgentLoop",
    "pi_binary": pi_binary,
    "episode_root": f"{run_dir}/episodes",
    "run_id": round_name,
    "policy_fingerprint": fingerprint.checksum(),
    "tool_schema_checksum": fingerprint.tool_schema_checksum,
    "verifier_script": f"{run_dir}/verify_mbpp_src.py",
    "tools": ["read", "bash", "write", "edit", "ls"],
    "episode_timeout_seconds": 480.0,
    # Deliberately the same number as rollout.response_length below: this is no
    # longer meant to be the binding limit. The transport clamps each request to
    # min(that, context.remaining), so the episode's real budget is what bounds a
    # generation. At 1024 every generation of the smoke16 run was cut off
    # mid-action (12/12), never emitted <|im_end|>, and was therefore rejected by
    # the native-EOS gate before any tool could run.
    "max_tokens_per_generation": 3584,
    "max_model_requests_per_episode": 8,
    "sampling_temperature": 1.0,
    "sampling_top_p": 1.0,
}
Path(loop_path).write_text(yaml.safe_dump([entry], sort_keys=False))
print(json.dumps({"agent_loop_yaml": str(loop_path),
                  "policy_fingerprint": fingerprint.checksum()}, indent=2))
PY

cd "$REPO"
# Deployment-specific settings, all measured on this host:
#
# * ppo_mini_batch_size is counted in PROMPTS and must be <= train_batch_size;
#   the framework multiplies it by rollout.n for the real sample count.
# * FSDP2 is what stage C measured at ~22GB/rank for this model; FSDP1 held
#   ~81GB in a single rank and OOM'd during the weight sync.
# * The FSDP2 wrap policy must be explicit: PeftModel._no_split_modules is a
#   set, which apply_fsdp2 cannot subscript.
# * param_offload stays false: rollout_mode() gathers the whole model onto the
#   GPU when params are offloaded, whereas FSDP2 already keeps a 1/2 shard
#   resident and the LoRA-only sync needs just the adapter tensors.
# * This vLLM build's custom all-reduce CUDA kernel faults on SM 12.0
#   (custom_all_reduce.cuh:455, 'invalid argument') and the platform still
#   reports it as supported, so vLLM does not disable it itself. Batch
#   invariance is the supported switch that forces it off.
# * Ray's object store is capped: the default sizing plus two FSDP ranks and a
#   colocated vLLM tripped Ray's node-memory OOM killer.
# * model_dtype must be set explicitly. FSDPEngineConfig defaults it to "fp32"
#   and fsdp_workers.py:387 uses it as the from_pretrained dtype, so that
#   default materialises the whole 14B checkpoint as anonymous host memory in
#   every rank. Measured on this host: 55.8 GiB/rank anonymous in fp32 versus
#   0.8 GiB in bfloat16, where the weights stay reclaimable mmap views. Two
#   fp32 ranks plus the foreign job put the node past Ray's 95% kill threshold.
#   bfloat16 also matches what the stage E driver loaded, so both paths train
#   at the same precision.
# * GRPO needs no critic, but the critic config is still instantiated and its
#   default model path is a placeholder that fails to resolve.
# * val_before_train is off. ray_trainer.py:1259 otherwise rolls the same tasks
#   out once more before training, with do_sample=False from val_kwargs, which
#   contradicts the temperature=1.0/top_p=1.0 sampling the PolicyFingerprint
#   freezes, and it spends a full extra set of Pi episodes on top. Evaluation is
#   the separate F holdout run, not this.
# * The engine must accept the tool declarations Pi sends. Pi sets
#   tool_choice="auto", and vLLM answers 400 'The "auto" tool choice requires
#   --enable-auto-tool-choice and --tool-call-parser to be set' without both.
#   "hermes" is the parser for the Qwen2.5 chat template's <tool_call> JSON;
#   this vLLM build registers hermes but no qwen25. rollout.engine_kwargs.vllm
#   is splatted straight into the engine args (vllm_async_server.py:215,326).
# * load_format and layered_summon together decide how the weight sync moves
#   the policy into the rollout engine, and the defaults are ruinous here.
#   fsdp_workers.py:742 sets base_sync_done = ("dummy" not in load_format), and
#   the shipped default load_format is "dummy", so the first sync (which
#   ray_trainer.py:1252 runs *before* any rollout) takes the base_sync_done=False
#   branch of collect_lora_params: FSDP.summon_full_params over the whole module
#   plus a full state_dict() of the 14B base copied to CPU -- per rank. Because
#   we render a real checkpoint, letting vLLM load the base itself
#   (load_format=safetensors) makes that branch unnecessary and reduces the
#   sync to the LoRA delta. layered_summon then keeps the collection
#   layer-by-layer (fsdp_utils.py:591 handles the fsdp2 prefix layout and calls
#   empty_cache between layers) instead of all-gathering the model a second
#   time, and it also pins sleep_level to 1: at vLLM's level 1 the weights are
#   offloaded to CPU and restored on wake, whereas level 2 discards them, which
#   is what makes fsdp_workers.py:786 re-collect the full base on every resume.
#   Measured before this: each rank climbed 19.7 -> 132.5 GiB and the node hit
#   Ray's kill threshold again, with no rollout ever having run.
#
# Never interleave comments inside the backslash-continued command below: a
# comment after a trailing backslash is swallowed into the same logical line
# and silently drops every override after it. This has bitten twice;
# tests/integrations/test_verl_pi_loop.py guards it.
exec "$VERL_PYTHON" -m verl.trainer.main_ppo \
  trainer.nnodes=1 \
  +ray_kwargs.ray_init.object_store_memory=2147483648 \
  trainer.n_gpus_per_node=2 \
  trainer.logger=[console] \
  trainer.project_name=agentic-rl \
  trainer.experiment_name="$ROUND" \
  trainer.total_epochs=1 \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=false \
  trainer.default_local_dir="$RUN_DIR/checkpoints" \
  actor_rollout_ref.model.path="$BASE_MODEL" \
  actor_rollout_ref.model.lora_rank=8 \
  actor_rollout_ref.model.lora_alpha=16 \
  actor_rollout_ref.model.lora_adapter_path="$ADAPTER" \
  "actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=false \
  "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen2DecoderLayer] \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.n_gpus_per_node=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
  actor_rollout_ref.rollout.prompt_length=2048 \
  actor_rollout_ref.rollout.response_length=3584 \
  actor_rollout_ref.rollout.max_model_len=6400 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=true \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=true \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  actor_rollout_ref.rollout.agent.default_agent_loop=pi_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_YAML" \
  "+actor_rollout_ref.rollout.agent.agent_loop_manager_class=src.integrations.verl.verl_manager.CertifiedVerlAgentLoopManager" \
  +pi_certification.run_id="$ROUND" \
  +pi_certification.episode_root="$RUN_DIR/episodes" \
  +pi_certification.minimum_group_size=4 \
  +pi_certification.max_attempts=1 \
  +pi_certification.policy_fingerprint_json="$ROUND_POLICY" \
  algorithm.adv_estimator=grpo \
  critic.model.path="$BASE_MODEL" \
  critic.model.tokenizer_path="$BASE_MODEL" \
  data.train_files="$DATASET" \
  data.val_files="$DATASET" \
  data.train_batch_size=1 \
  data.max_prompt_length=2048 \
  data.max_response_length=3584 \
  "$@"
