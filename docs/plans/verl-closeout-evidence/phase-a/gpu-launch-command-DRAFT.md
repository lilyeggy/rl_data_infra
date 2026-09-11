# Phase C GPU launch command — DRAFT ONLY, NOT EXECUTED (Phase A)

Status: draft. No GPU was touched. Execution requires: frozen P0 above,
explicit GPU UUIDs, a usage-window deadline, and a separate run ID.

Planned placement (from closeout §C, fixed initial settings):

- 2x RTX PRO 6000 Blackwell, colocated time-shared (verl hybrid mode)
- BF16, FSDP sharded over 2 GPUs, vLLM TP=1 (one replica per GPU)
- LoRA: load existing adapter rank8/alpha16 via `lora_adapter_path`
- micro batch 1/GPU, gradient checkpointing ON, actor/optimizer offload ON
- attention: native SDPA, remove-padding OFF
- vLLM gpu_memory_utilization 0.40 (single controlled retry at 0.30)
- no critic / reward model; KL coefficient 0; no FP8/QLoRA

Example config keys (verl v0.7.1, `HFModelConfig` + rollout):

```yaml
actor_rollout_ref:
  model:
    path: /home/cxr/agentic/models/qwen2.5-coder-14b-base
    lora_rank: 8
    lora_alpha: 16
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
    lora_adapter_path: /home/cxr/agentic/checkpoints/sft-runs/qwen14b-base-apps-clean-v2/epoch1
    enable_gradient_checkpointing: true
    use_remove_padding: false
  rollout:
    gpu_memory_utilization: 0.40
    tensor_model_parallel_size: 1
```

OOM policy: one controlled retry (shorter sequence budget + 0.30),
then STOP. Log the delta; no unbounded tuning.

Budget: this draft consumes 0 of the 4-hour GPU budget.
