# Day 6：Slime Adapter 与自定义双卡 Agentic RL 闭环

> 本阶段证明核心输出能真正训练模型。我们使用自定义双卡 staged 配置，不复现 Polar 官方 8-GPU 训练拓扑。

## 1. 阶段目标

1. 实现 `SlimeTrainerAdapter`；
2. 将 TrainingReadyBatch 确定性转换为 Slime Sample；
3. 完成 rollout/process/train/reload 的同步 policy iteration；
4. 至少产生 `policy_v0 → policy_v1 → new rollout`；
5. 逐步运行多个真实 GRPO update；
6. 保存 batch、optimizer、checkpoint 和 policy lineage。

## 2. Slime Adapter Contract

至少映射：

```text
tokens
response_length
loss_mask
reward
status
rollout_id
metadata.trajectory_id
metadata.task_id/group_id
metadata.policy_version
metadata.processing versions
```

如果目标 Slime loss/config 需要 old logprobs，则它必须成为 required capability；Adapter 不得补零或重新推断。

转换必须记录 adapter version、input/output checksum，并提供 golden Sample fixture。

## 3. 自定义两卡配置

第一轮保守配置：

```text
Model: Qwen/Qwen3-4B-Instruct-2507
Precision: BF16
Trainer TP: 2
Context: 4096
Max output: 512
Group size: 2
Prompt batch: 2–4
Updates per policy batch: 1 or explicitly bounded
Execution: synchronous staged
```

稳定后再尝试 context 8192、output 1024、group size 4。训练不切 FP8；rollout 量化只作为后续可选实验。

## 4. Policy Iteration

### Phase A：Rollout

1. 启动 SGLang 并加载 `policy_vK`；
2. Polar 对训练候选任务采样；
3. 所有 raw records 写入 `policy_vK`；
4. 如果 GroupBuilder 产生 ResampleRequest，在 policy 更新前完成补采。

### Phase B：Process

1. Source Adapter 转换；
2. 三个 Processor 运行；
3. 冻结 TrainingReadyBatch manifest；
4. Slime Adapter 生成 Sample；
5. 停止/释放 SGLang 占用。

### Phase C：Train

1. 两卡启动 Megatron TP2；
2. 载入 `policy_vK`；
3. 执行真实 forward/backward/optimizer；
4. 保存 loss、KL/entropy/clip、gradient norm、trainable tokens、step time 和 peak memory；
5. 保存 `policy_vK+1` checkpoint/checksum。

### Phase D：Reload

1. 停止 Trainer 或释放显存；
2. SGLang 加载新 checkpoint；
3. 验证 model/checkpoint identity；
4. 产生至少一条 `policy_vK+1` rollout；
5. 验证新记录没有误标为旧 policy。

## 5. On-Policy 约束

- 每个 TrainingReadyBatch 只包含一个明确的 policy version；
- policy 更新后，旧 batch 不用于无限多次训练；
- 补采必须发生在该 policy 被更新前；
- 如果恢复 checkpoint，controller 必须恢复对应 policy iteration 状态；
- 第一版不实现重要性采样或 off-policy correction。

## 6. 训练任务

使用 Day 3 pilot 中基础模型存在成功/失败差异、verifier 稳定、时长可控的任务。训练集与 held-out eval task 分开保存。

## 7. 训练层级验收

### Contract Smoke

一个 TrainingReadyBatch 完成一次 optimizer step。

### Closed-Loop Smoke

完成 `v0 rollout → process → update → v1 reload → v1 rollout`。

### Short Run

目标 10–20 个连续 policy/update cycle 或根据任务成本明确记录实际数量。不能把单个 step 表述为能力提升。

## 8. 失败恢复

- trainer crash 后从 checkpoint/batch manifest 恢复；
- reload 失败不分配新 policy version；
- ResampleRequest 不重复无限提交；
- 同一 batch 不重复 optimizer step，除非 run manifest 明确允许；
- 磁盘 weight reload 作为首选可靠路径，NCCL sync 可后续优化。

## 9. 产物

```text
src/trainers/slime.py
examples/polar_slime_staged_loop/
configs/training/qwen3-4b-2xpro6000.yaml
tests/trainers/test_slime_contract.py
artifacts/day-06/batches/
artifacts/day-06/checkpoints/
artifacts/day-06/policy-lineage.jsonl
docs/custom-two-gpu-training.md
```

## 10. 验收门

- [ ] TrainingReadyBatch 确定性转换为 Slime Sample；
- [ ] Megatron TP2 完成真实 optimizer step；
- [ ] checkpoint checksum 发生预期变化；
- [ ] 新 checkpoint 可被 SGLang 加载；
- [ ] new policy rollout 明确标记新 version；
- [ ] rejected/invalid records 没有进入 batch；
- [ ] 至少一个完整 policy iteration 可重复；
- [ ] 训练配置完全独立于官方 8-GPU 配置。

## 11. 执行记录

```text
状态：NOT_STARTED
Slime Adapter版本：
训练配置：
policy versions：
optimizer steps：
checkpoint/reload：
new rollout：
显存/step time：
未解决问题：
```
