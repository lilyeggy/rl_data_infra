# Day 7：Baseline 对照、故障注入、模型评估与项目包装

> 本阶段不再扩展核心范围，目标是用实验回答“数据处理模块是否有效、是否可插拔、是否真的参与了 Agentic RL”。

## 1. 阶段目标

1. 对比 minimal baseline 与 processed pipeline；
2. 使用可控故障验证基础设施失败不会污染 Trainer；
3. 运行短程 Agentic RL 并评估训练前后模型；
4. 证明核心模块不依赖 Polar/Slime；
5. 完成 runbook、架构、接口、实验、演示和简历材料。

## 2. 实验矩阵

### A. Offline Contract Experiment

对同一批 Polar/JSONL fixture 运行：

```text
baseline pipeline
processed pipeline
```

比较 invalid acceptance、group completion、reward variance、trainable tokens 和处理延迟。

### B. Fault Injection Experiment

注入：

- runtime prep failure；
- Harness crash；
- model backend error；
- verifier timeout；
- trace/token/mask 缺失；
- policy version 混合；
- group 少样本；
- reward 全同；
- duplicate trajectory。

每种故障必须有期望 decision/reason code，统计检出率和误拒绝。

### C. Live Polar Integration

用 Polar 产生一批真实 rollout，经 Pipeline 输出 ready groups/resample requests，确认核心决策与 offline fixture 一致。

### D. Agentic RL Training

使用相同模型、任务、token budget 和训练配置比较：

```text
baseline data path
processed data path
```

硬件时间不足时，baseline 至少完成 contract/short smoke，processed path 完成主要短程训练；差异必须如实说明。

### E. Held-Out Evaluation

在未参与训练的任务上比较 base checkpoint 与 trained checkpoint：

- success rate/mean reward；
- pass@k（样本量允许时）；
- rollout token 和工具调用数量；
- verifier/infra failure；
- 置信区间或至少样本数和逐任务结果。

不因短程结果不显著而隐藏负结果。

## 3. 核心指标

```text
invalid_infrastructure_to_trainer_rate
valid_group_completion_rate
no_signal_group_rate
resample_request_count
resample_success_rate
accepted_trainable_tokens
rollout_cost_per_ready_group
processing_latency
successful_optimizer_steps
checkpoint_reload_success
held_out_reward_before_after
```

## 4. 可插拔性证明

必须展示：

1. Core tests 在无 Polar/Slime 环境通过；
2. Polar Adapter 和 JSONL Adapter 产生兼容 contract；
3. Processor 可配置启用/关闭；
4. Pipeline 只输出 ResampleRequest，不直接调用 Polar；
5. Slime Adapter 位于 core 外部；
6. 新 Source/Trainer 可通过实现 Protocol 扩展，但未实现者不声称完整支持。

## 5. Clean Reproduction

Runbook 顺序：

```text
freeze environment
run Polar Calculator
run selected Coding rollout
save fixture
run source adapter
run baseline/processed pipeline
handle resample request
export Slime samples
run custom two-GPU update
reload checkpoint
run new-policy rollout
run held-out evaluation
generate report
```

所有命令、配置、commit、model revision 和 artifact 路径明确。不得依赖未记录的手工修改。

## 6. 文档集

### README

问题、边界、快速开始、Reference Provider/Consumer、真实结果和限制。

### Architecture

Source Adapter、contracts、processors、TrainingReadyBatch、ResampleRequest 与 Trainer Adapter。

### Polar Reproduction

只描述 rollout 参考复现、硬件调整和 fixture，不把官方 8-GPU training 写成项目依赖。

### Data Contract

字段、capability、required/optional、错误语义和示例。

### Processor Guide

三个 Processor 的规则、配置、reason code、输入输出与扩展方法。

### Training Runbook

双卡 staged loop、on-policy 边界、恢复和 checkpoint reload。

### Experiment Report

baseline、fault injection、训练和 held-out 结果，包括负结果与限制。

## 7. 演示流程

1. 展示 Polar 产生一条真实 Coding rollout；
2. 展示同一数据也可从 JSONL Adapter 输入；
3. 展示 infrastructure failure 被分类和拒绝；
4. 展示混合 policy group 被拒绝；
5. 展示 group 不足产生 ResampleRequest；
6. 展示 ready batch 转成 Slime Sample；
7. 展示真实 optimizer step 和 checkpoint checksum；
8. 展示新 checkpoint rollout；
9. 展示 baseline/processed 指标和 held-out 结果。

准备离线 artifact 演示，避免现场等待长 rollout/training。

## 8. 项目审计

- Core 不 import Polar/Slime；
- 外部 Harness 源码没有合并；
- secrets、私有 IP/路径和大模型权重未提交；
- 每个数字能反查 artifact；
- synthetic fault 明确标记；
- task failure 与 infra failure 没有混淆；
- policy version 没有跨 group 混用；
- 单步 smoke 没有表述成能力提升；
- 未实现的 VeRL/NeMo RL Adapter 没有表述成完整兼容。

## 9. 简历表述模板

```text
构建与 Rollout/Trainer 解耦的 Agentic RL 数据处理流水线，通过可插拔 Source
Adapter、基础设施失败分类、高信号筛选和 policy-consistent GRPO group builder，
将 Polar 产生的 Coding Agent rollout 转换为训练批次；在双 RTX PRO 6000 上以
自定义 Slime/Megatron 配置完成 on-policy 更新、checkpoint 回载和新策略 rollout。
```

第二条根据真实结果填写 invalid contamination、ready group rate、resample cost、训练 step 和 held-out reward；没有实际数字不填写。

## 10. 最终完成定义

- [ ] Polar rollout 参考复现和 fixture 完成；
- [ ] Core 可独立安装和测试；
- [ ] Polar/JSONL Source Adapter 通过 contract；
- [ ] 三个 Processor 可插拔；
- [ ] TrainingReadyBatch/ResampleRequest 可重复；
- [ ] Slime Adapter 与双卡训练闭环完成；
- [ ] checkpoint reload 和 new policy rollout 完成；
- [ ] baseline/fault injection/held-out 实验有报告；
- [ ] runbook 可从干净环境执行；
- [ ] 文档、演示、简历表述与真实证据一致。

## 11. 执行记录

```text
状态：NOT_STARTED
offline对照：
fault injection：
live Polar：
training cycles：
held-out结果：
可插拔性证明：
文档/演示：
最终限制：
```
