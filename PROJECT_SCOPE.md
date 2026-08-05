# Rollout-Agnostic Agentic RL Data Pipeline：项目范围

## 一句话定义

构建一个与 Rollout Producer 解耦、与 Trainer 解耦的 Agentic RL 数据处理模块，把异构 rollout 记录转换为 **有效、具有训练信号、policy-consistent 的 GRPO 训练批次**。

Polar 是第一个 Reference Rollout Provider；Slime/Megatron 是第一个 Reference Trainer Consumer。两者都不是核心模块的运行时硬依赖。

## 项目要解决的问题

Rollout 框架能够产生 Agent 执行，但不同来源的字段、失败语义和分组方式不一致，不能直接安全地交给 Agentic RL Trainer：

- task 真实失败与容器、Harness、模型服务、verifier 等基础设施失败可能混在一起；
- 同一 GRPO group 中的样本可能来自不同 policy version；
- 过滤无效 rollout 后，group size 和 reward variance 可能失效；
- token、action/loss mask、old logprobs 等训练字段可能缺失或不一致；
- Trainer 往往只看到样本，不知道应该拒绝、补采还是等待更多 rollout。

本项目在 Rollout Producer 与 Trainer 之间提供稳定的数据边界：

```text
Rollout Producer
→ Source Adapter
→ Canonical Rollout Batch
→ Pluggable Processors
→ Training-Ready Batch / Resample Request
→ Trainer Adapter
```

## 核心模块拥有的能力

- `RolloutRecord`、`RolloutBatch`、`TrainingReadyBatch` 等稳定数据契约；
- Source Adapter 协议和 capability 声明；
- Polar Adapter 与离线 JSONL Adapter；
- `valid_success / valid_failure / invalid_infrastructure` 分类；
- reward-variance/high-signal group 筛选；
- 按 `task_id + group_id + policy_version` 构建 GRPO group；
- 样本不足时生成 `ResampleRequest`，但不直接控制上游 rollout；
- Slime Sample Adapter；
- 处理报告、拒绝原因、输入输出 checksum 与最小 lineage；
- 无需安装 Polar 或 Slime 即可执行的核心单元测试。

## Reference Integration 拥有的能力

### Polar Reference Provider

- Calculator rollout 参考复现；
- SWE-Gym/SWE-bench 小规模 Coding Agent rollout；
- Harness、tool call、verifier、token metadata 的真实 fixture；
- `PolarSourceAdapter` 集成测试。

### Slime/Megatron Reference Consumer

- 将 `TrainingReadyBatch` 转换为 Slime `Sample`；
- 使用自定义双 RTX PRO 6000 配置完成 GRPO 更新；
- checkpoint 保存、SGLang 重新加载和新 policy rollout；
- 验证数据确实可以参与 Agentic RL，而不是只生成离线 JSON。

## 外部 Harness 的边界

外部 Harness 继续负责 Agent loop、上下文、工具决策、工具副作用和运行生命周期。本项目不复制或合并 Harness 源码，只通过 rollout 结果或 Source Adapter 接收数据。

## 输入

```text
RolloutRecord(s)
SourceCapabilities
ProcessingConfig
PolicyContext
GroupRequirements
```

## 输出

```text
TrainingReadyBatch
ValidityDecision(s)
ProcessingReport
ResampleRequest(s)
RejectedRecord(s)
```

## 第一版只实现的三个 Processor

1. `FailureClassifier`：区分真实成功、真实失败和基础设施无效；
2. `SignalFilter`：检查 group reward variance 与可训练信号；
3. `PolicyConsistentGroupBuilder`：构建固定大小、同 policy 的 GRPO group，并输出补采请求。

## 明确非目标

- 不重新实现 Polar、Agent loop、GRPO/PPO 或 Megatron；
- 不要求使用 Polar 才能运行核心模块；
- 不要求使用 Slime 才能生成通用输出；
- 不做多 Session Graph 或信用分配；
- 不做通用生产级数据湖、数据库、Dashboard 或复杂规则引擎；
- 不做 Process Reward Model；
- 不实现异步 off-policy correction；
- 第一版不声称完整支持 VeRL、NeMo RL 等多个 Trainer；
- 不复现 Polar 官方 8-GPU 训练拓扑；
- 不以 SWE-Bench SOTA 或显著模型涨点作为唯一完成条件。

## 成功标准

1. Polar Calculator 和至少一条真实 Coding Agent rollout 被保存为可重复 fixture；
2. 核心 pipeline 在没有 Polar/Slime 的环境中通过全部单元测试；
3. Polar 与 JSONL 两个 Source Adapter 产生相同 canonical contract；
4. 三类 validity 状态和拒绝原因可稳定复现；
5. Group Builder 不混合 policy version，并能在缺样本时产生正确补采请求；
6. Slime Adapter 能确定性导出训练 Sample；
7. 自定义双卡配置至少完成 `rollout → process → GRPO update → reload → new rollout`；
8. baseline 与启用处理器的对照实验可重复；
9. 文档明确区分核心模块、Reference Provider 和 Reference Consumer；
10. 所有简历和演示结论都有真实 artifact 支撑。
