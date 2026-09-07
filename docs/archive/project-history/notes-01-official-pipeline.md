# ARCHIVED：早期 Polar / Orchard 边界笔记

> 本笔记用于解释参考项目，不定义当前实施范围；权威范围见 [`PROJECT_SCOPE.md`](../PROJECT_SCOPE.md)。

## 1. Polar 提供什么

Polar 提供真实 Agent rollout 路径，包括 Harness/runtime、模型 API proxy、token trace、trajectory builder 和 verifier。它对本项目的价值是：

- 提供真实 model/tool/runtime/verifier artifact；
- 验证 token、status、reward 和 failure 的来源；
- 作为首个 capture/source integration；
- 证明契约不是凭空编造。

Polar 更关注从 Harness 调用中重建可训练 trajectory。本项目的新主线更关注统一执行观测、Harness 对比和回归决策。

## 2. Orchard 提供什么

Orchard 的核心抽象是让不同 Harness 使用统一、可编程的 Agent environment/sandbox 服务。它解决环境创建、命令执行、文件/patch 和生命周期等运行底座问题。

本项目不复制完整 Orchard 平台。我们借鉴的是：

- Harness 与 Environment 解耦；
- 统一 environment-side API；
- 多 Harness 可以在相同环境约束下运行；
- 环境产物可以复用于 rollout、evaluation 和 training。

本项目向上增加的数据价值是：

```text
Any Harness × Controlled Environment
→ Unified Execution Trace
→ Diagnose / Compare / Regression Gate
```

## 3. 三者的区别

| 系统 | 主要抽象 | 主要输出 |
|---|---|---|
| Orchard | Harness-agnostic environment service | 可执行 Sandbox/environment |
| Polar | Harness/model-call rollout capture | token-faithful trajectory/training data |
| 本项目 | Multi-Harness execution data plane | Episode、诊断、A/B 对比和 Gate |

## 4. 现有训练路线如何保留

旧链路：

```text
Polar → RolloutRecord → TrainingReadyBatch → Slime/Megatron
```

保留为可选扩展：

```text
AgentEpisode → TrainingViewExporter → RolloutRecord → optional trainer
```

因此已有 Polar adapter、token/mask/reward capability、checksum 和 policy gate 不删除，但 Slime/GRPO 不再是本周完成定义。

## 5. 当前 reference E2E

```text
same tasks/model/environment/verifier
          ├→ Harness v1
          └→ Harness v2
                 ↓
     model + sandbox + verifier capture
                 ↓
             AgentEpisode
                 ↓
 metrics + attribution + paired comparison
                 ↓
           Regression Gate + UI
```

一个有效演示必须能从 Gate 结论反查 comparison、diagnosis、event 和 artifact。
