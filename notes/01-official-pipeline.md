# Polar 官方链路与本项目边界

> 本文件只记录上游参考语义。项目权威定义见 `PROJECT_PLAN.md`。

## 1. Polar 本身负责什么

Polar 是面向真实 Agent Harness 的 rollout-as-a-service：Rollout Server/Gateway 负责调度任务、启动 Harness/runtime、代理模型请求、收集 token 级 trace、构建 trajectory，并执行或连接 evaluator。

Calculator、Count Stars 和 SWE-bench Verified 等 rollout 示例不需要 Megatron，也不包含模型参数更新。

## 2. 官方完整 Agentic RL 示例负责什么

`examples/swegym_slime_grpo` 把 Polar rollout 接到 Slime/Megatron：Polar 产生 Agent trajectory，Slime 组织 GRPO，Megatron 更新模型，SGLang 接收新权重。其官方硬件拓扑是训练参考，不是 Polar rollout 的运行前提，也不是本项目必须照搬的配置。

## 3. 本项目复现的官方范围

Day 1–3 只复现和审计：

```text
task
→ Polar Rollout Server
→ Polar Gateway
→ Harness + tools
→ SGLang model calls
→ trajectory
→ verifier/reward
→ raw artifacts
```

不把官方 8-GPU Slime/Megatron 拓扑作为复现目标。

## 4. Polar 在本项目中的身份

Polar 是第一个 `RolloutSourceAdapter` 的真实数据来源，也是生成 integration fixture 的参考系统。核心 pipeline：

- 不启动 Polar；
- 不 import Polar；
- 不依赖 Polar API；
- 不理解 Gateway 的控制逻辑；
- 只接收满足统一 contract 的 rollout records。

核心测试必须能够使用 JSONL/golden fixtures 独立运行。

## 5. 本项目在 Polar 之后做什么

```text
Polar raw result
→ PolarSourceAdapter
→ Canonical Rollout Batch
→ FailureClassifier
→ SignalFilter
→ PolicyConsistentGroupBuilder
→ TrainingReadyBatch / ResampleRequest
```

其他 Provider 只需实现新的 Source Adapter，即可复用相同 Processor。

## 6. 自定义 Agentic RL 验证

本项目使用自己的双 RTX PRO 6000 staged 配置：

1. 当前 policy 通过 Polar/SGLang 产生 rollout；
2. 数据模块处理并组装 batch；
3. 释放 rollout GPU 占用；
4. Slime/Megatron TP2 更新 Qwen3-4B；
5. 新 checkpoint 重新加载到 SGLang；
6. 新 policy 产生下一轮 rollout。

Slime 是第一个 Trainer Adapter，不是核心数据模块的唯一消费者。

## 7. 本项目的改动点

- producer-agnostic rollout contract 与 capability；
- 基础设施失败与真实任务失败分离；
- reward-variance/high-signal group 检查；
- 同 task、同 policy、固定大小的 GRPO group builder；
- 缺少有效样本时输出显式 `ResampleRequest`；
- Polar fixture、离线 fixture 与 Slime Sample 的可重复转换；
- 自定义两卡 on-policy 训练闭环和对照实验。
