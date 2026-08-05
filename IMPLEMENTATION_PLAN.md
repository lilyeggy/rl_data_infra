# Rollout-Agnostic Agentic RL Data Pipeline：Day 1–Day 7 实施索引

> `Day 1–Day 7` 表示七个顺序阶段。当前优先完成 Polar rollout 参考复现，再开发独立的数据模块，最后使用自定义双卡配置进行真实 Agentic RL 验证。

## 最终目标

```text
Any Rollout Producer
        ↓
Source Adapter
        ↓
Canonical Rollout Batch
        ↓
Failure Classification + Signal Filtering + Group Building
        ↓
Training-Ready Batch / Resample Request
        ↓
Any Trainer Adapter
```

本项目的 Reference E2E 路径为：

```text
External Coding Harness
→ Polar rollout
→ PolarSourceAdapter
→ pluggable data processors
→ SlimeSampleAdapter
→ custom two-GPU GRPO
→ new checkpoint
→ SGLang reload
→ next Polar rollout
```

## 固定技术决策

| 层 | 第一版选择 | 核心是否绑定 |
|---|---|---|
| Rollout Provider | Polar + 离线 JSONL fixture | 否 |
| Policy serving | SGLang | 否 |
| Coding task/runtime | SWE-Gym/SWE-bench 小规模任务 | 否 |
| Core contract | `RolloutRecord` / `TrainingReadyBatch` | 是 |
| Core processors | FailureClassifier / SignalFilter / GroupBuilder | 是 |
| Reference Trainer | Slime | 否 |
| Training backend | Megatron-LM TP2 | 否 |
| Algorithm | GRPO | 第一版验证目标 |
| Policy model | `Qwen/Qwen3-4B-Instruct-2507` | 第一版固定 |
| GPU | 2 × RTX PRO 6000 96GB | staged shared execution |

## 七个实施阶段

| 阶段 | 核心交付 | 详细文档 |
|---|---|---|
| Day 1 | 环境、版本和双卡容量基线；确认旧检查结果可复用 | [Day 1](plans/day-01-environment.md) |
| Day 2 | Polar Calculator rollout 参考复现与 raw fixture | [Day 2](plans/day-02-polar-smoke.md) |
| Day 3 | Polar Coding/SWE rollout、verifier 与真实 fixture | [Day 3](plans/day-03-swegym-rollout.md) |
| Day 4 | Producer-agnostic 数据契约、capability 与 Source Adapter | [Day 4](plans/day-04-data-contract.md) |
| Day 5 | 三个可插拔 Processor 与 GRPO group/resample 语义 | [Day 5](plans/day-05-processors.md) |
| Day 6 | Slime Adapter 与自定义双卡 on-policy GRPO 闭环 | [Day 6](plans/day-06-custom-grpo.md) |
| Day 7 | Baseline 对照、故障注入、模型评估、文档和演示 | [Day 7](plans/day-07-packaging.md) |

## 两个阶段边界

### 阶段 A：先实现并理解 Polar（Day 1–3）

- 只复现 rollout、Harness、tool、verifier 与数据产物；
- 不复现官方 8-GPU Agentic RL 训练拓扑；
- 不在 Polar 内实现本项目核心逻辑；
- 输出 raw artifacts、字段说明和 golden fixtures。

### 阶段 B：实现独立数据模块并训练验证（Day 4–7）

- 核心包不得 import Polar 或 Slime；
- Polar/Slime 只存在于 adapter、example 和 integration test；
- Processor 通过配置组合，不硬编码数据源；
- 实际 GRPO 采用双卡 staged loop，每次 policy 更新后重新 rollout。

## 全程执行规则

1. Polar 是 Reference Provider，不是核心依赖。
2. Slime/Megatron 是 Reference Consumer，不是核心依赖。
3. Adapter 只做字段映射，不静默伪造缺失 token、logprob、reward 或 policy version。
4. Processor 通过 capability 声明其前置字段；能力不足时拒绝、降级或 quarantine。
5. 核心只保留三个 Processor，新增功能必须证明与主问题直接相关。
6. `invalid_infrastructure` 不能当作模型 reward=0 进入训练。
7. 同一 GRPO group 不得混合 policy version。
8. 数据模块只输出 `ResampleRequest`，不直接控制上游 rollout。
9. On-policy GRPO 每个 iteration 使用当前 policy 重新采样；旧数据不无限复用。
10. 原始 artifact、处理配置、代码版本和输出 checksum 必须可追溯。

## 核心里程碑

- Day 1：现有硬件、SGLang、Megatron smoke 结论被冻结，无需无意义重做。
- Day 2：Polar Calculator 完整 rollout 不依赖 Trainer 即可运行。
- Day 3：至少一条真实 Coding Agent rollout 带 verifier 证据保存为 fixture。
- Day 4：同一 fixture 经 Polar/JSONL Adapter 进入统一 contract；核心测试不安装 Polar。
- Day 5：过滤基础设施无效样本后可构建完整同-policy group或产生补采请求。
- Day 6：自定义双卡配置完成真实 GRPO 更新与新 policy rollout。
- Day 7：Baseline 与处理后路径有可重复数据和模型侧结果。
