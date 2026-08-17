# Legacy Agentic RL Documentation Archive

> 归档日期：2026-08-12
> 状态：只读历史材料，不是当前实施计划。

## 为什么归档

项目最初定位为 Rollout-Agnostic Agentic RL Data Pipeline，目标是把 Polar rollout 转成 policy-consistent GRPO batch，并接入 Slime/Megatron。当前项目已经转向 Multi-Harness Agent Execution Data Plane，因此旧路线中的 TrainingReadyBatch、GRPO group、Slime Adapter 和双卡训练不再是本周主线。

这些文档仍包含真实 Polar fixture、硬件 smoke、字段映射和旧 contract 的设计依据，所以保留而不删除。

## 归档内容

```text
legacy-agentic-rl/
├── day1实现.md
├── plans/                  # 旧 Day 1–7 计划与替代说明
├── docs/
│   ├── runbooks/           # Polar Calculator/Coding 历史执行手册
│   └── learning/           # 旧实现深度导读
└── notes/                  # Day 1–2 收口与 Polar 字段审计
```

## 可以继续复用

- Polar raw artifact 与字段来源；
- task failure / infrastructure failure 的语义区分；
- checksum、lineage、fixture redaction；
- capability gate 和“不伪造缺失训练字段”；
- `RolloutRecord v1`、Polar/JSONL Adapter 的实现解释；
- 已完成的环境和硬件事实。

## 不再执行

- 旧 Day 5 GRPO Processor 计划；
- 旧 Day 6 Slime/Megatron 训练闭环；
- 旧 Day 7 以模型训练前后结果作为主要验收；
- 任何与当前权威文档冲突的 TODO 或完成定义。

## 返回当前文档

- [Documentation Index](../../README.md)
- [Project Scope](../../../PROJECT_SCOPE.md)
- [Project Plan](../../../PROJECT_PLAN.md)
- [Implementation Plan](../../../IMPLEMENTATION_PLAN.md)
- [Current Data Contract](../../data-contract.md)

归档内旧文档保留原始叙述和部分原始相对路径；阅读时应通过本页返回当前文档，不沿旧“下一步”继续执行。
