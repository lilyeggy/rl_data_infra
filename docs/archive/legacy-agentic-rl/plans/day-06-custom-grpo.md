# 已替代：旧 Day 6 自定义 GRPO 训练计划

> 本文原计划实现 Slime Adapter、双卡 Megatron optimizer step 和 checkpoint reload。项目于 2026-08-12 转向 Agent Infra 后，该训练闭环降级为可选扩展，不再阻塞主线完成。

新的 Day 6 计划见：

- [Day 6：Harness Evaluation、Failure Attribution 与 Regression Gate](day-06-harness-evaluation.md)

现有训练相关 contract 和硬件验证不删除；未来可以通过 `TrainingViewExporter` 将 `AgentEpisode` 投影为 `RolloutRecord`，再恢复 Trainer 集成。当前优先证明的是 Harness 数据可被统一捕获、诊断、比较并用于回归决策。
