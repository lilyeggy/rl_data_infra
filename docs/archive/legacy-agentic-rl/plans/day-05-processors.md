# 已替代：旧 Day 5 GRPO Processor 计划

> 本文原计划实现 `FailureClassifier`、`SignalFilter` 和 `PolicyConsistentGroupBuilder`，服务于 GRPO training batch。项目于 2026-08-12 转向 Multi-Harness Agent Execution Data Plane 后，该计划不再是当前实施主线。

新的 Day 5 计划见：

- [Day 5：Multi-Harness Execution Data Plane](day-05-execution-data-plane.md)

旧思路中仍被保留的部分：

- task failure 与 infrastructure failure 必须分离；
- capability 缺失不能通过伪造字段绕过；
- reason code、checksum 和 lineage 必须可审计；
- `TrainingReadyBatch` 和 `ResampleRequest` 继续作为 optional Training View 使用。

不再作为本周验收条件的部分：

- reward-variance filtering；
- policy-consistent GRPO group builder；
- Slime Trainer Adapter 的前置 batch；
- 以训练数据进入 Trainer 作为项目有效性的主要证明。
