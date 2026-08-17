# Documentation Index

> 本页是后续开发唯一入口。若其他文档与“权威文档”冲突，以权威文档为准。

## 权威文档

按顺序阅读：

1. [项目范围](../PROJECT_SCOPE.md)：定位、边界、非目标和完成定义；
2. [项目设计](../PROJECT_PLAN.md)：架构、数据模型、Capture、分析、对比、Gate 和 UI；
3. [实施索引](../IMPLEMENTATION_PLAN.md)：当前阶段、严格顺序和验收命令；
4. [Canonical Data Contract](data-contract.md)：`TraceEvent`、`AgentEpisode`、manifest、diagnosis 和 Gate 契约。

## 当前执行计划

只按下面三份计划继续实现：

1. [Day 5：Execution Data Plane](../plans/day-05-execution-data-plane.md)
2. [Day 6：Harness Evaluation](../plans/day-06-harness-evaluation.md)
3. [Day 7：Observatory & Packaging](../plans/day-07-observatory-and-packaging.md)

Day 5 未通过验收前，不提前把 UI 或复杂诊断作为主任务；Day 6 未产生可审计 comparison/Gate 前，不把 UI 图表当成项目有效性证明。

## 当前辅助文档

- [项目阅读顺序](learning/project-reading-order.md)
- [新旧数据契约学习笔记](learning/data-contract.md)
- [V1 Agent Infra 教学复盘](learning/v1-agent-infra-review-guide.md)
- [V2 真实 Pi Harness Decision 教学复盘](learning/v2-harness-decision-guide.md)
- [V2 交付与真实实验结果](releases/v2-harness-decision.md)
- [Polar / Orchard 边界](../notes/01-official-pipeline.md)
- [Harness Comparison Task Pilot](../notes/task-pilot-report.md)

## 历史归档

旧 Agentic RL / GRPO 路线、Day 1–4 执行材料和旧教学长文已移至：

- [Legacy Agentic RL Archive](archive/legacy-agentic-rl/README.md)

归档材料只用于解释已有代码和历史证据，不能作为当前 TODO、验收门或架构依据。

## 当前主线检查清单

开始一项任务前先确认它直接服务于以下至少一项：

- 多 Harness 执行捕获；
- canonical `TraceEvent` / `AgentEpisode`；
- trace integrity、artifact lineage 或 capability；
- metrics 与带证据的 failure attribution；
- controlled paired comparison；
- Regression Gate；
- 使用同一 canonical artifact 的 Observatory UI；
- optional Training View 兼容，但不阻塞主线。

若任务只服务于 Slime、Megatron、GRPO group 或模型 checkpoint 更新，应先放入 backlog，而不是本周主线。
