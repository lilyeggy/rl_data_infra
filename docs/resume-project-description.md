# Agent Improvement Data Plane｜简历项目描述

> 状态：当前版本（SFT 与三轮单卡 on-policy GRPO cycle 已执行；candidate 未取得稳定提升，未晋升正式 policy-v1）
> 更新日期：2026-09-09

## 项目定位

面向 Agentic RL 数据生产，构建从 Agent 隔离执行、轨迹采集、结果验证到训练数据生成的端到端工程闭环。

## 简历描述

- 设计统一执行数据模型，以 `ExecutionIdentity`、`TraceEvent`、`AgentEpisode`、`ExecutionBundle` 贯通模型调用、工具执行、Sandbox、Verifier 与产物，实现跨组件轨迹关联和完整 Lineage。
- 搭建容器化 Rollout Workflow，编排 Model Proxy、Agent Harness、Verifier 与 Finalizer，将原始事件和 Artifact 确定性组装为可回放、可校验的标准轨迹。
- 实现 Fail-closed 数据认证与 Dataset Compiler，区分任务失败和基础设施异常，校验轨迹完整性、Policy 一致性与 Verifier 证据，将合格轨迹编译为 Evaluation、SFT、Preference 和 RL-ready 数据。
- 针对重复、乱序、缺失、超时和中断，实现幂等写入、Checksum、Quarantine、Backpressure（有界队列 + HTTP 429）与断点恢复机制，并通过真实 Docker 闭环和 220+ 自动化测试验证关键边界。

## 当前能力边界

当前版本可以表述为“已完成可信 Agent 轨迹生产与训练数据准入基础设施，并在单卡上跑通 SFT + on-policy GRPO 的完整训练循环（policy-v0 → v3）”，但暂不表述为：

- RL 或 SFT 带来可复核的模型能力提升；
- 已完成大规模、多卡或异步 RL 训练；
- 已接入 Slime/verl 等正式 trainer；
- 已验证生产级海量并发能力。

## 已执行的实验事实（面试口径，全部可回溯到 `docs/experiments/`）

- 模型：Qwen2.5-Coder-14B-Instruct + LoRA（clean-v2 SFT epoch2 为 policy-v0）；
- 数据：APPS train split 认证轨迹训练包；RL rollout 任务取自 APPS train 难度分层选择（HumanEval/MBPP/BigCodeBench 全部封存为评测集，泄漏防护为 split 不相交，未做语料级 decontamination）；
- Trainer：独立极简 GRPO LoRA trainer（PPO-clip + KL，`grpo-lora-trainer/v1`）；Slime 正式接入为后续工作；
- 循环：apps-rl-cycle-001（3 题 12 轨迹微型闭环）→ 002（20 题 80 轨迹）→ 003（难度路由 + 软用例奖励）；
- 固定 APPS holdout 50 题：base 48% / SFT-v0 50% / RL-v3 50%；
- 官方 EvalPlus / BigCodeBench（HumanEval 164 / MBPP 378 / BigCodeBench-Hard 148）：HumanEval base 87.20% / SFT-v0 89.02% / RL-v2 86.59% / RL-v3 87.20%（与 base 持平；HumanEval+ 81.10% 低于 base 84.15%）；MBPP base 84.66% / SFT-v0 83.86% / RL-v2 83.07%；BigCodeBench-Hard 25.00% / 22.97% / 20.95%；
- 结论：数据生产 → 认证 → 训练 → 评测 → Gate 的工程链路完整、可运行、可审计，但 candidate 未取得超过 base/SFT 的稳定提升，Gate 未晋升正式 policy-v1。

## 措辞约束

在 candidate 于固定 holdout 取得可复核提升之前，简历与面试叙事停留在“可信数据生产 + 完整训练循环已验证”，不得表述为“RL 提升了模型能力”。数据清洗的代表性故事是 `docs/experiments/sft-apps-260905-data-quality-postmortem.md`（`[REDACTED]` thinking token、宿主绝对路径泄漏、逐轮分布偏移三个真实缺陷及其 fail-closed 修复）。
