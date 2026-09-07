# Agent Improvement Data Plane｜简历项目描述

> 状态：当前版本（14B SFT 与固定 Holdout 已执行，candidate 未通过 Gate；Agentic RL 未完成）
> 更新日期：2026-08-26

## 项目定位

面向 Agentic RL 数据生产，构建从 Agent 隔离执行、轨迹采集、结果验证到训练数据生成的端到端工程闭环。

## 简历描述

- 设计统一执行数据模型，以 `ExecutionIdentity`、`TraceEvent`、`AgentEpisode`、`ExecutionBundle` 贯通模型调用、工具执行、Sandbox、Verifier 与产物，实现跨组件轨迹关联和完整 Lineage。
- 搭建容器化 Rollout Workflow，编排 Model Proxy、Agent Harness、Verifier 与 Finalizer，将原始事件和 Artifact 确定性组装为可回放、可校验的标准轨迹。
- 实现 Fail-closed 数据认证与 Dataset Compiler，区分任务失败和基础设施异常，校验轨迹完整性、Policy 一致性与 Verifier 证据，将合格轨迹编译为 Evaluation、SFT、Preference 和 RL-ready 数据。
- 针对重复、乱序、缺失、超时和中断，实现幂等写入、Checksum、Quarantine、Backpressure 与恢复机制，并通过真实 Docker 闭环和 300+ 自动化测试验证关键边界。

## 当前能力边界

当前版本可以表述为“已完成可信 Agent 轨迹生产和训练数据准入基础设施”，但暂不表述为：

- 已完成正式 Agentic RL 模型更新；
- 已证明模型在通用 Benchmark 上获得稳定提升；
- 已完成大规模、多卡或异步 RL 训练；
- 已验证生产级海量并发能力。

当前已在 A6000 上完成一次 Qwen2.5-Coder-14B LoRA SFT，并使用同一 Pi Harness、task snapshot 和 verifier 对 base/candidate 做固定 unseen DEV 对照。两者均未解决任务，Gate 为 `REJECT / NO_IMPROVEMENT`，因此这次训练只证明数据到训练再到回溯评测的工程链路可运行，不证明模型能力提升。

## 训练闭环完成后的更新项

只有后续 candidate 通过固定 Holdout，或完成正式 Agentic RL 更新并取得可复核提升后，才更新本文件的第四条简历描述：

```text
policy-v0
  → real rollout
  → certified dataset
  → SFT / Agentic RL trainer
  → policy-v1
  → fixed holdout evaluation
```

建议替换文案：

> 将认证轨迹接入 SFT/Agentic RL Trainer，完成 policy-v0 → policy-v1 模型更新，并在固定 Holdout 上对比任务成功率、成本和轨迹行为，验证可信数据生产闭环的有效性。

更新时必须补充实际实验信息：

- 模型、Checkpoint 与 Adapter 版本；
- 训练样本数量及其 Dataset Manifest；
- Trainer 和主要训练配置；
- Holdout 数据集及隔离证据；
- policy-v0 / policy-v1 的成功率、成本、延迟和异常率；
- 未提升或退化的指标及其原因。
