# Documentation Index

本页是当前文档入口。发生冲突时，以根目录 Scope、Plan 和 accepted ADR 为准。

## 当前权威文档

1. [项目范围](../PROJECT_SCOPE.md)
2. [实施计划](../PROJECT_PLAN.md)
3. [ADR-0002：双闭环 Agent Improvement Data Plane](adr/0002-agent-improvement-data-plane.md)
4. [Canonical Data Contract](data-contract.md)
5. [核心架构说明](architecture.md)
6. [Polar live 集成边界](polar-integration.md)
7. [Local Launcher-first 执行流程](local-launcher.md)

## 当前实现入口

- `src/contracts/`：不可变执行和训练契约；
- `src/producers/`：live rollout producer 边界；
- `src/validation/`：Episode 语义认证；
- `src/certification/`：统一 consumer eligibility；
- `src/assembly/`、`src/storage/`：事实组装和存储；
- `src/analysis/`：Harness 分析与 Gate；
- `src/learning/`、`src/training/`：认证后的学习视图与资格。
- `src/integrations/polar/`：官方 HTTP 边界和 stable result 严格适配；
- `src/integrations/slime/`：面向官方 Polar–Slime bridge 的 trainer-neutral admission envelope；
- `src/orchestration/`：单 GPU phase/resource/evidence handoff 契约；
- `src/launchers/`：Local Docker Launcher、容器资源与网络隔离边界；
- `src/capture/model_proxy.py`：模型请求/响应与 token/logprob/policy capability 证据；
- `configs/project.yaml`：Mac sandbox + 用户租用单 GPU 的有限资源拓扑。

## 历史材料

- [统一归档索引](../archive/README.md)
- [旧 Agentic RL Archive](archive/legacy-agentic-rl/README.md)
- `docs/archive/project-history/`：旧 README、Scope、Plan、release notes 和 runbook。

历史材料只能用于理解过去的实验，不能作为当前运行命令或能力声明。
