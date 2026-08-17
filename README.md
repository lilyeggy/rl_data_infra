# Multi-Harness Agent Execution Data Plane

面向 Agent Infra / Harness Runtime 的 Trace、Data 与 Observability 基础设施。系统把模型、工具、Sandbox、Harness Hook 和 Verifier 产生的异构事实统一为不可变 `TraceEvent`，再确定性组装为 `AgentEpisode`，用于回放、指标、故障归因、Harness 版本对比和回归决策；训练数据是可选下游视图，不是当前主链路。

## 当前可运行版本

`v1-observability` 已实现单次运行的纵向闭环：

```text
Capture adapters
      ↓
append-only raw-events.jsonl
      ↓
EpisodeAssembler
      ↓
AgentEpisode + integrity/lineage
      ↓
Metrics + evidence-linked attribution
```

V1 覆盖：

- 严格且不可变的 `TraceEvent`、manifest、`ArtifactRef`、`AgentEpisode`；
- append-only JSONL、`event_id` 幂等、冲突 quarantine、坏行隔离；
- 内容寻址 artifact store；
- model/tool/environment/verifier capture 与 optional Harness Hook；
- 乱序、重复、sequence gap、orphan span、missing terminal、partial/corrupt 组装语义；
- outcome、token、延迟、工具行为和 verifier 指标；
- SANDBOX / MODEL_BACKEND / EVALUATOR / HARNESS / UNKNOWN 的证据关联归因；
- 成功、有效任务失败和 infra-invalid 三类冻结演示数据。

V1 尚未实现 Harness A/B、Regression Gate 和 Observatory UI；这些属于 V2，不能用当前 artifact 声称“某项 Harness 修改已经被验证”。

## 快速开始

项目核心只依赖 Python 3.10+ 标准库，不需要 GPU、Polar、Slime 或训练框架。

```bash
python3 -m unittest discover -s tests -v
python3 -m src.cli demo-v1 --output artifacts/v1-observability
python3 -m src.cli inspect \
  --episodes artifacts/v1-observability/episodes.jsonl \
  --episode-id episode-tool-loop
```

生成结果：

```text
artifacts/v1-observability/
├── raw-events.jsonl       # append-only facts
├── blobs/                 # content-addressed evidence
├── episodes.jsonl         # canonical assembled executions
├── metrics.json           # derived observations
├── diagnoses.jsonl        # versioned rules + evidence IDs
└── summary.json            # release scope and limits
```

## 三条必须守住的语义

1. `FAILURE + VALID` 是一次可信的任务失败；`UNKNOWN + INFRA_INVALID` 是执行系统没有产生可信任务结果。二者不能都写成 reward 0。
2. `NOT_OBSERVABLE` 表示采集能力不存在，不能用 `0` 或文本猜测填补；`UNKNOWN` 表示能力存在但当前事实不足以判定。
3. raw event 是事实；metric、diagnosis 和 Gate 是带输入 checksum 与规则版本的派生结论，不能反写原始数据。

## 阅读入口

- 项目边界：[PROJECT_SCOPE.md](PROJECT_SCOPE.md)
- 架构与工程取舍：[docs/architecture.md](docs/architecture.md)
- V1 复盘教程：[docs/learning/v1-agent-infra-review-guide.md](docs/learning/v1-agent-infra-review-guide.md)
- 两版迭代计划：[docs/v1-v2-roadmap.md](docs/v1-v2-roadmap.md)
- V1 交付记录：[docs/releases/v1-observability.md](docs/releases/v1-observability.md)
- Canonical contract：[docs/data-contract.md](docs/data-contract.md)

## 准确的面试表述

> 我实现了一个 Harness-neutral 的 Agent Execution Data Plane。它采用 append-only event log 与确定性 assembler，把 model/tool/sandbox/verifier/harness 事实组装成带完整性、能力声明和数据血缘的 Episode；在此之上以版本化规则计算指标和失败归因，为后续受控 Harness A/B 与 Regression Gate 提供可审计证据。

当前不应表述为“自动优化 Harness”或“已完成大规模 benchmark 提升”。
