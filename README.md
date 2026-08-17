# Multi-Harness Agent Execution Data Plane

面向 Agent Infra / Harness Runtime 的 Trace、Data 与 Observability 基础设施。系统把模型、工具、Sandbox、Harness Hook 和 Verifier 产生的异构事实统一为不可变 `TraceEvent`，再确定性组装为 `AgentEpisode`，用于回放、指标、故障归因、Harness 版本对比和回归决策；训练数据是可选下游视图，不是当前主链路。

## 当前可运行版本

`v2-harness-decision` 已在 V1 单次运行纵向闭环上，实现真实 Pi 的横向 Harness 决策闭环：

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

当前覆盖：

- 严格且不可变的 `TraceEvent`、manifest、`ArtifactRef`、`AgentEpisode`；
- append-only JSONL、`event_id` 幂等、冲突 quarantine、坏行隔离；
- 内容寻址 artifact store；
- model/tool/environment/verifier capture 与 optional Harness Hook；
- 乱序、重复、sequence gap、orphan span、missing terminal、partial/corrupt 组装语义；
- outcome、token、延迟、工具行为和 verifier 指标；
- SANDBOX / MODEL_BACKEND / EVALUATOR / HARNESS / UNKNOWN 的证据关联归因；
- 成功、有效任务失败和 infra-invalid 三类冻结演示数据。
- Pi 0.84.2 + 固定 `opencode-go/gpt-5.6-luna` 的真实 NDJSON capture/adapter；
- `ExperimentManifest`、逐对 compatibility、paired comparison；
- 三态 Regression Gate、只读 Observatory、`TrainingCandidateView`；
- 3 对真实 error-recovery reference case：control 0/3、candidate 3/3，但因 token/latency 超预算，Gate 诚实输出 `REJECT`。

## 快速开始

项目核心只依赖 Python 3.10+ 标准库，不需要 GPU、Polar、Slime 或训练框架。

```bash
python3 -m unittest discover -s tests -v
python3 -m src.cli demo-v1 --output artifacts/v1-observability
python3 -m src.cli demo-v2 --output artifacts/v2-harness-decision
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
- V2 教学复盘：[docs/learning/v2-harness-decision-guide.md](docs/learning/v2-harness-decision-guide.md)
- V2 交付记录：[docs/releases/v2-harness-decision.md](docs/releases/v2-harness-decision.md)
- Canonical contract：[docs/data-contract.md](docs/data-contract.md)

## 准确的面试表述

> 我实现了一个 Harness-neutral 的 Agent Execution Data Plane。它把 model/tool/sandbox/verifier/harness 事实确定性组装成可审计 Episode，并在真实 Pi 轨迹上完成受控 Harness A/B、failure attribution 和三态 Regression Gate；每个上线结论都能下钻到原始事件证据。

当前不应表述为“已完成大规模 benchmark 提升”或“teacher trace 已经是 on-policy RL rollout”。
