# Multi-Harness Agent Execution Data Plane：一周实施索引

> 权威范围见 [PROJECT_SCOPE.md](PROJECT_SCOPE.md)。
> Day 1–4 是已完成或已启动的历史基础；项目转向后，Day 5–7 是新的交付主线。

## 最终链路

```text
Harness A / Harness B
        ↓
Model Proxy + Sandbox Capture + Optional Harness Hooks
        ↓
append-only TraceEvent
        ↓
EpisodeAssembler → AgentEpisode
        ↓
Metrics + Failure Attribution + Paired Compare
        ↓
Regression Gate + Harness Observatory
        └── optional TrainingViewExporter → RolloutRecord
```

## 阶段状态

| 阶段 | 状态 | 作用 | 文档 |
|---|---|---|---|
| Day 1 | 已归档基础 | 环境、版本和硬件证据 | [历史计划](docs/archive/legacy-agentic-rl/plans/day-01-environment.md) |
| Day 2 | 已归档基础 | Polar Calculator rollout 与真实 fixture | [历史计划](docs/archive/legacy-agentic-rl/plans/day-02-polar-smoke.md) |
| Day 3 | 已归档、部分完成 | Coding/SWE 多轮任务与 verifier fixture | [历史计划](docs/archive/legacy-agentic-rl/plans/day-03-swegym-rollout.md) |
| Day 4 | 已归档基础 | `RolloutRecord`、capability、Polar/JSONL adapter | [历史计划](docs/archive/legacy-agentic-rl/plans/day-04-data-contract.md) |
| Day 5 | 新主线 | Canonical execution contract、采集与 Episode 组装 | [计划](plans/day-05-execution-data-plane.md) |
| Day 6 | 新主线 | 指标、故障归因、Harness A/B 与 Regression Gate | [计划](plans/day-06-harness-evaluation.md) |
| Day 7 | 新主线 | Harness Observatory、reference improvement、演示包装 | [计划](plans/day-07-observatory-and-packaging.md) |

旧 Day 1–4 文档用于历史证据，不再产生当前 TODO；当前实施从 Day 5 开始。

## 固定技术决策

| 层 | 第一版选择 | 说明 |
|---|---|---|
| Raw storage | append-only JSONL | 简单、可审计、便于 fixture |
| Canonical model | `TraceEvent` + `AgentEpisode` | 项目核心契约 |
| Capture | model proxy + environment adapter + optional hook | 支持不同可观测等级 |
| Query/aggregation | Python；需要时 DuckDB/Parquet | 不先建设数据库服务 |
| Diagnosis | versioned deterministic rules | LLM judge 仅可选辅助 |
| Comparison | controlled paired runs | 同 task/model/env/evaluator |
| Decision | `ACCEPT / REJECT / INSUFFICIENT_EVIDENCE` | 输出证据与阈值 |
| UI | 本地只读 Web UI | 消费 canonical JSON，不承载核心逻辑 |
| Training | optional exporter | 现有 Rollout contract 保留 |

## Day 5–7 严格顺序

### Day 5：先让不同 Harness 可被统一观察

1. 定义 `TraceEvent`、`AgentEpisode`、manifest、artifact 和 capability；
2. 实现 append-only writer 与 `EpisodeAssembler`；
3. 实现至少一条 model capture 和一条 environment capture 路径；
4. 接入两个 Harness 或同 Harness 两个版本；
5. 产出成功、任务失败、基础设施失败 fixture。

### Day 6：再证明数据可用于判断 Harness 修改

1. 计算 outcome、behavior、cost、latency、reliability 指标；
2. 做一级/二级故障归因并绑定 event/artifact evidence；
3. 固定实验变量，运行同任务 paired comparison；
4. 实现 Regression Gate；
5. 选择一个问题形成 v1/v2 reference improvement case。

### Day 7：最后做可展示的闭环

1. UI 展示 Episode、Trace Timeline 和 Harness Compare；
2. 运行 v1/v2 并生成可重复 artifact；
3. 输出 Gate 结论和限制；
4. 完成快速开始、架构图、实验报告和面试叙事；
5. 保留 optional training exporter，但不让它阻塞主线。

## 执行约束

1. 原始事件 append-only；诊断结果不得覆盖原始事实。
2. `event_id`、`episode_id`、`span_id`、`parent_span_id` 和 sequence 必须可追踪。
3. 对比前检查 manifest；控制变量不一致时不得输出强结论。
4. 不可观测字段明确标记，不从文本猜测内部 decision。
5. infra-invalid episode 不计入 Harness 任务失败率，但必须单独报告。
6. Gate 的规则、阈值、样本量和证据必须随结果保存。
7. UI 只是 consumer；所有核心结果可通过 CLI/JSON 获得。
8. 新代码不要求 Polar、Slime、Megatron 或 GPU 才能运行单元测试。

## 最小验收命令形态

最终 CLI 名称可在实现时调整，但能力必须对应：

```text
capture  <run-config>        → raw-events.jsonl
assemble raw-events.jsonl    → episodes.jsonl
inspect  <episode-id>        → trace + evidence
compare  <control> <candidate> → comparison.json
gate     comparison.json     → ACCEPT/REJECT/INSUFFICIENT_EVIDENCE
serve-ui <artifact-dir>      → Harness Observatory
```

## 时间分配

- 70%：contract、capture、assembly、analysis、comparison 和真实证据；
- 20%：UI；
- 10%：README、实验报告和演示。

若时间不足，先缩任务数量与 UI 交互，不删减 canonical contract、对比控制变量、evidence 和 Regression Gate。
