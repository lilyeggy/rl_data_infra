# Multi-Harness Agent Execution Data Plane：旧实施索引（已归档）

> 权威范围见 [PROJECT_SCOPE.md](PROJECT_SCOPE.md)。
> Day 1–4 是历史基础；项目转向后的 Day 5–7 已完成 V1/V2 本地数据平面基线。V2.1 decision-aware runtime 与 V2.2 Multi-Agent graph/training view 已完成当前 contract 验收，进入 V2.3 生产级采集、存储与吞吐。

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
| Day 5 | 已完成（V1） | Canonical execution contract、采集与 Episode 组装 | [计划](plans/day-05-execution-data-plane.md) |
| Day 6 | 已完成（V2） | 指标、故障归因、真实 Pi A/B 与 Regression Gate | [计划](plans/day-06-harness-evaluation.md) |
| Day 7 | 已完成（V2） | Harness Observatory、Training View、演示包装 | [计划](plans/day-07-observatory-and-packaging.md) |

旧 Day 1–4 文档用于历史证据，不再产生当前 TODO；Day 5–7 用于复盘已完成设计。V2.1 以 bounded FILE_NOT_FOUND recovery 为 reference workload，并已完成 Hook/controller/local runtime contract；真实 Pi decision capture 与效率 Gate 仍保持明确证据边界。V2.2 已完成 explicit Multi-Agent graph/training view contract；V2.3 当前聚焦 partitioned ingestion/storage。Gate 阈值保持冻结。

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

## Day 5–7 已完成顺序

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

## 当前验收命令

```bash
python3 -m unittest discover -s tests -v
python3 -m src.cli demo-v1 --output artifacts/v1-observability
python3 -m src.cli demo-v2 --output artifacts/v2-harness-decision
python3 -m src.cli demo-v21 --output artifacts/v2.1-recovery-replay
python3 -m src.cli demo-v21-runtime --output artifacts/v2.1-runtime-evidence
python3 -m src.cli demo-v22 --output artifacts/v2.2-multi-agent-evidence
python3 -m src.cli demo-v23 --output artifacts/v2.3-storage-smoke
python3 -m src.cli demo-v24 --output artifacts/v2.4-offline-training-view
python3 -m src.cli run-real-experiment --model deepseek-v4-flash \
  --output artifacts/v2.1-live
python3 -m src.cli inspect \
  --episodes artifacts/v2-harness-decision/episodes-control.jsonl \
  --episode-id episode-v2-real-control-1
```

V2 当前结果：control 0/3、candidate 3/3，错误循环 3→0；token +43.54%、latency +51.15%，因此 Gate 为 `REJECT`。

V2.1 live（真实服务器，deepseek-v4-flash）：control 0/3、candidate 3/3，循环 3→0；token +12%（PASS），latency 受 provider 方差影响（+41%~+472%），Gate 为 `REJECT`（latency）。真实脱敏 NDJSON fixtures 位于 `tests/fixtures/pi/v2.1-live/`。

## V2.1–V3 路线

```text
V1/V2：本地数据平面基线（保留）
           ↓
V2.1：深层 Trace 与决策可观测性
           ↓
V2.2：Multi-Agent 因果关系与训练语义
           ↓
V2.3：生产级采集、存储与吞吐
           ↓
V3：基于完整证据改进 Harness
```

V2.1 不修改 canonical TraceEvent/AgentEpisode 语义，也不放宽 reference Gate；V2.2 先收集真实 multi-agent artifacts 再冻结 Agent identity、branch、reward ownership 和 Training View；V2.3 只扩展传输、存储和查询规模；V3 才在完整 decision evidence 上实施 Harness policy 改进。

## 时间分配

- 70%：contract、capture、assembly、analysis、comparison 和真实证据；
- 20%：UI；
- 10%：README、实验报告和演示。

若时间不足，先缩任务数量与 UI 交互，不删减 canonical contract、对比控制变量、evidence 和 Regression Gate。
