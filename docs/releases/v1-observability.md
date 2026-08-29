# V1 Release：Single-run Observability Loop

> Release：`v1-observability`
> 日期：2026-08-17
> 状态：核心闭环完成；Harness A/B 属于 V2

## 本版回答的问题

> 一次 Agent/Harness 执行能否在不依赖具体 Harness 和训练框架的前提下，被可靠采集、组装为可审计 Episode，并产生有证据的指标与失败归因？

答案是：V1 已用冻结的三结果演示和单元测试完成机制验证。

## 实现清单

| 层 | 实现 |
|---|---|
| Contract | `TraceEvent`、四类 manifest、`ArtifactRef`、`AgentEpisode`、Outcome、Integrity、Capability |
| Capture | `EventWriter`、`EventJsonlReader`、`ArtifactStore`、`TraceRecorder`、`EnvironmentCapture`、`HarnessHook` |
| Assembly | 多 Episode 分组、排序、去重、冲突 quarantine、gap/orphan/missing 检查、partial/corrupt |
| Metrics | duration、turn、tool、duplicate、loop、recovery、verification、token、component latency |
| Attribution | trace integrity、infra component、tool-error loop、unknown fallback |
| Interface | `demo-v1` 与 `inspect` CLI |
| Evidence | success、valid task failure、infra invalid 三条 canonical Episode |

## 冻结演示结果

运行：

```bash
python3 -m src.cli demo-v1 --output artifacts/v1-observability
```

预期摘要：

| Episode | Task | Validity | Integrity | Diagnosis |
|---|---|---|---|---|
| `episode-success` | SUCCESS | VALID | COMPLETE | 无失败诊断 |
| `episode-tool-loop` | FAILURE | VALID | COMPLETE | `TOOL_ERROR_FEEDBACK_LOSS` |
| `episode-infra-timeout` | UNKNOWN | INFRA_INVALID | COMPLETE | `SANDBOX_OPERATION_TIMEOUT` |

`episode-tool-loop` 的 diagnosis evidence 指向第一次失败的 `TOOL_RESULT`、Hook 捕获的 retry decision 和下一次参数完全相同的 `TOOL_CALL`；第一个 event 还能继续追到内容寻址 stderr artifact。

## 关键工程决定

1. raw log 使用 at-least-once + idempotent consumer 语义，而不是假设 exactly-once delivery。
2. metadata 通过 `EpisodeContext` 提供；缺失 context 的事件 quarantine，Assembler 不从日志猜 task/model/Harness。
3. Episode 可以 partial；capture 中断不是删除数据的理由。
4. 指标缺少 capability 时返回 `None + NOT_OBSERVABLE`，不填 0。
5. attribution 是 versioned derived record，不进入 canonical Episode truth。
6. `estimated_cost` 在没有 versioned pricing 时保持不可观测。

## 测试覆盖重点

- nested JSON freeze、strict top-level schema、UTC、enum 和 checksum；
- event append、重启幂等、冲突 replay、坏行隔离、secret redaction；
- artifact content addressing；
- out-of-order、duplicate、sequence gap、missing terminal、missing context；
- Episode roundtrip 与 input/output lineage；
- capability-aware metric；
- success/infra-invalid/tool-loop attribution 和 black-box 降级。

## 已知限制

- EventWriter 是单进程 writer，没有跨进程锁；
- model capture 目前是嵌入式 recorder API，不是已部署的 OpenAI-compatible HTTP reverse proxy；
- environment capture 接收外部 Sandbox 的执行结果，不负责容器隔离；
- attribution v1 是窄规则集，不等于通用根因分析；
- 未实现跨 Harness comparison、Gate、UI 和真实 reference patch；
- 未导出或训练 Agentic RL 模型。

## V2 入口

V2 不会推倒 V1。它将用同一份 `AgentEpisode` 增加 manifest compatibility、paired comparison、Regression Gate 和 Observatory，并用 Structured Tool Error Feedback 完成一次可追溯 Harness 改进案例。
