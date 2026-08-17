# Multi-Harness Agent Execution Data Plane

> 状态：方向已冻结，进入一周实现阶段（2026-08-12）
> 目标岗位：Agent Infra / Agent Harness / Runtime Data / Evaluation / Observability
> 权威边界：[PROJECT_SCOPE.md](PROJECT_SCOPE.md)

## 1. 项目定义

本项目构建一个面向多 Harness 的统一执行数据层。它从模型代理、Sandbox/Environment 和可选 Harness Hook 捕获运行事实，将异构事件组装为 `AgentEpisode`，再提供回放、指标、故障归因、受控 A/B 对比和版本回归判断。

项目不以“更新模型参数”为主目标，也不声称自动优化 Harness。它回答的是：

> 当 Harness 发生变化时，我们能否用可复现、可审计的数据判断行为为什么变化，以及这项变化是否值得发布？

## 2. 与现有工作的关系

项目早期以 Rollout-Agnostic Agentic RL Data Pipeline 为目标，已经实现了一批有价值的基础组件：

- 版本化 `RolloutRecord` / batch contract；
- capability gate；
- Polar 和 JSONL Source Adapter；
- checksum、lineage 和 fixture 测试；
- execution status、verifier status 和 tool event 的部分表达。

这些代码不删除。新的架构把它们放在正确位置：

```text
                     新主线
raw TraceEvent → AgentEpisode → Analyze / Compare / Gate / UI
                         │
                         └── optional TrainingViewExporter
                                      ↓
                            现有 RolloutRecord / TrainingReadyBatch
```

因此改造方式是“增加更通用的上游层”，不是重写现有代码。

## 3. 系统边界

### 3.1 Harness 负责

- Agent loop 和任务状态；
- prompt/context 构造；
- tool selection 和参数；
- retry、compaction、verification、termination 等策略；
- 对工具和环境产生实际调用。

### 3.2 本项目负责

- 捕获不同 Harness 的可观测运行事实；
- 保存 append-only raw events；
- 将事件规范化并组装为 Episode；
- 声明本次采集具备哪些 capability；
- 计算统一指标和故障归因；
- 在控制变量一致时比较 Harness/版本；
- 输出可解释的 Regression Gate 结论；
- 展示 Episode、Trace 和对比证据；
- 可选导出训练视图。

### 3.3 外部系统负责

- 模型推理服务；
- Sandbox/容器的实际隔离和资源执行；
- task benchmark 与 verifier 的业务正确性；
- 人工或自动提出 Harness patch；
- Trainer 的 loss、optimizer 和 checkpoint（如果使用训练 exporter）。

## 4. 总体架构

```text
               Experiment Manifest
     task / model / seed / env / tools / evaluator
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
        Harness A                 Harness B/v2
            │                         │
            └──────── Instrumentation ┘
                ├─ Model Proxy
                ├─ Sandbox Adapter
                └─ Optional Harness Hook
                         │
                         ▼
              append-only Raw Event Store
                         │
                         ▼
                  EpisodeAssembler
                         │
                         ▼
                    AgentEpisode
               ┌─────────┼──────────┐
               ▼         ▼          ▼
           Trace View  Metrics  Failure Attribution
               └─────────┼──────────┘
                         ▼
                  Paired Comparison
                         │
                         ▼
                   Regression Gate
             ┌───────────┴───────────┐
             ▼                       ▼
     Harness Observatory      TrainingViewExporter
```

## 5. Canonical Contract

### 5.1 TraceEvent

首批 event type：

```text
MODEL_REQUEST
MODEL_RESPONSE
TOOL_CALL
TOOL_RESULT
SANDBOX_STARTED
SANDBOX_COMMAND
SANDBOX_FINISHED
VERIFICATION_STARTED
VERIFICATION_FINISHED
EPISODE_FINISHED
```

Hook-enabled 时可增加：

```text
HARNESS_DECISION
CONTEXT_SELECTED
CONTEXT_COMPACTED
RETRY_SCHEDULED
TERMINATION_DECIDED
```

每个事件至少包含：

```text
schema_version
event_id / episode_id / trace_id
span_id / parent_span_id
sequence / timestamp
event_type / component / status / attempt
attributes / artifact_refs
```

### 5.2 AgentEpisode

一次 task execution 的 envelope：

```text
episode_id
task_id
run_id
harness_manifest
model_manifest
environment_manifest
evaluator_manifest
capabilities
events
artifacts
outcome
termination
integrity
```

`AgentEpisode` 可以是 partial。assembler 不因为缺事件而编造内容，而是记录 gap、重复、乱序修复和完整性状态。

### 5.3 Manifest

Harness 对比前至少检查：

- harness name、version、config digest；
- model/provider/revision、sampling 参数；
- task/dataset/revision；
- sandbox image/runtime revision；
- tool schema digest；
- evaluator/verifier revision；
- random seed 和时间限制。

除 Harness 变量外存在不一致时，结果必须标记 confounder 或 `INSUFFICIENT_EVIDENCE`。

### 5.4 ArtifactRef

大型或二进制内容不直接塞入事件：

```text
artifact_id
kind
uri/path
media_type
sha256
size_bytes
producer_event_id
```

## 6. Capture 设计

### 6.1 Model Proxy

提供 OpenAI-compatible 接口，透明转发 Harness 请求并记录：

- messages、tool definitions 和参数；
- response message、tool call 和 finish reason；
- input/output token、latency、backend status；
- request/response checksum；
- backend 实际提供的 token ids/logprobs。

不记录或声称模型未显式返回的隐藏 chain-of-thought。

### 6.2 Sandbox/Environment Adapter

记录：

- environment create/start/stop；
- command、cwd、exit code、stdout/stderr artifact；
- timeout、resource error 和 lifecycle failure；
- changed files、patch 和 final artifact；
- verifier start/result/evidence。

### 6.3 Harness Hook

Hook 是增强能力，不是接入前提。它用于记录只能由 Harness 自己知道的 decision：

- 哪些上下文被保留或丢弃；
- 何时 compact、compact 前后 token；
- 为什么 retry；
- 为什么继续、验证或终止；
- loop detector 和 task state 的变化。

没有 Hook 时，相关字段为 `NOT_OBSERVABLE`，不能由外部轨迹反推成事实。

## 7. Storage 与一致性

第一版使用本地文件即可证明架构：

```text
artifacts/<run_id>/
├── manifest.json
├── raw-events.jsonl
├── episodes.jsonl
├── metrics.json
├── diagnoses.jsonl
├── comparison.json
└── gate-result.json
```

规则：

- raw event append-only；
- event 以 `event_id` 去重；
- assembler 以 `(episode_id, sequence)` 排序，并保留原始顺序信息；
- 原始事实、派生指标、诊断和 Gate 结果分别保存；
- 所有派生产物包含输入 checksum 和生成器版本；
- stdout、patch 等大字段通过 `ArtifactRef` 关联。

后续规模化可以把 Episode/metric 写为 Parquet 并用 DuckDB 查询，但不作为一周 MVP 的必要条件。

## 8. Metrics

### Outcome

- success / valid task failure / infra invalid；
- verifier outcome；
- termination reason；
- failure layer。

### Behavior

- turns 和 tool calls；
- duplicate-action rate；
- loop count；
- tool-error recovery rate；
- premature termination；
- verification attempts。

### Context（需要 Hook）

- prompt/context token growth；
- compaction count；
- before/after token；
- preserved/dropped item statistics。

### Performance 与可靠性

- model/tool/sandbox/verifier latency；
- wall time；
- input/output token 与估算 cost；
- timeout、retry、worker failure、incomplete trace 和 recovery time。

## 9. Failure Attribution

一级类别：

```text
MODEL
HARNESS
SANDBOX
MODEL_BACKEND
EVALUATOR
EXTERNAL_SERVICE
UNKNOWN
```

首批 Harness 二级 reason code：

```text
CONTEXT_LOSS
COMPACTION_INFORMATION_LOSS
TOOL_SCHEMA_FAILURE
TOOL_ERROR_FEEDBACK_LOSS
LOOP_CONTROL_FAILURE
RETRY_POLICY_FAILURE
PREMATURE_TERMINATION
VERIFICATION_POLICY_FAILURE
STATE_MANAGEMENT_FAILURE
```

每个诊断包含：

```text
diagnosis_id
episode_id
layer / reason_code
evidence_event_ids
evidence_artifact_ids
confidence
rule_version
explanation
```

允许 multi-label、`UNKNOWN` 和 `INSUFFICIENT_EVIDENCE`。第一版优先确定性规则，LLM judge 只能作为有标签的辅助意见。

## 10. Harness 对比协议

### 10.1 控制变量

固定：

```text
model revision + sampling
task revision
sandbox image
tool schema
evaluator/verifier
seed and limits
```

只改变 Harness version 或单项策略。每个 task 形成 control/candidate 配对。

### 10.2 Comparison 输出

- manifest compatibility；
- paired outcome diff；
- turns、tool、token、cost、latency diff；
- loop、duplicate action、error recovery 和 verifier diff；
- failure slice diff；
- infra-invalid rate；
- 样本量与不确定性说明；
- severe regression 列表。

### 10.3 Regression Gate

Gate 读取版本化阈值，例如：

```text
candidate success 不低于 control
目标 failure slice 改善
token/cost/latency 增幅不超过阈值
不得出现新的 severe regression
infra-invalid rate 不增加
样本与 paired coverage 足够
```

输出：

- `ACCEPT`：满足改进目标且无越线回归；
- `REJECT`：明确违反阈值；
- `INSUFFICIENT_EVIDENCE`：样本、能力或控制变量不足。

Gate 不是统计显著性的伪装。MVP 的小样本用于证明机制和逐任务证据，不宣称普遍提升。

## 11. Reference Improvement Case

第一版只选择一个主案例，优先级如下：

### 首选：Structured Tool Error Feedback

```text
v1: 把原始 stderr 直接返回给模型
v2: 返回 error_type、command、exit_code、stderr_summary、retryable
```

观察 tool error 后的恢复、重复命令、turn、token、success 和延迟。

### 备选：Verification Completion Gate

```text
v1: 模型声明完成即可终止
v2: 必须获得 verifier evidence 才允许成功终止
```

观察 premature termination、verification attempts、success 和额外成本。

这项修改可以人工提出；项目价值在于捕获证据、定位问题和验证修改，而不是自动生成 patch。

## 12. Harness Observatory

UI 是只读 consumer，直接读取标准化 artifact。

### Episode Explorer

- 按 harness/version、model、task、outcome、failure、termination 和 capability 过滤；
- 展示 duration、tokens、tool calls、verifier 和 integrity。

### Trace Timeline

- Model、Tool、Sandbox、Harness、Verifier 分 lane；
- 显示 parent/child span、状态、latency 和 artifact；
- 将 diagnosis 定位到具体 evidence；
- 明确展示未采集能力。

### Harness Compare

- control/candidate 同任务并排；
- 对齐关键行为差异；
- 展示聚合指标、失败切片和回归；
- 展示 Gate verdict、规则和证据。

## 13. 代码结构目标

```text
src/
├── contracts/
│   ├── trace_event.py
│   ├── agent_episode.py
│   ├── manifests.py
│   ├── artifacts.py
│   └── rollout_record.py       # existing Training View
├── capture/
│   ├── event_writer.py
│   ├── model_proxy.py
│   ├── environment.py
│   └── harness_hooks.py
├── assembly/
│   └── episode_assembler.py
├── analysis/
│   ├── metrics.py
│   ├── attribution.py
│   ├── compare.py
│   └── regression_gate.py
├── exporters/
│   └── training_view.py
├── cli.py
└── ui/                         # 或独立 frontend 目录
```

目录是目标边界，不要求一次性创建空文件。

## 14. 测试策略

### Contract

- schema validation；
- event ordering、dedupe、partial episode；
- span parent/child；
- checksum 和 artifact lineage；
- capability truthfulness。

### Capture/Assembly

- request/response 与 tool call pairing；
- sandbox success/error/timeout；
- out-of-order events；
- missing terminal event；
- duplicate delivery 和 crash recovery。

### Analysis

- failure reason rules；
- UNKNOWN/INSUFFICIENT_EVIDENCE；
- metric denominator；
- infra invalid exclusion；
- manifest mismatch；
- paired comparison 和 threshold boundary。

### End-to-End

- 两个 Harness/版本运行相同任务；
- 生成统一 Episode；
- 发现一个目标 failure；
- v2 修改后重跑；
- Gate 输出可解释结论；
- UI 使用同一批 artifact 展示结果。

## 15. 完成定义

只有以下条件全部满足，项目主线才算完成：

1. 两个 Harness 或版本能进入同一 execution contract；
2. Model、Tool/Sandbox 和 Verifier 核心事件可捕获；
3. Episode assembler 正确处理重复、乱序和不完整事件；
4. Trace 可以回放且能追溯 artifact；
5. metrics 和 failure attribution 有稳定规则与测试；
6. comparison 会检查控制变量，不比较不可比运行；
7. Regression Gate 能输出三态结论和证据；
8. 至少一个 reference Harness 修改完成 v1/v2 闭环；
9. UI 展示 Episode、Timeline 和 Compare；
10. 现有训练 contract 仍可用，但训练不阻塞项目验收；
11. 所有数字、截图和简历结论能反查 artifact；
12. 文档明确写出限制和不可观测能力。

## 16. 面试表述

一句话版本：

> 构建 Multi-Harness Agent Execution Data Plane，通过 model proxy、sandbox capture 和 optional hooks 将不同 Harness 的执行统一为可审计 AgentEpisode，并支持 trace replay、失败归因、受控 A/B 对比和 regression gate。

项目故事版本：

> 我先通过真实 Polar rollout 建立了 token、tool、verifier 和 failure evidence 的可靠采集基础；随后把训练导向的 RolloutRecord 降为可选视图，新增更通用的 TraceEvent/AgentEpisode 数据面。系统能够在相同模型、任务和环境下比较两个 Harness 版本，定位工具错误反馈或终止策略问题，并通过版本化 Gate 判断修改是否改善，同时在 UI 中把结论追溯到具体 span 和 artifact。

## 17. 文档索引

- [项目范围](PROJECT_SCOPE.md)
- [一周实施索引](IMPLEMENTATION_PLAN.md)
- [Canonical Data Contract](docs/data-contract.md)
- [Day 5：Execution Data Plane](plans/day-05-execution-data-plane.md)
- [Day 6：Harness Evaluation](plans/day-06-harness-evaluation.md)
- [Day 7：Observatory & Packaging](plans/day-07-observatory-and-packaging.md)
- [项目阅读顺序](docs/learning/project-reading-order.md)
