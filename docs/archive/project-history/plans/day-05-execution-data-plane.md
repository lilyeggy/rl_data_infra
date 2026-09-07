# Day 5：Multi-Harness Execution Data Plane（已归档）

> 状态：已完成，作为 V1 Observability Loop 复盘材料保留。
> 目标：让两个不同 Harness 或 Harness 版本产生的数据进入同一个可审计 `AgentEpisode` 契约。

## 1. 阶段目标

1. 定义 `TraceEvent`、`AgentEpisode`、manifest、artifact 与 execution capability；
2. 实现 append-only JSONL event writer；
3. 实现确定性的 `EpisodeAssembler`；
4. 实现 Model 与 Sandbox/Environment 的最小采集路径；
5. 提供 optional Harness Hook 接口；
6. 接入两个 Harness/版本并保存真实或可重复 fixture；
7. 保持现有 `RolloutRecord` 测试不回归。

本阶段同时登记但不凭空解决 multi-agent 训练语义：`RolloutRecord v1` 只表示一条
线性训练 trace；`TraceEvent`/`AgentEpisode` 先保存可观察的 trace/span 父子关系。
只有取得真实子 Agent artifact 后，才冻结 agent identity、父子 Episode、并发分支、
reward ownership、credit aggregation 和多 trace Training View 导出规则。

## 2. 目录目标

```text
src/
├── contracts/
│   ├── trace_event.py
│   ├── agent_episode.py
│   ├── manifests.py
│   └── artifacts.py
├── capture/
│   ├── event_writer.py
│   ├── model_proxy.py
│   ├── environment.py
│   └── harness_hooks.py
└── assembly/
    └── episode_assembler.py
```

仅创建真正使用的模块，不为了目录完整添加空实现。

## 3. 先实现的事件

```text
MODEL_REQUEST / MODEL_RESPONSE
TOOL_CALL / TOOL_RESULT
SANDBOX_STARTED / SANDBOX_COMMAND / SANDBOX_FINISHED
VERIFICATION_STARTED / VERIFICATION_FINISHED
EPISODE_FINISHED
```

内部 context、compaction、retry 和 termination decision 通过 Hook 扩展；如果接入的 Harness 暂不支持，Episode capability 中明确缺失。

## 4. EventWriter

要求：

- 逐行 append JSONL；
- event 写入后不原地修改；
- 自动生成/校验 `event_id`、timestamp 和 schema version；
- 保留 producer、run、episode、trace、span 和 sequence；
- 对 stdout、patch 等大字段写 artifact 并记录 SHA-256；
- 同 `event_id` 重放不产生第二条 canonical event；
- secret/header 有明确 redaction 点。

MVP 不需要 Kafka、数据库或分布式事务。

## 5. EpisodeAssembler

输入 raw event stream，输出：

```text
episodes
quarantined_events
assembly_warnings
integrity_report
input/output checksums
```

必须处理：

- 乱序到达；
- duplicate event；
- sequence gap；
- missing terminal event；
- orphan span；
- capture 进程中断；
- 同 run 多个 Episode 交错。

assembler 可以排序和去重，但不能覆盖 raw JSONL。

## 6. Capture adapters

### Model Proxy

最小实现捕获 OpenAI-compatible request/response、tool call、finish reason、usage、latency 和 backend error。真实 backend 没有提供的 token ids/logprobs 保持缺失。

### Environment Adapter

最小接口记录 command、cwd、exit code、timeout、stdout/stderr artifact、changed file/patch 和 verifier evidence。它可以先包装项目内 reference runner，不要求先实现通用容器平台。

### Harness Hook

定义小而稳定的 emitter API：

```python
hook.emit_decision(...)
hook.emit_context_compacted(...)
hook.emit_retry_scheduled(...)
hook.emit_termination_decided(...)
```

Hook 无法接入时不阻塞 black-box capture。

## 7. 两个 Harness 的定义

第一版允许：

- 两个真实 Harness；或
- 同一 Harness 的两个版本/策略配置。

两边必须经过相同 Capture API 和 canonical contract。仅把 fixture 文件名改成 A/B 不算多 Harness 接入。

## 8. Fixture 矩阵

至少保存：

| 场景 | 期望 |
|---|---|
| task success | valid execution + verifier passed |
| valid task failure | valid execution + verifier failed |
| sandbox timeout/error | infra invalid + task outcome unknown |
| tool error then recovery | paired tool result and later successful action |
| partial trace | Episode retained with `PARTIAL` integrity |
| duplicate/out-of-order delivery | deterministic assembled output |

## 9. 测试

- schema validation 和 JSON roundtrip；
- event/episode identity；
- span parent/child；
- artifact checksum/lineage；
- capability 只按真实采集能力声明；
- assembler dedupe/order/partial；
- model request/response pairing；
- tool/sandbox error 与 task failure 不混淆；
- 现有 77 个测试（基线数字按实际测试结果更新）不回归。

## 10. 验收门

- [ ] `TraceEvent` / `AgentEpisode` / manifest / artifact contract 完成；
- [ ] raw event append-only 且可重放；
- [ ] assembler 对重复、乱序和 partial 有稳定输出；
- [ ] Model 与 Environment 核心事件均可捕获；
- [ ] 两个 Harness/版本产生兼容 Episode；
- [ ] UI 或分析层不需要读取源 Harness 私有日志；
- [ ] 不可观测 Harness decision 被明确标记；
- [ ] 多 Agent 运行若被采集，trace/span 因果关系不丢失；未验证的 agent identity、reward ownership 与 credit aggregation 明确标为证据不足；
- [ ] 旧 Rollout contract 与 Source Adapter 测试继续通过。

## 11. 阶段产物

```text
src/contracts/<execution contracts>
src/capture/
src/assembly/
tests/contracts/
tests/capture/
tests/assembly/
artifacts/day-05/<two harness runs>
docs/data-contract.md
```

## 12. 执行记录

```text
状态：COMPLETED（V1 canonical/capture/assembly 纵向闭环已完成；V2 已接入真实 Pi capture/adapter）
Harness/版本：reference-harness v1（冻结演示）；真实 v1/v2 对照尚未开始
事件类型：Model / Tool / Sandbox / Verifier / Harness Hook / Episode terminal
capability：显式 Episode capability；缺失项返回 NOT_OBSERVABLE
fixture：artifacts/v1-observability 三结果矩阵
测试：contract、writer、artifact、recorder、assembler、metrics、attribution 已覆盖；以最新全量测试输出为准
已知不可观测项：无 pricing 时 cost；无 Hook 时 Harness 内部 decision；无 native backend 字段时 token ids/logprobs
multi-agent 状态：RolloutRecord v1 不支持完整子 Agent 语义；等待真实 artifact 决定公共 identity 与 credit aggregation
```
