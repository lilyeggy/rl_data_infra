# V1 Agent Infra 复盘教程

这份文档供你隔几天回来后快速恢复上下文。目标不是背字段，而是能沿着一次真实失败解释系统为什么这样设计。

## 1. 先记住项目故事

Agent 运行失败后，我们通常只看到最后答案或一段散乱日志，无法确定是模型做错、Harness 丢失错误反馈、Sandbox 超时，还是 Verifier 本身坏了。V1 建立统一执行数据层：先保存事实，再组装 Episode，最后用可版本化规则计算观测和诊断。

```text
发生了什么？        TraceEvent
这些事实属于谁？    AgentEpisode + Manifest
数据可信吗？        Integrity + Capability
表现怎么样？        EpisodeMetrics
可能坏在哪里？      Diagnosis + Evidence + Rule Version
```

Harness 改进是最终用途，但 V1 只解决“看清当前行为”；V2 才解决“比较修改前后并决定是否接受”。

## 2. 推荐阅读顺序

### 第一步：一个不可变事实

读 `src/contracts/trace_event.py`。

重点观察：

- 为什么 identity、span、sequence 和 timestamp 都需要；
- 为什么 direct constructor 要求 enum，而 `from_dict` 负责恢复 enum；
- 为什么 attributes 会递归 freeze；
- 为什么 unknown top-level field 被拒绝；
- checksum 为什么来自 canonical JSON，而不是 Python `repr`。

你应该能回答：相同字段不同 dict key 顺序为什么得到同一个 checksum？因为 canonical serialization 对 object key 排序、禁止 NaN，并使用固定 separators。

### 第二步：写入事实而不是等待最终结果

读 `src/capture/event_writer.py` 和 `src/capture/recorder.py`。

重点观察：

- `EventWriter.append` 为什么相同 ID/相同 checksum 返回 `appended=False`；
- 为什么相同 ID/不同 checksum 必须报错；
- reader 为什么逐行隔离错误；
- `fsync` 解决什么，不能解决什么；
- secret 为什么必须在 raw storage 之前 redaction；
- `HarnessHook` 提供了哪些 black-box proxy 看不到的信息。

面试延伸：当前是单进程文件 writer。生产化可用 Kafka key=`event_id`、数据库 unique constraint 或对象存储 immutable shard；但不能只说“换 Kafka”，还要保留 schema、idempotency、quarantine 和 lineage 语义。

### 第三步：从 at-least-once stream 得到 Episode

读 `src/assembly/episode_assembler.py`。

按下面路径跟代码：

1. 保存原始输入 checksum；
2. 用 `event_id` 去重并识别 conflicting replay；
3. 缺少 `EpisodeContext` 就 quarantine；
4. 按 Episode 分组；
5. 用稳定 key 排序；
6. 检查 gap、sequence collision、orphan span、missing pair、missing artifact；
7. 从显式 `EPISODE_FINISHED` 声明恢复 outcome；
8. 生成 Integrity 和 SourceLineage；
9. 构造不可变 `AgentEpisode`。

核心取舍：Assembler 可以改变“读取顺序”，不能改变“原始事实”。排序后的 Episode 是 canonical view，raw arrival order 仍保存在 lineage。

### 第四步：理解三组不同状态

读 `src/contracts/agent_episode.py`。

不要混淆：

- TaskStatus：任务做对了吗？
- ExecutionValidity：这次执行能用于判断任务能力吗？
- IntegrityState：我们记录的数据完整可靠吗？

例子：Sandbox timeout 可以被完整记录，所以是 `UNKNOWN + INFRA_INVALID + COMPLETE`。这里 COMPLETE 描述 trace，不描述任务。

### 第五步：从事实派生可观测性

读 `src/analysis/metrics.py`。

重点观察 metric 分母：

- turn 由 `MODEL_RESPONSE` 计数；
- tool 指标只有 `TOOL_IO` capability 存在才计算；
- token 只读取 backend 明确给出的 usage；
- tool recovery 是“失败后是否出现后续成功结果”的 V1 操作性定义；
- cost 没有 pricing version 时保持不可观测。

一个 metric 的定义不仅是公式，还包括输入事件、capability、缺失值和 exclusion policy。V2 聚合 success rate 时还要明确 infra-invalid 不进入任务分母。

### 第六步：把“观察”与“归因”分开

读 `src/analysis/attribution.py`。

工具错误后重复完全相同动作是可观察行为；“Harness 没有正确反馈错误”是派生判断。如果 Hook 没启用，规则会降级为 UNKNOWN，而不是强行归责 Harness。

每个 Diagnosis 包含：

```text
layer / reason_code
evidence_event_ids / evidence_artifact_ids
confidence / explanation
rule_version / input_episode_checksum
```

这让诊断具备可审计性和可重算性。confidence 不是统计概率；它只是当前规则对证据充分度的版本化声明。

## 3. 用冻结 artifact 走一遍

先生成：

```bash
python3 -m src.cli demo-v1 --output artifacts/v1-observability
```

再检查工具循环：

```bash
python3 -m src.cli inspect \
  --episodes artifacts/v1-observability/episodes.jsonl \
  --episode-id episode-tool-loop
```

复盘时按这个顺序讲：

1. `MODEL_REQUEST/RESPONSE` 建立一次模型 turn；
2. 第一次 `TOOL_CALL` 调用不存在的文件；
3. `TOOL_RESULT FAILED` 关联 stderr artifact；
4. Hook 记录 `RETRY_UNCHANGED`；
5. 第二次 `TOOL_CALL` 参数完全相同；
6. Verifier 正常运行并返回失败，所以不是 infra-invalid；
7. terminal 声明 `FAILURE + VALID`；
8. Attribution rule 用失败 result 和重复 call 作为 evidence；
9. 未来 v2 修改结构化错误反馈后，用同模型、任务、环境和 Verifier 重跑。

再检查 infra timeout：它应该被归因到 SANDBOX，task status 必须是 UNKNOWN，不能说模型任务失败。

## 4. V1 是如何承接前面 Stage 1–3 的

Stage 1–3 让我们认识真实 rollout、模型服务、工具执行、Verifier 和基础设施故障；旧 `RolloutRecord` 重点检查 token/mask/reward 是否满足训练要求。转向后没有丢弃这些成果，而是把它放到更合理的下游：

```text
真实执行事实 → AgentEpisode → Harness analysis
                           └→ optional Training View → RolloutRecord
```

旧阶段最重要的语义继续保留：native token 不能伪造、verifier failure 与 verifier error 不同、infra failure 不能冒充 reward 0、capability 不能靠默认值补齐。

## 5. 与 Agentic RL 数据的关系

V1 Episode 可以成为训练候选数据，但不是看到 SUCCESS 就自动“训练就绪”。未来 exporter 还要检查：

- behavior model、target model 和 policy version；
- tokenizer/token ID 是否来自可信 producer；
- old logprob 是否存在以及属于哪个 policy；
- action/loss mask 的边界；
- Verifier、reward 和数据过滤规则版本；
- teacher trajectory 是蒸馏/SFT，还是学生自己的 on-policy rollout。

因此当前系统负责先产生可审计执行数据，不把 teacher off-policy 轨迹伪装成 GRPO on-policy 数据。

## 6. 面试高频追问

### 为什么不用 OpenTelemetry 原始 span 直接解决？

OTel 很适合作为传输和通用 tracing 基础，但 Agent 还需要 task outcome、execution validity、Verifier 语义、capability、experiment manifest 和 training lineage。可以把 OTel span 转为 `TraceEvent`，但通用 telemetry schema 不能自动提供这些领域约束。

### 你实现 exactly-once 了吗？

没有。V1 采用更现实的 at-least-once producer + idempotent ingestion：同一 `event_id` 的相同内容安全重放，不同内容进入冲突处理。跨进程 exactly-once 不在当前文件 writer 的承诺范围。

### sequence 和 timestamp 冲突时信谁？

sequence 是 producer 逻辑顺序，timestamp 用于 wall-clock/latency。Assembler 用稳定复合 key 展示，但 collision 会降低 integrity；它不会假装已经解决 producer bug。

### 为什么 diagnosis 不直接放进 Episode？

Episode 是稳定事实 envelope，诊断规则会演化。分离后能对同一历史 Episode 重跑 v2 规则，并比较诊断变化。

### 如何避免把模型错误归责给 Harness？

先要求可观测 evidence，再检查 capability。没有 Harness Hook 时只报告外部行为模式或 UNKNOWN；A/B 阶段还要固定模型、任务、环境和 Verifier，排除 confounder。

### 单卡 A6000 与当前 Infra 有什么关系？

核心 Trace/Data/Observability 测试不需要 GPU。A6000 可用于本地模型产生轨迹或后续学生模型训练；Harness A/B 可以直接固定同一模型 API。计算资源与 canonical schema 解耦正是数据平面的价值。

## 7. 复盘自测

不看文档尝试回答：

1. 为什么 `UNKNOWN + INFRA_INVALID + COMPLETE` 不矛盾？
2. conflicting duplicate 与 exact duplicate 的处理为什么不同？
3. 为什么没有 `TOOL_IO` 时 tool count 不能是 0？
4. raw arrival order 和 canonical event order 分别保存在哪里？
5. 哪些证据允许把 tool loop 归因到 Harness？
6. V2 为什么必须先检查 manifest compatibility，再计算 success delta？
7. teacher 轨迹为什么不能直接冒充学生 GRPO rollout？

能够用冻结的三个 Episode 举例回答这些问题，就已经掌握了 V1 的核心，而不需要逐行背代码。
