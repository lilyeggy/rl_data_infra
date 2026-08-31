# 学习笔记：从 Training Record 到 Agent Execution Contract（已归档）

## 为什么需要两层数据模型

旧 `RolloutRecord` 回答的是：

> 这条轨迹是否具备 token、mask、reward、policy 等训练字段？

新 `AgentEpisode` 回答的是：

> 这个 Agent/Harness 在执行任务时发生了什么，哪些事实可观察，失败在哪里，能否与另一版本公平比较？

二者不能强行合并。训练需要紧凑的 token-level payload；Harness 分析需要 event、span、context、tool、sandbox、verifier、artifact 和 manifest。把后者全塞进 `RolloutRecord` 会让 universal contract 被 GRPO 语义绑死。

```text
TraceEvent → AgentEpisode → Analysis / Compare / Gate
                         └→ TrainingViewExporter → RolloutRecord
```

## 为什么先保存 TraceEvent

执行中会出现并发、重试、中断和乱序。若直接写一个最终 Episode：

- 进程崩溃时可能什么都不剩；
- 后写入的诊断可能覆盖原始事实；
- 无法验证 event 是否重复或丢失；
- 很难关联父子调用和 artifact。

因此先 append immutable event，再由 assembler 确定性生成 Episode。

## 为什么 capability 比“字段为 null”更重要

没有 Hook 时，系统可能知道工具调用结果，却不知道 Harness 为什么 retry 或 termination。`null` 无法区分：

- 事件真实发生但值为空；
- capture path 没启用；
- producer 不支持；
- 数据在传输中丢失。

Capability 明确本次 Episode 可支持哪些分析。缺少 `CONTEXT_COMPACTION` 时，系统不能诊断 compaction information loss，只能返回 `INSUFFICIENT_EVIDENCE`。

## 为什么 raw fact 和 diagnosis 分开

下面是事实：

```text
tool command exit_code=1
下一次 command 与上一次相同
最终 verifier 未通过
```

下面是派生判断：

```text
reason_code=TOOL_ERROR_FEEDBACK_LOSS
confidence=0.8
rule_version=tool-feedback/v1
```

规则以后可能升级，但历史事件不应改变。分离后可以用同一批 raw Episode 重跑不同诊断版本并比较结果。

## 为什么 Harness 对比需要 Manifest

如果 control 使用不同模型、不同 Sandbox image 或不同 verifier，success/token/latency 差异不能归因于 Harness。因此 comparison 首先检查：

```text
task + model + sampling + environment + tools + evaluator + seed
```

只有 Harness version/目标 policy 可以变化。无法控制时，最诚实的 Gate 结果是 `INSUFFICIENT_EVIDENCE`。

## 旧 contract 仍然解决什么

旧代码中的原则仍有效：

- sampled token ID 不能靠重新 tokenize 文本伪造；
- missing logprob 不能填零；
- batch capability 应取交集；
- verifier failure 与 verifier error 不同；
- source checksum 和 lineage 必须保留；
- infrastructure failure 不能冒充模型 reward 0。

这些约束现在属于 optional Training View，而不是 universal execution schema。

## 你需要能回答的问题

1. 为什么 `AgentEpisode` 不能取代 raw event store？
2. 为什么 `RolloutRecord` 不适合做 universal trace contract？
3. `NOT_OBSERVABLE` 与 `UNKNOWN` 有什么区别？
4. 为什么 diagnosis 必须保存 evidence IDs 和 rule version？
5. 为什么 infra-invalid 不应进入 task success rate 分母？
6. 什么情况下 Regression Gate 应返回证据不足？
