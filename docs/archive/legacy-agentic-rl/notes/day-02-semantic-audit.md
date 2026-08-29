# Day 2 Calculator Fixture 语义审计

审计对象：服务器提交 `cb456c02960cc1dc0a91e09b97ee675ae6036620`。

## 完整性结论

两个 fixture 的 manifest、文件大小、SHA256、路径、仓库大小和明显敏感字段检查均通过。
Fault fixture 是可用的 synthetic infrastructure failure：runtime prepare 明确失败，
session status 为 `ERROR`，没有模型 completion，也没有把错误映射为 reward 0。

## 命名与 outcome 语义

`tests/fixtures/polar/calculator_success/` 并不是任务成功：

```text
summary.status = COMPLETED
evaluation.outcome_reward = 0.0
evaluation.report.resolved = false
evaluation.report.empty_generation = true
trace_count = 1
```

模型只生成第一次文本形式的 tool call，`qwen_code` 随后退出；没有 tool result 或
patch。因此它是一条“执行链正常结束的 VALID_FAILURE”。Day 2 runbook 中
`calculator_success` 指 rollout/evaluator 路径成功完成，并不保证模型 reward=1；目录名
容易误读，使用时必须同时查看 reward/resolved。

## 处理决定

- 保留服务器原始提交，不改写证据；
- 验证器增加 Calculator execution-path 语义检查：必须 `COMPLETED`、有 evaluator report、
  evaluator 没有 crash/timeout；不要求 reward=1；
- Day 2 状态接受为 `COMPLETED_WITH_NOTES`；
- Day 3 可以进入，但仍必须取得真正的 Coding `VALID_SUCCESS`；
- `qwen_code` 单轮退出作为已知限制继续定位，不能把本 fixture 描述成模型任务成功。

## 对 Data Contract 的影响

Pinned Polar builder 的 `response_logprobs` 是采样策略在 response token 上的原生 logprob，
`PolarSourceAdapter` 将其映射为 canonical `old_logprobs`，同时保留原 JSON path。服务端
字段表所说“没有字面名为 old_logprobs 的字段”仍是事实，但不再等于该语义不可获得。

该映射不消除另一个阻塞：fixture 没有 `policy_version` 和 `group_id`，所以 training
capability gate 仍会拒绝它进入 `TrainingReadyBatch`。
