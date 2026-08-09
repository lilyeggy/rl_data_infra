# 学习笔记 01：为什么先做 Data Contract

## 先看问题，不先看类名

Polar、另一个 rollout server，甚至一个离线 JSONL 文件都可以产生 Agent 轨迹，
但它们不会天然使用相同字段。Trainer 又不能仅凭一段文本安全训练：它需要知道
token、mask、reward、group 和 policy version 到底从哪里来。

因此我们的第一道边界不是“转换字段名”，而是回答两个问题：

1. 源系统实际提供了什么？
2. 下游准备做的操作需要什么？

`RolloutRecord` 回答第一个问题，`Capability` gate 回答第二个问题。

## 一条记录如何流动

假设 Polar 产生了一条成功轨迹：

```text
session artifact + Gateway completion
  ├─ native token IDs
  ├─ action/loss mask
  ├─ reward=1
  ├─ policy_version=v0
  └─ verifier passed
```

`PolarSourceAdapter` 将来只负责显式读取这些真实位置，得到：

```text
RolloutRecord(
  trajectory_id="...",
  task_id="...",
  group_id="...",
  policy_version="v0",
  token_ids=(...),
  loss_mask=(...),
  reward=1.0,
  verifier_status=PASSED,
)
```

如果 completion 中没有 sampled logprobs，正确结果是：

```text
old_logprobs=None
Capability.OLD_LOGPROBS 不存在
```

错误结果则是填一串 `0.0`。填零让数据看起来完整，却改变了训练目标的数学含义，
而且下游已经无法判断它来自模型还是来自 Adapter。

## 为什么 batch capability 是交集

假设一个 batch 有两条记录：

```text
record A: TOKEN_IDS, MASK, REWARD, POLICY_VERSION
record B: TOKEN_IDS, MASK, REWARD
```

这个 batch 只能保证：

```text
TOKEN_IDS, MASK, REWARD
```

如果取并集，Trainer 会以为每条记录都有 policy version，然后在处理 B 时才崩溃，
或更糟糕地把 B 混入错误 policy 的 group。取交集使失败发生在处理开始之前。

## 为什么 source envelope 会变化

Polar 记录导出为 JSONL 再读回来时，轨迹语义没有变化，但当前直接来源已经变成
JSONL 文件。因此 Adapter 会更新：

```text
source_type
source_record_id
source_payload_ref
source_payload_sha256
```

训练字段、状态和 opaque metadata 保持相同。这就是计划中“除 source envelope 外
等价”的具体含义。

## 为什么现在还不猜 Polar 映射

`PolarSourceAdapter` 必须依据 Day 2–3 的真实 fixture 写 JSON path。如果先根据文档
猜字段，代码很可能能通过我们自己编造的测试，却无法转换真实 artifact。

所以当前正确顺序是：

```text
先完成 producer-agnostic contract 和 JSONL Adapter
→ 取得真实 Polar fixture
→ 审计真实 JSON path
→ 实现显式 Polar mapping 和 golden tests
```

这不是拖延核心代码，而是在维护“不伪造源数据”这条边界。

## 真实 Day 2 fixture 带来的修正

服务器 artifact 显示 Polar trace 已经直接提供：

```text
prompt_ids
response_ids
loss_mask（与 response_ids 对齐）
response_logprobs（与 response_ids 对齐）
```

因此 Adapter 不需要 tokenizer，只进行可逆拼接：

```text
token_ids = prompt_ids + response_ids
full_loss_mask = [0] * len(prompt_ids) + loss_mask
prompt_token_count = len(prompt_ids)
```

`prompt_token_count` 是真实 fixture 迫使 contract 明确表达的边界：以后 Slime Adapter
需要 response length 时，可以从完整 tokens 中确定性得到，而不是用 loss mask 的和代替。
工具结果等非训练 token 可能 mask=0，但仍属于 response 段，两者不能混淆。

锁定 Polar `prefix_merging` builder 明确定义 `response_logprobs` 为 sampled response
logprobs，因此 Adapter 显式映射到 canonical `old_logprobs` 并记录原 JSON path。不过
当前 fixture 仍没有 `policy_version` 和 `group_id`，capability gate 会继续阻止训练。

真实数据还显示 `calculator_success` 实际 reward=0、resolved=false：这里的 success 是
rollout/evaluator 执行路径成功，不是模型任务成功。这说明读取 fixture 时必须把执行状态
和任务 outcome 分开，也是我们给 fixture verifier 增加 execution-path 语义门的原因。

## 你需要能回答的四个问题

1. 为什么缺失 `old_logprobs` 不能填零？
2. 为什么 batch capability 应该取交集而不是并集？
3. `verifier failed` 和 `verifier error` 为什么不能都当 reward 0？
4. 为什么 `TrainingReadyBatch` 不能混合两个 policy version？

如果能用自己的话回答这四点，你已经掌握了 Day 4 最重要的设计，而不是只记住
几个 dataclass 的字段名。
