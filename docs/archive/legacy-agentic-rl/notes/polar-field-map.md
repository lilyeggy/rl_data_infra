# Polar Raw Artifact 字段审计表

状态：等待 Day 2 Calculator 真实 artifact。本文档只定义审计方法，不提前宣称字段存在。

## 记录规则

- `source_file` 和 `json_path/log_location` 必须能直接定位证据；
- `provenance` 只能是 `direct`、`derived` 或 `missing`；
- `derived` 必须写清算法，且不能用于伪造训练字段；
- 未观察到的字段写 `missing`，不能留成含糊的“可能有”；
- Polar 特有对象只记录在 source mapping，不提前进入 canonical contract。

## Identity

| Logical field | Source file | JSON path / log location | Observed type | Provenance | Notes |
|---|---|---|---|---|---|
| task_id | TBD | TBD | TBD | missing | |
| rollout_id | TBD | TBD | TBD | missing | |
| session_id | TBD | TBD | TBD | missing | |
| request_id | TBD | TBD | TBD | missing | |
| group_id | TBD | TBD | TBD | missing | 不得从目录名猜测 |
| policy_version | TBD | TBD | TBD | missing | 不得从时间戳猜测 |

## Model and token provenance

| Logical field | Source file | JSON path / log location | Observed type | Provenance | Notes |
|---|---|---|---|---|---|
| model_id | TBD | TBD | TBD | missing | |
| model_revision | TBD | TBD | TBD | missing | |
| tokenizer_revision | TBD | TBD | TBD | missing | |
| prompt_token_ids | TBD | TBD | TBD | missing | 必须来自源 trace |
| output_token_ids | TBD | TBD | TBD | missing | 禁止重新 tokenize 文本 |
| sampled_logprobs | TBD | TBD | TBD | missing | 禁止填零 |

## Harness and tool execution

| Logical field | Source file | JSON path / log location | Observed type | Provenance | Notes |
|---|---|---|---|---|---|
| harness | TBD | TBD | TBD | missing | |
| tool_call | TBD | TBD | TBD | missing | |
| tool_result | TBD | TBD | TBD | missing | |
| file_patch | TBD | TBD | TBD | missing | |
| final_output | TBD | TBD | TBD | missing | |

## Lifecycle and failure state

| Logical field | Source file | JSON path / log location | Observed type | Provenance | Notes |
|---|---|---|---|---|---|
| rollout_status | TBD | TBD | TBD | missing | |
| termination_reason | TBD | TBD | TBD | missing | |
| runtime_status | TBD | TBD | TBD | missing | |
| harness_status | TBD | TBD | TBD | missing | |
| model_backend_status | TBD | TBD | TBD | missing | |
| verifier_status | TBD | TBD | TBD | missing | |
| timeout/error | TBD | TBD | TBD | missing | |

## Evaluator and reward

| Logical field | Source file | JSON path / log location | Observed type | Provenance | Notes |
|---|---|---|---|---|---|
| evaluator_command | TBD | TBD | TBD | missing | |
| evaluator_exit_code | TBD | TBD | TBD | missing | |
| evaluator_stdout | TBD | TBD | TBD | missing | |
| evaluator_stderr | TBD | TBD | TBD | missing | |
| reward | TBD | TBD | TBD | missing | 不得从 success 文本猜测 |
| verifier_evidence_ref | TBD | TBD | TBD | missing | |

## Day 2 审计结论模板

```text
源系统直接存在的训练字段：
明确缺失的训练字段：
只能作为 opaque metadata 的字段：
成功与 task failure 的区分依据：
infrastructure failure 的可观察依据：
进入 Day 4 contract 前仍需回答的问题：
```
