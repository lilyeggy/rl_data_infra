# Polar Coding Artifact 字段审计表

状态：等待 Day 3 真实 artifact。未观察到的字段必须写 `missing`，禁止从文本或文件名猜测。

| Logical field | Source file | JSON path / log location | Observed type | Provenance | Notes |
|---|---|---|---|---|---|
| task_id | TBD | TBD | TBD | missing | |
| session_id | TBD | TBD | TBD | missing | |
| request_id | TBD | TBD | TBD | missing | |
| trajectory_id | TBD | TBD | TBD | missing | |
| group_id | TBD | TBD | TBD | missing | 不从 task/session ID 推断 |
| policy_version | TBD | TBD | TBD | missing | 不从时间戳推断 |
| model_id | TBD | TBD | TBD | missing | |
| model_revision | TBD | TBD | TBD | missing | |
| tokenizer_revision | TBD | TBD | TBD | missing | |
| prompt_token_ids | TBD | TBD | TBD | missing | 禁止重新 tokenize |
| output_token_ids | TBD | TBD | TBD | missing | 禁止重新 tokenize |
| action/loss_mask | TBD | TBD | TBD | missing | |
| sampled_logprobs | TBD | TBD | TBD | missing | 禁止填零 |
| tool_calls | TBD | TBD | TBD | missing | |
| tool_results | TBD | TBD | TBD | missing | |
| file_patch | TBD | TBD | TBD | missing | 记录 patch SHA256 |
| rollout_status | TBD | TBD | TBD | missing | |
| termination_reason | TBD | TBD | TBD | missing | |
| runtime_status | TBD | TBD | TBD | missing | |
| harness_status | TBD | TBD | TBD | missing | |
| model_backend_status | TBD | TBD | TBD | missing | |
| verifier_status | TBD | TBD | TBD | missing | 区分 failed/error/timeout |
| verifier_command | TBD | TBD | TBD | missing | |
| verifier_exit_code | TBD | TBD | TBD | missing | |
| verifier_resolved | TBD | TBD | TBD | missing | |
| reward | TBD | TBD | TBD | missing | infra failure 必须为 null |
| runtime_image_identity | TBD | TBD | TBD | missing | |
| base_commit | TBD | TBD | TBD | missing | |

## 审计结论

```text
源系统直接存在的训练字段：
明确缺失的训练字段：
success 与 valid failure 的证据边界：
invalid infrastructure 的证据边界：
只能作为 opaque metadata 的 Polar 字段：
进入 PolarSourceAdapter 前仍需回答的问题：
```
