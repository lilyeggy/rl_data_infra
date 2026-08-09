# Polar Raw Artifact 字段审计表

状态：已按 Day 2 真实 Calculator artifact 审计（run `20260809T164046Z-calculator`，polar f0e8343a，2026-08-09/10）。

## 记录规则

- `source_file` 和 `json_path/log_location` 必须能直接定位证据；
- `provenance` 只能是 `direct`、`derived` 或 `missing`；
- `derived` 必须写清算法，且不能用于伪造训练字段；
- 未观察到的字段写 `missing`，不能留成含糊的“可能有”；
- Polar 特有对象只记录在 source mapping，不提前进入 canonical contract。

source 文件约定：`<F>` = `tests/fixtures/polar/calculator_success/`，`<FF>` = `tests/fixtures/polar/calculator_fault/`；raw 路径见 `<run-dir>/raw/` 与 `staging-provenance.json`。

## Identity

| Logical field | Source file | JSON path / log location | Observed type | Provenance | Notes |
|---|---|---|---|---|---|
| task_id | `<F>summary.json` | `$.task_id` | string | direct | `calculator-qwen_code-20260809T165213Z`；fault `<FF>request.json $.task_id` 带 `-fault` 后缀 |
| rollout_id | — | — | — | missing | Polar 未暴露独立 rollout_id 字段 |
| session_id | `<F>summary.json` | `$.session_id` | string | direct | `sk-polar-<uuid>`，opaque metadata |
| request_id / completion_id | raw completion record | `$.completion_id` | string | direct | `msg_ff634f2fb0a0`；fixture `response.json $.id` 是 backend response id，非 completion_id |
| group_id | — | — | — | missing | 不得从目录名猜测 |
| policy_version | — | — | — | missing | 不得从时间戳/文件名猜测 |

## Model and token provenance

| Logical field | Source file | JSON path / log location | Observed type | Provenance | Notes |
|---|---|---|---|---|---|
| model_id (requested) | `<F>request.json` / `<F>summary.json` | `$.model` / `$.trajectory.metadata.model_requested` | string | direct | `qwen3-coder-plus`（qwen-code CLI 请求） |
| model_id (used) | `<F>response.json` / `<F>summary.json` | `$.model` / `$.trajectory.metadata.model_used` | string | direct | `Qwen/Qwen3-4B-Instruct-2507`（Gateway 重写） |
| model_revision | manifest `source.model_revision` | — | string | derived | `cdbee75f…`；来自 upstream-lock，artifact 内无 revision 字段；response.metadata.weight_version=default |
| tokenizer_revision | manifest `source.tokenizer_revision` | — | string | derived | 同 model_revision snapshot |
| prompt_token_ids | `<F>summary.json` / `<F>response.json` | `$.trajectory.traces[0].prompt_ids` / `$.choices[0].input_token_ids` | int[] (14846) | direct | 来自 Gateway completion record 原生字段 |
| output_token_ids | `<F>response.json` / `<F>summary.json` | `$.choices[0].token_ids` / `$.trajectory.traces[0].response_ids` | int[] (99) | direct | **原生数值 token IDs**，非重新 tokenize |
| sampled_logprobs | `<F>response.json` / `<F>summary.json` | `$.choices[0].logprobs.content` / `$.trajectory.traces[0].response_logprobs` | obj[]/float[] (99) | direct | 每条含 token/token_id/logprob |
| loss_mask | `<F>summary.json` | `$.trajectory.traces[0].loss_mask` | int[] (99) | direct | 与 response_ids 等长 |
| old_logprobs | — | — | — | missing | 训练语义（采样时 policy 的 logprob）在源中无此字段 |

## Harness and tool execution

| Logical field | Source file | JSON path / log location | Observed type | Provenance | Notes |
|---|---|---|---|---|---|
| harness | `<FF>request.json` / manifest | `$.agent.harness` | string | direct | `qwen_code`（@qwen-code/qwen-code@0.14.5） |
| tool_call | `<F>summary.json` | `$.trajectory.traces[0].response_messages[0].reasoning_content` | string | direct | 文本 `<tool_call>` 内含 read_file；结构化 `response.json $.choices[0].message.tool_calls` = `[]`（SGLang 未填结构化字段） |
| tool_result | — | — | — | missing | agent 在首次 tool_call 后退出，无工具结果轮次 |
| file_patch | `<F>summary.json` | `$.trajectory.metadata.evaluation.report.empty_generation` | bool | derived | empty_generation=true ⇒ patch 为空，无 file_patch |
| final_output | `<F>summary.json` | `$.trajectory.traces[0].response_messages[0].content` | string (空) | direct | agent 未完成，content 空 |

## Lifecycle and failure state

| Logical field | Source file | JSON path / log location | Observed type | Provenance | Notes |
|---|---|---|---|---|---|
| rollout_status | `<F>summary.json` | `$.status` | string | direct | `COMPLETED`（success）；fault `ERROR` |
| termination_reason | — | — | — | missing | 无独立字段；模型侧 finish_reason 见下 |
| finish_reason (model) | `<F>response.json` | `$.choices[0].finish_reason` | string | direct | `stop` |
| runtime_status | `<FF>summary.json` | `$.error` | string | direct | fault: `runtime initialization failed: prepare action 4 failed with exit code 1`（INIT 失败，run_ms=0, record_count=0） |
| harness_status | — | — | — | missing | 无独立字段 |
| model_backend_status | — | — | — | missing | 无独立字段；由 completion 存在与否推断 |
| verifier_status | — | — | — | missing | Calculator 用 evaluator（test_on_output），无 verifier |
| timeout/error | `<F>summary.json` | `$.trajectory.metadata.evaluation.report.test_timeout` | bool | direct | success: false；fault 无 evaluator 执行 |

## Evaluator and reward

| Logical field | Source file | JSON path / log location | Observed type | Provenance | Notes |
|---|---|---|---|---|---|
| evaluator_strategy | `<F>summary.json` | `$.trajectory.metadata.evaluation.strategy` | string | direct | `test_on_output` |
| evaluator_command | `<FF>request.json` | `$.evaluator.config.test_command` | string | direct | success fixture 的 request.json 是 completion request（无 evaluator）；test_command 见 fault task payload 同款配置（`python3 test_calculator.py && echo 'PASSED test_calculator'`） |
| evaluator_exit_code | — | — | — | missing | 源无显式 exit code；report 用布尔标志 |
| evaluator_stdout/stderr | — | — | — | missing | 未入 fixture；原始 evaluator 输出在 runtime 日志（未保留） |
| reward | `<F>summary.json` | `$.trajectory.metadata.evaluation.outcome_reward` | float | direct | success: 0.0（模型未解出，evaluator 正常执行）；fault 无 evaluator ⇒ missing |
| verifier_evidence_ref | — | — | — | missing | 无 verifier |

## Day 2 审计结论

```text
源系统直接存在的训练字段：
  prompt_token_ids（$.trajectory.traces[0].prompt_ids / response $.choices[0].input_token_ids）
  output_token_ids（$.trajectory.traces[0].response_ids / response $.choices[0].token_ids）
  sampled_logprobs（$.trajectory.traces[0].response_logprobs / response $.choices[0].logprobs.content）
  loss_mask、finish_reason、reward（evaluation.outcome_reward）、task_id、session_id

明确缺失的训练字段：
  policy_version、group_id、old_logprobs（训练语义）

只能作为 opaque metadata 的字段：
  session_id（sk-polar-<uuid>）、completion_id（msg_*）、task_id

成功与 task failure 的区分依据：
  summary.status=COMPLETED 且 evaluation.report（empty_generation/resolved/failed_apply_patch/error_eval/test_timeout）；
  outcome_reward=0.0 是模型未解出（VALID_FAILURE 语义），不是基础设施失败。

infrastructure failure 的可观察依据：
  summary.status=ERROR 且 $.error 含 "runtime initialization failed: prepare action N failed with exit code M"
  （INIT 阶段失败；run_ms=0、record_count=0、无 completion——synthetic_fault fixture）

进入 Day 4 contract 前仍需回答的问题：
  1. policy_version 在 Polar 侧无字段；adapter 需显式声明或由上游提供，不得从文件名/时间戳推断。
  2. old_logprobs 训练语义缺失；GRPO 是否可由当前采样 logprob + loss_mask 重建需 Day 6 验证。
  3. qwen-code CLI 单轮 tool_call 后静默退出（run 阶段 2.7s，1 completion）；原因未定位，可能为 CLI 与
     模型工具调用格式或运行时网络交互问题。记录为 known limitation。
  4. SGLang response 结构化 message.tool_calls 为空，工具调用只出现在 reasoning_content 文本中；
     提取 tool events 需定义文本解析规则（Day 4/5 范围）。
  5. 本地镜像 polar-localhost-calculator:latest 以 image ID 标识，干净复现需 registry RepoDigest 或 tar checksum。
```
