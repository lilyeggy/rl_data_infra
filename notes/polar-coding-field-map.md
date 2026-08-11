# Polar Coding Artifact 字段审计表

状态：已按 Day 3 真实 SWE-bench rollout 审计（run `20260811T030617Z-swebench`，polar f0e8343a，2026-08-11）。

规则：`source_file`/`json_path` 可定位证据；`provenance` 只能是 direct/derived/missing；未观察写 missing。

source 约定：`<F>` = `tests/fixtures/polar/coding_valid_failure/`，`<II>` = `tests/fixtures/polar/coding_invalid_infra/`；raw 见 `<run-dir>/raw/`。实例 pytest-dev__pytest-5809。

| Logical field | Source file | JSON path / log location | Observed type | Provenance | Notes |
|---|---|---|---|---|---|
| task_id | `<F>summary.json` | `$.task_id` | string | direct | `swebench-qwen_code-pytest-dev-pytest-5809-20260811T041305Z` |
| session_id | `<F>summary.json` | `$.session_id` | string | direct | `sk-polar-caf2bff2-…`，opaque |
| request_id / completion_id | raw completion record | `$.completion_id` | string | direct | `msg_b3ffe0123480`；fixture `response.json $.id` 为 backend id |
| trajectory_id | — | — | — | missing | Polar 无独立 trajectory_id |
| group_id | — | — | — | missing | 不从 task/session ID 推断 |
| policy_version | — | — | — | missing | 不从时间戳推断 |
| model_id (requested/used) | `<F>response.json` / `<F>summary.json` | `$.model` / `$.trajectory.metadata.model_used` | string | direct | `Qwen/Qwen3-4B-Instruct-2507`（本 run 提交 model_name 即锁定模型，Gateway 无重写） |
| model_revision | manifest `source.model_revision` | — | string | derived | `cdbee75f…`；来自 upstream-lock |
| tokenizer_revision | manifest `source.tokenizer_revision` | — | string | derived | 同 model snapshot |
| prompt_token_ids | `<F>response.json` / `<F>summary.json` | `$.choices[0].input_token_ids` / `$.trajectory.traces[0].prompt_ids` | int[] (14907) | direct | 原生数值，非重新 tokenize |
| output_token_ids | `<F>response.json` / `<F>summary.json` | `$.choices[0].token_ids` / `$.trajectory.traces[0].response_ids` | int[] (191) | direct | 原生数值 |
| action/loss_mask | `<F>summary.json` | `$.trajectory.traces[0].loss_mask` | int[] (191) | direct | 与 response_ids 等长 |
| sampled_logprobs | `<F>response.json` / `<F>summary.json` | `$.choices[0].logprobs.content` / `$.trajectory.traces[0].response_logprobs` | obj[]/float[] (191) | direct | 非填零 |
| tool_calls | `<F>summary.json` | `$.trajectory.traces[0].response_messages[0].reasoning_content` | string | direct | 文本 `<tool_call>`；结构化 `$.choices[0].message.tool_calls` 为空（SGLang 未填） |
| tool_results | — | — | — | missing | agent 单轮退出，无工具执行轮 |
| file_patch | `<F>patch.diff` | 空文件 | bytes(0) | direct | `empty_generation=true`；patch_sha256=`e3b0c442…`（空文件） |
| rollout_status | `<F>summary.json` | `$.status` | string | direct | `COMPLETED`（valid failure）；`<II>` `ERROR` |
| termination_reason | — | — | — | missing | 无独立字段；模型侧 finish_reason=stop |
| runtime_status | `<II>summary.json` | `$.error` | string | direct | `runtime initialization failed: prepare action 1 failed with exit code 42`（INIT 失败，run_ms=0） |
| harness_status | — | — | — | missing | 无独立字段 |
| model_backend_status | — | — | — | missing | 无独立字段 |
| verifier_status | `<F>verifier-evidence.json` | `$.result.*` | bool | derived | verifier_completed=true；timed_out=false；crashed=false；note: empty_generation 短路，测试未实际执行 |
| verifier_command | `<F>verifier-evidence.json` | `$.test.command` | string | direct | pytest test_create_new_paste（来自 evaluator config） |
| verifier_exit_code | `<F>replay.json` | `$.exit_code` | int | direct | 4（clean replay 实测） |
| verifier_resolved | `<F>verifier-evidence.json` | `$.result.resolved` | bool | derived | false（来自 evaluation.report.resolved） |
| reward | `<F>summary.json` | `$.trajectory.metadata.evaluation.outcome_reward` | float | direct | valid failure: 0.0；invalid: **null**（`<II>verifier-evidence.json $.result.reward=null`） |
| runtime_image_identity | manifest `source.runtime_image_identity` | — | string | derived | `polar-swebench-runtime:pytest-dev-pytest-5809`（本地镜像 ID `0d14b09e…`） |
| base_commit | `<F>verifier-evidence.json` | `$.base_commit` | string | direct | `8aba863a634f40560e25055d179220f0eefabe9a`（pytest-5809，来自 dataset） |

## 审计结论

```text
源系统直接存在的训练字段：
  prompt_token_ids（input_token_ids/prompt_ids）、output_token_ids（token_ids/response_ids）、
  sampled_logprobs（logprobs.content/response_logprobs）、loss_mask、reward（success=1.0/failure=0.0）、
  task_id、session_id、model_used、tool_calls（多轮：todo_write/read_file/grep/edit/write_file/run_shell）、
  patch（重建自工具调用，patch_sha256=f80ba962…，replay 4 passed）

明确缺失的训练字段：
  policy_version、group_id、old_logprobs、trajectory_id

success 与 valid failure 的证据边界：
  success（coding_success）：verifier grading_report.patch_exists=true、patch_successfully_applied=true、
  FAIL_TO_PASS success、PASS_TO_PASS 无回归、exit_code=0、resolved=true、reward=1.0；clean replay 4 passed。
  valid failure：summary.status=COMPLETED + evaluation.report（resolved=false, error_eval=false）+ clean replay
  一致（exit 4, resolved=false）。注意补丁前 failure 均 empty_generation（evaluator 短路，测试未实际执行）。

invalid infrastructure 的证据边界：
  summary.status=ERROR + $.error="runtime initialization failed: prepare action N failed with exit code M"；
  run_ms=0、record_count=0、无模型调用；verifier-evidence result.resolved=null、reward=null。

只能作为 opaque metadata 的 Polar 字段：
  session_id（sk-polar-<uuid>）、completion_id（msg_*）、task_id

进入 PolarSourceAdapter 前仍需回答的问题：
  1. policy_version 无来源；adapter 需显式声明或由上游提供。
  2. old_logprobs 训练语义缺失（源仅有当前采样 logprobs）。
  3. SGLang v0.5.13 需本地补丁 patches/sglang/qwen3-tool-call-fix.patch 才能正确剥离思考块内
     <tool_call> 并解析 JSON 风格工具调用；该补丁是 Day 3 后多轮 rollout 的前置条件。
  4. swebench_harness 对空 patch 短路，test_timeout 注入不可观察（Day 2 同款行为）。
  5. evaluator 原始 patch 随 session 目录清理；本项目以工具调用参数重建（rebuild_patch.py），
     若需原生 patch 保留，Polar 侧需调整 artifacts 保留策略。
  6. 最后一条超大 edit 的流式结构化 tool_calls 为空（content 文本携带，qwen-code 自解析成功）——
     SGLang 流式 JSON 解析对超长 arguments 仍不完整。
```

## 转换规则（derived 字段）

- model_revision / tokenizer_revision / runtime_image_identity：来自 upstream-lock / 打包参数（manifest），非 artifact 内字段。
- verifier_resolved / reward（valid failure）：直接读取 `evaluation.report.resolved` 与 `evaluation.outcome_reward`（derived-from-summary）。
- 无效 infra 的 resolved/reward：恒为 null（不映射 0）。
