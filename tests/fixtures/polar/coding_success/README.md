# Coding VALID_SUCCESS Fixture

状态：`CAPTURED`（run `20260811T030617Z-swebench`，polar f0e8343a，2026-08-11）

- instance：pytest-dev__pytest-5809（base `8aba863a…`），harness `qwen_code`，evaluator `swebench_harness`
- **前置**：SGLang v0.5.13 本地补丁 `patches/sglang/qwen3-tool-call-fix.patch`（sha256 `136fb0af…`）——修复思考块内工具调用剥离 + JSON 风格 tool_call 解析
- 真实多轮 rollout：10 轮模型调用（read_file/grep/edit/write_file/run_shell），10 个 Gateway completion；原生 token_ids/logprobs 已核实
- outcome：`resolved=true`、`reward=1.0`；verifier FAIL_TO_PASS `test_create_new_paste` 通过 + 3 个 PASS_TO_PASS 无回归（exit_code 0）
- patch.diff：从真实工具调用参数重建（evaluator 原始 patch 随 session 目录清理）；clean replay 在 fresh runtime 上 4 passed、resolved=true（replay.json）
- 缺失（不得猜测）：policy_version、group_id、old_logprobs

原始证据：`/data/day-01-workspace/artifacts/day-03/20260811T030617Z-swebench/raw/`（服务器本地，未提交 Git）。
