# Coding VALID_FAILURE Fixture

状态：`CAPTURED`（run `20260811T030617Z-swebench`，polar f0e8343a，2026-08-11）

- instance：pytest-dev__pytest-5809（base `8aba863a…`），harness `qwen_code`，evaluator `swebench_harness`
- 真实模型 rollout：1 次模型请求（input 14907 / output 191 token，原生 token_ids + logprobs 已核实）
- outcome：session COMPLETED、`empty_generation=true`（agent 单轮退出，未产生 patch）、resolved=false、reward=0.0
- verifier：evaluator 正常完成但因空 patch 短路，测试命令未实际执行（note 已写入 verifier-evidence.json）
- clean replay：fresh runtime + base commit + 空 patch → verifier exit 4、resolved=false、与原一致（replay.json）
- 缺失（不得猜测）：policy_version、group_id、old_logprobs、tool_results、file_patch

原始证据：`/data/day-01-workspace/artifacts/day-03/20260811T030617Z-swebench/raw/`（服务器本地，未提交 Git）。
