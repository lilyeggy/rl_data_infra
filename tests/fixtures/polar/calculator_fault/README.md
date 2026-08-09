# Calculator Synthetic Fault Fixture

状态：`CAPTURED`（run `20260809T164046Z-calculator`，polar f0e8343a，2026-08-09/10）

Manifest：`fixture_type=calculator_fault`，`synthetic_fault=true`。

- 注入点：`request.json $.runtime.prepare` 追加 `{"type":"exec","command":"echo SYNTHETIC_FAULT_INJECTED_AT_INIT && false"}`（prepare 第 5 条，索引 4）；同时 `$.evaluator.config.test_timeout` 60.0→0.1（记录于 fault-diff，因 agent 空 patch 使 evaluator 短路而不可观察）；
- 预期失败阶段：runtime INIT；实际 `summary.json $.status=ERROR`，`$.error="runtime initialization failed: prepare action 4 failed with exit code 1"`（4 个 session 均 ERROR，run_ms=0，record_count=0，无模型调用）；
- 非破坏性：未停止 SGLang/Gateway、未改模型与 runtime 镜像、未污染共享环境；
- 未保留 synthetic fault 的 evaluator/reward（INIT 阶段前失败），reward/old_logprobs 等按 missing 处理；
- 该故障是基础设施失败（INVALID_INFRASTRUCTURE），不得作为模型 reward=0 训练数据。

原始证据：`/data/day-01-workspace/artifacts/day-02/20260809T164046Z-calculator/raw/`（服务器本地，未提交 Git）；首次 test_timeout-only 尝试归档于 `raw/fault-test_timeout-attempt/`。
