# Coding INVALID_INFRASTRUCTURE Fixture

状态：`CAPTURED`（run `20260811T030617Z-swebench`，polar f0e8343a，2026-08-11）

- manifest：`fixture_type=coding_invalid_infra`，`synthetic_fault=true`，`resolved=null`、`reward=null`（非 0）
- 注入点：`request.json $.runtime.prepare` 追加 `exit 42`（唯一变更；payload SHA256 原/改见 fault-injection.json）
- 预期/实际失败阶段：runtime INIT；实际 `summary.json $.status=ERROR`、`$.error="runtime initialization failed: prepare action 1 failed with exit code 42"`，run_ms=0、无模型调用
- 非破坏性：未停服务、未改镜像/Docker daemon/共享目录
- 第二种 fault（verifier timeout，`evaluator.config.test_timeout=0.1`）已记录于服务器 raw evidence：因 agent 空 patch 使 swebench_harness 短路而不可观察（`report.test_timeout=false`）
- 该故障是基础设施失败（INVALID_INFRASTRUCTURE），不得作为模型 reward=0 训练数据

原始证据：`/data/day-01-workspace/artifacts/day-03/20260811T030617Z-swebench/raw/pytest-dev--pytest-5809-fault-runtime-prepare/`（服务器本地）。
