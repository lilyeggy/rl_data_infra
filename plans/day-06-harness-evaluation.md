# Day 6：Harness Evaluation、Failure Attribution 与 Regression Gate

> 状态：已完成；真实 Pi reference case 的 Gate 为 `REJECT`（效率超预算）。
> 目标：把 Episode 变成能够支持 Harness 改进决策的证据，而不是只收集日志。

## 1. 阶段目标

1. 实现统一 Episode metrics；
2. 实现带 evidence 的 failure attribution；
3. 实现 manifest compatibility check；
4. 实现同任务 control/candidate paired comparison；
5. 实现三态 Regression Gate；
6. 从真实 trace 中选择一个 reference improvement case。

## 2. Metrics

第一版只实现能回答 Harness 行为差异的指标：

```text
success_rate
infra_invalid_rate
turn_count
tool_call_count
duplicate_action_rate
loop_count
tool_error_recovery_rate
verification_attempts
premature_termination_count
input/output_tokens
estimated_cost
model/tool/sandbox/verifier_latency
wall_time
```

Context/compaction 指标只在相应 capability 存在时计算。

每个指标定义必须写清：输入事件、分母、缺失值、infra-invalid 是否排除、metric version。

## 3. Failure Attribution

一级层：

```text
MODEL / HARNESS / SANDBOX / MODEL_BACKEND /
EVALUATOR / EXTERNAL_SERVICE / UNKNOWN
```

优先实现与 reference case 直接相关的 Harness reason code；不要求一次实现全部枚举。每条 Diagnosis 必须包含：

- reason code；
- evidence event/artifact IDs；
- confidence；
- rule version；
- 简短 explanation；
- 输入 Episode checksum。

规则无法区分时输出 multi-label、UNKNOWN 或 INSUFFICIENT_EVIDENCE，不强行单因归责。

## 4. 可比性检查

比较前固定：

```text
task/dataset revision
model/provider/revision
sampling params
sandbox image and limits
tool schema
evaluator/verifier
seed and timeout
```

只允许 Harness version 或目标 policy flag 变化。检查失败时保存 mismatch 列表，Gate 默认为 `INSUFFICIENT_EVIDENCE`。

## 5. Paired Comparison

以 `task_id + seed + attempt` 对齐 control/candidate，输出：

- 每对 Episode outcome 和行为差异；
- 成功、失败、infra-invalid 的迁移；
- token/cost/latency 差异；
- tool recovery、loop、duplicate action、verifier 行为差异；
- failure slice 聚合；
- unmatched Episode；
- 样本量和 paired coverage。

样本允许较小，但必须展示逐任务结果，不能把机制演示表述成统计普遍提升。

## 6. Regression Gate

示例配置：

```yaml
minimum_paired_coverage: 1.0
minimum_episode_pairs: 3
success_rate_drop_tolerance: 0.0
target_slice: TOOL_ERROR_FEEDBACK_LOSS
require_target_slice_improvement: true
max_token_increase_ratio: 0.20
max_latency_increase_ratio: 0.25
max_infra_invalid_increase: 0.0
reject_new_severe_regression: true
```

输出：

```text
ACCEPT
REJECT
INSUFFICIENT_EVIDENCE
```

GateResult 保存每条规则的 actual、threshold、pass/fail、evidence 和版本。

## 7. Reference improvement case

默认选择 Structured Tool Error Feedback：

```text
v1: raw stderr
v2: error_type + command + exit_code + stderr_summary + retryable
```

选择依据：容易构造确定性 task，能观察错误恢复、重复命令、turn、token、latency 和最终 outcome，也属于真实 Harness 控制层策略。

若现有 Harness 不便修改，使用 Verification Completion Gate 作为备选。只做一个完整案例优于多个半成品。

## 8. 测试

- metric denominator 与缺失 capability；
- infra-invalid exclusion；
- tool error recovery 正/反例；
- duplicate/loop detection；
- attribution evidence 完整性；
- manifest match/mismatch；
- paired alignment 与 unmatched case；
- Gate threshold 边界；
- ACCEPT/REJECT/INSUFFICIENT_EVIDENCE 三态。

## 9. 验收门

- [ ] 指标能从 canonical Episode 独立计算；
- [ ] Diagnosis 可追溯到具体 event/artifact；
- [ ] 不可观测行为不会得到伪诊断；
- [ ] 不可比运行不会产生强 A/B 结论；
- [ ] paired comparison 同时展示效果与成本；
- [ ] Gate 三态均有测试；
- [ ] 已从 trace 选择并冻结一个 v1/v2 改进案例；
- [ ] CLI/JSON 输出足以在没有 UI 时完成审计。

## 10. 阶段产物

```text
src/analysis/metrics.py
src/analysis/attribution.py
src/analysis/compare.py
src/analysis/regression_gate.py
configs/gates/reference.yaml
tests/analysis/
artifacts/day-06/
docs/experiment-protocol.md
```

## 11. 执行记录

```text
状态：NOT_STARTED
metric版本：
归因规则版本：
paired episodes：
reference case：
Gate结果：
限制：
```
