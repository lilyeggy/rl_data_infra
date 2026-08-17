# Multi-Harness Comparison Task Pilot Report

> 状态：等待选择并运行 reference tasks。
> 目的：选择能稳定暴露 Harness 行为差异的任务，不用于模型训练选样。

## 候选任务

| Task ID | Environment revision | Verifier | Target Harness behavior | Deterministic | Exclusion reason |
|---|---|---|---|---|---|
| TBD | TBD | TBD | tool error recovery | TBD | |
| TBD | TBD | TBD | verification before finish | TBD | |
| TBD | TBD | TBD | loop/retry control | TBD | |

## 控制变量

```text
model/provider/revision：
sampling config：
sandbox image/runtime：
tool schema digest：
evaluator/verifier revision：
seed/timeout：
唯一 candidate variable：
```

## Pilot 统计

| Task ID | Harness/version | Episodes | Success | Valid failure | Infra invalid | Mean turns/tools | Mean tokens | Wall time |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TBD | control | 0 | 0 | 0 | 0 | TBD | TBD | TBD |
| TBD | candidate | 0 | 0 | 0 | 0 | TBD | TBD | TBD |

## Capture coverage

| Capability | Control | Candidate | Evidence |
|---|---|---|---|
| MODEL_IO | TBD | TBD | TBD |
| TOOL_IO | TBD | TBD | TBD |
| SANDBOX_COMMAND_IO | TBD | TBD | TBD |
| VERIFIER_EVIDENCE | TBD | TBD | TBD |
| HARNESS_DECISIONS | TBD | TBD | TBD |
| CONTEXT_COMPACTION | TBD | TBD | TBD |

## Infrastructure faults

| Fault | Natural/synthetic | Injection path | Expected validity | Evidence SHA256 |
|---|---|---|---|---|
| verifier timeout | synthetic | verifier timeout config | INFRA_INVALID | TBD |
| sandbox prepare failure | synthetic | environment prepare | INFRA_INVALID | TBD |
| partial capture | synthetic | terminate writer before finish | PARTIAL | TBD |

## Reference improvement selection

```text
选择的 Harness 问题：
control 行为：
candidate 修改：
目标 metric/failure slice：
允许的 token/latency 代价：
Gate thresholds：
选择原因：
排除任务：
仍需解决的问题：
```
