# V2 Release：Real Pi Harness Decision Loop

> Release：`v2-harness-decision`
> 日期：2026-08-17
> 状态：完整证据链已实现；当前 candidate 的 Gate 结论为 `REJECT`

## 本版回答的问题

> 在模型、任务、环境和 Verifier 不变时，只修改 Harness 的错误恢复策略，我们能否捕获真实执行、进行成对比较，并给出一项可审计的上线决策？

答案是：能。V2 使用真实 Pi 0.84.2 和固定的 `opencode-go/gpt-5.6-luna` 跑了 3 组 control/candidate；不是用 mock 输出替代真实 Harness。

## 实验结果

| 指标 | Control | Candidate | 结论 |
|---|---:|---:|---|
| exact-verifier success | 0/3 | 3/3 | 行为正确性明显改善 |
| `OBSERVED_TOOL_ERROR_LOOP` | 3 | 0 | 重复错误动作被消除 |
| mean tokens | 1647.67 | 2365.00 | +43.54%，超过 +20% 阈值 |
| mean duration | 6354 ms | 9604 ms | +51.15%，超过 +25% 阈值 |
| infra-invalid | 0 | 0 | 无基础设施退化 |

最终 Gate 为 `REJECT`。这不表示 candidate 没有效果，而是表示它虽然把成功率从 0% 提到 100%，但未通过预先固定的 token 和 latency 预算。系统没有为了得到好看的 `ACCEPT` 而事后放宽阈值。

## 真实实验设计

- 三个非敏感 synthetic filesystem task，已知 FAILURE 记录数分别为 2、2、3；
- 每组的用户任务相同；
- Control policy：第一次 `FILE_NOT_FOUND` 后原样再读一次，然后终止；
- Candidate policy：第一次 `FILE_NOT_FOUND` 后使用 `find` 发现正确文件，再读取并回答；
- 模型、provider、Pi 版本、thinking level、工具集合、Evaluator 和 task snapshot 固定；
- seed 在该 API 中不可观测，显式记录为 `NOT_OBSERVABLE`，不伪造数值。

## 实现清单

| 层 | V2 实现 |
|---|---|
| Real Harness Capture | Pi JSON runner、NDJSON parser、semantic backend failure、同模型重试/no fallback |
| Adapter | Pi protocol → canonical `TraceEvent`，工具调用 join、usage、真实源时间戳、脱敏 |
| Experiment | `ExperimentManifest`、固定配对键、数据集/run/policy 声明 |
| Comparison | compatibility/confounder、pair delta、aggregate outcome/token/latency/slice |
| Gate | `ACCEPT / REJECT / INSUFFICIENT_EVIDENCE`，每条规则携带 actual/threshold/evidence |
| Observatory | 只读 Compare、Gate、Episode Explorer，同一 canonical artifact 驱动 |
| Training View | SFT/on-policy RL eligibility，行为模型与目标模型关系 |

## 证据边界

1. 3 对真实执行足以证明链路和 reference mechanism，不足以宣称 benchmark 普遍提升。
2. 黑盒 Pi 输出能证明“工具错误后重复了相同动作”，但看不到 Pi 内部决定，因此归因为 `OBSERVED_TOOL_ERROR_LOOP`，不能升级成高置信度 Harness 根因。
3. task success 来自外部 exact JSON Verifier；模型自己写 `"status":"SUCCESS"` 不会被直接信任。
4. teacher 是 `gpt-5.6-luna`，目标学生是本地小模型，因此轨迹是 off-policy；可作为 SFT candidate，不具备 on-policy RL 所需的目标策略 token ids/logprobs。

## 复现

```bash
python3 -m src.cli demo-v2 --output artifacts/v2-harness-decision
python3 -m unittest discover -s tests -v
```

重点文件：

- `summary.json`：V2 结论和 claim boundary；
- `capture-evidence.json`：六条源轨迹 checksum 与独立 verifier 证据；
- `comparison.json`：逐对和聚合差异；
- `gate-result.json`：最终决策及每条规则；
- `observatory.html`：不修改数据的浏览界面；
- `training-candidates.json`：训练资格视图，不等于训练数据已经生产完成。

## 下一迭代

当前最有价值的下一步不是放宽 Gate，而是优化 candidate：减少一次或多次模型回合、压缩错误反馈、缓存目录发现结果，然后用同一 Experiment/Gate 重跑。只有在成功率保持、failure slice 不反弹且效率进入预算后，Gate 才应变为 `ACCEPT`。
