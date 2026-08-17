# Day 7：Harness Observatory、Reference Improvement 与项目包装

> 状态：已完成；Observatory、TrainingCandidateView、release 与教学材料均已交付。
> 目标：用同一批 canonical artifact 完成 CLI、UI、实验报告和面试演示闭环。

## 1. 阶段目标

1. 完成 reference Harness v1/v2 重跑；
2. 生成 Episode、metrics、diagnosis、comparison 和 Gate artifact；
3. 实现只读 Harness Observatory；
4. 完成 clean reproduction、测试和限制审计；
5. 完成 README、架构图、实验报告和面试材料。

## 2. UI 页面

### Episode Explorer

- Episode 列表和 outcome/integrity 摘要；
- harness/version、task、model、failure、termination、capability 筛选；
- duration、tokens、tool calls、verifier 和 infra validity。

### Trace Timeline

- Model、Tool、Sandbox、Harness、Verifier lane；
- parent/child span、status、attempt 和 latency；
- request/result 摘要和 artifact 链接；
- diagnosis 高亮到 evidence；
- `NOT_OBSERVABLE` capability 提示。

### Harness Compare

- control/candidate 同 task 并排；
- outcome、turn、tool、duplicate、token、cost、latency 和 verifier diff；
- aligned behavior changes 和 failure slices；
- Gate verdict、规则、actual/threshold 和证据。

UI 读取本地生成的 JSON；不做 auth、在线数据库、WebSocket、Harness 编辑器或微服务拆分。

## 3. 最终实验流程

```text
1. 固定 ExperimentManifest
2. 运行 Harness v1/control
3. capture + assemble + analyze
4. 在 UI 中定位目标 failure
5. 应用单项 Harness policy 修改
6. 运行 Harness v2/candidate
7. compare + gate
8. 在 UI 中展示行为差异和结论证据
```

首选故事：工具执行失败后，v1 丢失关键错误结构，v2 使用结构化错误反馈并改善恢复行为。

## 4. Clean reproduction

Runbook 至少覆盖：

```text
install core dependencies
run contract/unit tests
generate or run control traces
generate or run candidate traces
assemble episodes
compute metrics and diagnoses
compare runs
evaluate regression gate
start UI with artifact directory
```

所有命令、配置、版本、seed、artifact 路径和预期输出明确。现场演示使用冻结 artifact，避免等待长 rollout。

## 5. 项目审计

- raw event 与 derived diagnosis 分离；
- secret/private path 已清理；
- synthetic fault 明确标记；
- task failure 与 infra invalid 未混淆；
- Harness 内部不可观测项未被推断；
- control/candidate manifest 可比；
- UI 和 CLI 使用同一份 canonical artifact；
- 所有指标可以反查 Episode/event；
- 小样本不声称统计显著或普遍涨点；
- optional training exporter 不被描述为主线完成条件。

## 6. 文档集

- README：问题、定位、快速开始、真实结果和限制；
- Architecture：capture、contract、assembly、analysis、compare、Gate、UI；
- Data Contract：字段、capability、integrity 和 error semantics；
- Experiment Protocol：控制变量、paired comparison 和 Gate；
- Reference Improvement Report：问题、证据、修改、结果、回归和限制；
- Demo Script：3–5 分钟稳定演示。

## 7. 演示脚本

1. 打开 Episode Explorer，筛选目标 failure cluster；
2. 进入 v1 Trace Timeline，定位 tool error 与后续重复/终止；
3. 展示 Diagnosis 的 event/artifact evidence；
4. 打开 v1/v2 Harness Compare；
5. 展示 v2 的恢复行为、结果和资源代价；
6. 展示 Regression Gate verdict 与阈值；
7. 回到架构图说明数据如何跨 Harness 统一。

## 8. 面试回答边界

可以说：

> Infra 能捕获不同 Harness 的统一执行数据，定位特定失败模式，并在控制变量一致的 paired run 中验证 Harness 修改，最终由回归门给出可审计发布结论。

不应说：

> 系统能够自动生成最优 Harness，或已经证明在大规模 benchmark 上普遍提升。

## 9. 验收门

- [ ] v1/v2 完整 artifact 可重复生成；
- [ ] UI 三个页面使用 canonical JSON；
- [ ] Trace Timeline 能定位 diagnosis evidence；
- [ ] Compare 页面展示效果、成本和可靠性；
- [ ] Gate verdict 与 CLI 输出一致；
- [ ] clean runbook 可执行；
- [ ] 测试全部通过；
- [ ] README、实验报告、截图和简历描述均可反查证据；
- [ ] 限制与不可观测能力清楚可见。

## 10. 阶段产物

```text
src/ui/ or ui/
artifacts/reference-improvement/
docs/architecture.md
docs/experiment-protocol.md
docs/reference-improvement-report.md
docs/demo-script.md
README.md
```

## 11. 执行记录

```text
状态：NOT_STARTED
UI入口：
control/candidate：
paired episodes：
Gate verdict：
测试：
演示时长：
最终限制：
```
