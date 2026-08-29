# 本周两版迭代边界

## 原则

两版不是“多写一些字段”和“再美化 UI”，而是两个不同的 Agent Infra 证据闭环：

- V1 证明一次执行能够被可靠记录、组装和解释；
- V2 证明一项 Harness 修改能够在控制变量下被比较和决策。

## 前两天：V1 Observability Loop

交付状态：核心代码与冻结演示已完成，剩余时间用于审计和文档校准。

验收问题：

1. 进程中断后，已经发生的事实是否仍然存在？
2. 重复或乱序到达是否得到确定性 Episode？
3. task failure 和 infra invalid 是否被分开？
4. 缺失 capability 是否保持 `NOT_OBSERVABLE`？
5. 每个 diagnosis 是否能回到具体 event/artifact？
6. 不安装 GPU/Polar/Trainer 是否能跑完整单元测试？

范围：

```text
TraceEvent / Manifest / ArtifactRef / AgentEpisode
EventWriter / ArtifactStore / TraceRecorder / EnvironmentCapture / HarnessHook
EpisodeAssembler / Integrity / Lineage
EpisodeMetrics / AttributionEngine
CLI inspect / deterministic three-outcome demo
```

明确不进入 V1：Harness A/B、统计提升结论、Regression Gate、Observatory UI、训练循环。

## 后三天：V2 Harness Improvement Decision Loop

交付状态：真实 Pi 成对 capture、comparison、Gate、Observatory 和 Training View 已完成。reference candidate 正确性改善但效率超预算，当前 Gate 为 `REJECT`；下一轮目标是在不放宽阈值的前提下优化 recovery policy。

目标链路：

```text
同任务 + 同模型 API + 同环境 + 同 Verifier
              │
      Harness v1 / Harness v2
              │
    paired AgentEpisode comparison
              │
 ACCEPT / REJECT / INSUFFICIENT_EVIDENCE
```

计划交付：

1. `ExperimentManifest` 与逐字段 compatibility/confounder report；
2. 按 `task_id + seed + attempt` 对齐 control/candidate；
3. outcome migration、token/latency/tool recovery/failure slice diff；
4. 三态 Regression Gate 与阈值 evidence；
5. Structured Tool Error Feedback reference improvement：v1 raw stderr，v2 structured feedback；
6. 只读 Observatory：Episode Explorer、Trace Timeline、Harness Compare；
7. `TrainingCandidateView`：只标记来源、Verifier、behavior model 和可训练能力，不进入训练；
8. 更新技术讲解网页和 3–5 分钟面试演示脚本。

V2 验收约束：

- 模型必须固定为同一 API 模型和 revision；
- 任何非 Harness 控制变量不一致都必须出现在 confounder；
- infra-invalid 单独报告，不进入 task success 分母；
- 样本小就只证明机制，不声称统计普遍提升；
- Gate 的每条规则都有 actual、threshold、pass/fail 与 evidence。

## 两版之间不改变的公共契约

V2 只能消费或扩展 V1 canonical artifacts，不能为了展示结果重写 raw event。这样面试时可以从最终 Gate 一直向下追溯到原始工具错误，而不是展示两套彼此无关的 demo。
