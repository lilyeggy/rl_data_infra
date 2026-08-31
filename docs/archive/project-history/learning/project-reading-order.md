# 项目阅读顺序与旧方向导读（已归档）

> 适用项目：Multi-Harness Agent Execution Data Plane
> 方向快照：2026-08-12
> 目标：先理解新主线，再阅读旧代码如何作为基础被复用。

## 0. 先记住一句话

> 本项目统一捕获不同 Agent Harness 的运行数据，并用这些数据做轨迹回放、故障归因、受控版本对比和回归验证；训练数据只是可选输出。

这不是 Agentic RL Trainer，也不是自动修改 Harness 的 Optimizer。

## 1. 第一遍：只读权威范围

依次阅读：

1. [`PROJECT_SCOPE.md`](../../PROJECT_SCOPE.md)：项目做什么、不做什么、如何验收；
2. [`PROJECT_PLAN.md`](../../PROJECT_PLAN.md)：架构、契约、采集、分析与实验协议；
3. [`IMPLEMENTATION_PLAN.md`](../../IMPLEMENTATION_PLAN.md)：Day 1–7 的新旧阶段关系。

读完应能解释：

```text
Harness
→ Capture
→ TraceEvent
→ AgentEpisode
→ Metrics / Attribution / Compare
→ Regression Gate / UI
```

以及为什么：

```text
AgentEpisode → RolloutRecord
```

是可选 Training View，而不是反方向。

## 2. 第二遍：理解数据契约

阅读 [`docs/data-contract.md`](../data-contract.md)，重点回答：

1. `TraceEvent` 和 `AgentEpisode` 为什么要分开？
2. raw fact、metric、diagnosis 和 Gate 为什么不能混在同一对象里？
3. `span_id/parent_span_id` 与简单时间顺序有什么区别？
4. partial trace 为什么仍然值得保存？
5. capability 缺失为什么必须返回 `NOT_OBSERVABLE` 或 `INSUFFICIENT_EVIDENCE`？
6. 为什么比较 Harness 前要检查 Experiment Manifest？

一条数据链应该这样理解：

```text
Model Proxy 捕获 MODEL_REQUEST
→ Environment Adapter 捕获 TOOL/SANDBOX events
→ Optional Hook 捕获 Harness decision
→ EventWriter append raw JSONL
→ EpisodeAssembler 去重、排序、检查完整性
→ AgentEpisode
→ Analysis 生成独立的 metrics/diagnosis
```

## 3. 第三遍：阅读后续实现计划

1. [`plans/day-05-execution-data-plane.md`](../../plans/day-05-execution-data-plane.md)
2. [`plans/day-06-harness-evaluation.md`](../../plans/day-06-harness-evaluation.md)
3. [`plans/day-07-observatory-and-packaging.md`](../../plans/day-07-observatory-and-packaging.md)

三天不是三个独立功能：

```text
Day 5：让不同 Harness 可被同一种方式观察
Day 6：让这些数据可以支持改进决策
Day 7：完成一次真实改进闭环并展示证据
```

如果 Day 5 的事件、capability 或 lineage 不可信，Day 6 的归因和对比就没有意义；如果 Day 6 没有可审计结论，Day 7 的 UI 只是漂亮日志页面。

## 4. 现有代码应该怎样读

### 4.1 Contract 基础

先读：

```text
src/contracts/rollout_record.py
src/contracts/rollout_batch.py
src/contracts/training_batch.py
src/contracts/capabilities.py
src/contracts/_json.py
```

这些代码展示了已经具备的工程原则：

- immutable/versioned contract；
- 不伪造缺失字段；
- checksum 和 deterministic serialization；
- capability gate；
- execution status 与 task outcome 分离。

但它们的字段仍然面向训练。不要把所有 event、span、manifest 和 diagnosis 继续塞入 `RolloutRecord`；新建上游 execution contract。

### 4.2 Source Adapter 基础

再读：

```text
src/sources/base.py
src/sources/polar.py
src/sources/jsonl.py
```

需要复用的是 adapter 隔离、显式字段映射、warning/error 和 fixture-driven testing。新的 capture adapter 与这些 source adapter 解决不同问题：

```text
旧 Source Adapter：已有 rollout payload → training view
新 Capture Adapter：运行中的可观测行为 → raw TraceEvent
```

### 4.3 测试

最后用测试反推设计：

```text
tests/contracts/
tests/sources/
tests/fixtures/
```

先保证旧测试不回归，再为 `contracts/capture/assembly/analysis` 增加新测试。不要为了新方向删除原来的正确约束。

## 5. Day 1–4 文档如何看

Day 1–4 是历史基础，不是废弃工作：

- Day 1：环境、版本、硬件证据；
- Day 2：Polar Calculator rollout 和真实字段；
- Day 3：Coding/SWE 多轮任务与 verifier 证据；
- Day 4：training-oriented contract 和 adapters。

它们证明我们不是凭空设计 schema，而是从真实 rollout artifact 出发。阅读时要区分两类内容：

历史材料已统一收入口：[Legacy Agentic RL Archive](../archive/legacy-agentic-rl/README.md)。除非需要核查已有代码或 Polar fixture 来源，后续开发不必逐份阅读。

| 内容 | 新项目中的位置 |
|---|---|
| model/tool/verifier/runtime 原始证据 | 继续复用为 capture/fixture 基础 |
| checksum、lineage、capability | 继续复用为通用工程原则 |
| RolloutRecord/TrainingReadyBatch | optional Training View |
| GRPO group、Slime、双卡训练 | 延后扩展，不是当前主线 |

深度学习文档保留的是旧阶段实现快照。若其中“下一步 Day 5/6”与权威计划冲突，以本导读和根目录三份权威文档为准。

## 6. 如何判断 Infra 有效

不能只展示“收集了多少日志”。最小证据链是：

```text
相同 task/model/environment/evaluator
→ Harness v1/v2 产生统一 Episode
→ metrics/diagnosis 发现可解释差异
→ trace 指向具体 event/artifact
→ Regression Gate 根据版本化规则给出结论
```

评价维度包括：

- 数据是否真实、完整、可重放；
- 不同 Harness 是否能进入同一契约；
- 不可观测项是否被诚实表达；
- 归因是否有 evidence；
- 对比是否控制变量；
- Gate 是否同时考虑效果、成本、延迟和可靠性；
- 一项 Harness 修改能否被重复验证。

## 7. Reference case 阅读方式

第一版优先做 Structured Tool Error Feedback：

```text
v1: raw stderr
v2: structured error object
```

不要只看最终 success。沿 Trace 检查：

1. tool error 是否被正确捕获；
2. Harness 反馈给模型的信息发生了什么变化；
3. 下一次 action 是否仍然重复；
4. 是否成功恢复；
5. 额外消耗了多少 turn、token 和时间；
6. Gate 为什么接受或拒绝。

## 8. UI 的正确角色

Harness Observatory 不计算隐藏逻辑。它展示已经由 core 生成的 canonical artifact：

```text
Episode Explorer：找到问题
Trace Timeline：解释问题
Harness Compare：验证修改
Regression Gate：给出发布结论
```

如果关掉 UI，CLI 和 JSON 仍应能完成整个审计；这能证明 UI 与 Infra 边界清楚。

## 9. 建议代码阅读顺序

当前代码：

1. `src/contracts/_json.py`
2. `src/contracts/capabilities.py`
3. `src/contracts/rollout_record.py`
4. `src/contracts/rollout_batch.py`
5. `src/sources/base.py`
6. `src/sources/jsonl.py`
7. `src/sources/polar.py`
8. 对应 tests 和 fixtures

新代码完成后：

1. `contracts/trace_event.py`
2. `contracts/agent_episode.py`
3. `capture/event_writer.py`
4. `assembly/episode_assembler.py`
5. `analysis/metrics.py`
6. `analysis/attribution.py`
7. `analysis/compare.py`
8. `analysis/regression_gate.py`
9. CLI 与 UI consumer

## 10. 面试自测题

1. Orchard 与这个项目的边界有什么不同？
2. 为什么不能只用 `RolloutRecord` 表示全部 Harness 执行？
3. Model Proxy 能看到什么、看不到什么？
4. Hook-enabled capture 比 black-box capture 多了什么可信事实？
5. 如何处理乱序、重复和不完整事件？
6. task failure 与 infra-invalid 如何影响成功率分母？
7. 怎样证明两次 Harness 运行可比较？
8. 为什么故障归因需要 evidence event IDs 和 rule version？
9. Regression Gate 为什么需要第三种 `INSUFFICIENT_EVIDENCE`？
10. UI 如何证明项目不是普通日志 Dashboard？
11. 一个 Harness 修改由谁提出？本项目具体优化了什么环节？
12. TrainingViewExporter 为什么不属于主链路？

## 11. 最终口述版本

> 项目最初从 Polar rollout 到训练数据的可靠转换入手，因此已经有 canonical record、capability、checksum 和 adapter 基础。后来我们把目标聚焦到 Agent Infra：在这些基础上增加多 Harness 执行数据面，通过 model proxy、environment capture 和 optional hooks 产生 TraceEvent，再组装为 AgentEpisode。分析层把原始事实转成带证据的指标和故障归因，在固定模型、任务、环境和 evaluator 后比较 Harness 版本，并用三态 regression gate 判断修改。UI 只是同一批 artifact 的可视化 consumer，现有 RolloutRecord 则保留为可选训练视图。
