# Multi-Harness Agent Execution Data Plane：项目范围

> 状态：V2 已完成（2026-08-17）；进入 V2.1 recovery efficiency iteration
> 面试方向：Agent Infra / Agent Harness / Runtime Data / Evaluation / Observability

## 一句话定义

构建一个面向多种 Agent Harness 的执行数据基础设施：在不绑定具体 Harness、模型或训练框架的前提下，捕获 Model、Tool、Sandbox、Harness 与 Verifier 事件，组装为统一的 `AgentEpisode`，用于轨迹回放、故障归因、Harness/版本对比和回归验证，并可选导出为模型训练数据。

```text
Any Harness × Same Task/Model/Environment
                    ↓
        Proxy + Environment Capture + Hooks
                    ↓
          Canonical AgentEpisode
                    ↓
 Trace / Metrics / Attribution / Compare / Regression Gate
                    ↓
        CLI + Harness Observatory UI
                    └── optional Training View
```

## 核心问题

不同 Harness 对模型调用、上下文、工具错误、循环、重试、验证和终止的处理方式不同，但它们产生的数据通常无法直接比较，也无法回答：

- 失败来自模型、Harness、Sandbox、模型后端还是 Evaluator？
- 某个 Harness 修改究竟提升了成功率，还是只增加了 token、延迟或基础设施失败？
- 两次运行使用的模型、任务、环境、工具和评估器是否真的可比？
- 一项结论能否追溯到具体 event、span、artifact 和配置版本？
- 当前采集链路看不到哪些内部行为，哪些判断只能标记为证据不足？

本项目不承诺自动写出更好的 Harness；它提供可复现的证据和回归门，判断人工或自动提出的 Harness 修改是否更好。

## 核心模块拥有的能力

### 1. 多源执行捕获

- OpenAI-compatible Model Proxy：捕获 request/response、tool schema、usage、latency 和可用的 token metadata；
- Environment/Sandbox Adapter：捕获命令、stdout/stderr、exit code、文件/patch、生命周期、超时和 verifier artifact；
- Optional Harness Hook：捕获 context selection、compaction、retry、loop control、verification 和 termination decision；
- append-only raw event writer：先保存事实，再异步派生 Episode 和诊断。

### 2. Canonical Execution Contract

- `TraceEvent`：一个可排序、可关联的执行事实；
- `AgentEpisode`：一次 task execution 的完整 envelope；
- `HarnessManifest`、`EnvironmentManifest`：保证对比时控制变量可检查；
- `ArtifactRef`：文件、patch、日志、verifier 报告等外部证据；
- schema version、checksum、lineage、deduplication 和 partial episode 语义。

### 3. 可观测性与诊断

- outcome、turn、tool、token、cost、latency、loop、duplicate action、recovery 和 verifier 指标；
- MODEL / HARNESS / SANDBOX / MODEL_BACKEND / EVALUATOR / EXTERNAL_SERVICE / UNKNOWN 一级归因；
- Harness 二级 reason code 和逐条 evidence；
- 原始事实与派生诊断分离，规则版本可追踪。

### 4. Harness 对比与回归验证

- 在相同 task、model、environment、toolset、evaluator 和 seed 下比较 Harness A/B 或 v1/v2；
- 输出逐任务 paired diff、聚合指标、失败切片和成本/延迟变化；
- Regression Gate 输出 `ACCEPT`、`REJECT` 或 `INSUFFICIENT_EVIDENCE`；
- 至少完成一个 reference improvement case，证明数据能支持 Harness 迭代。

### 5. 展示与扩展

- CLI：capture、inspect、compare；
- Harness Observatory：Episode Explorer、Trace Timeline、Harness Compare；
- 可选 `TrainingViewExporter`：把满足能力要求的 Episode 转为现有 `RolloutRecord`，供 SFT/RL 使用。

## 采集能力等级

| 等级 | 能看到什么 | 能做什么 |
|---|---|---|
| Black-box | 模型请求/响应、外部执行、结果与 verifier | 跨 Harness 基础对比、性能和外部失败分析 |
| Hook-enabled | context、compaction、retry、termination 等 Harness decision | 更精确的 Harness 归因与策略诊断 |
| Managed | snapshot、replay、branching、策略替换 | 可控反事实实验；第一版不要求完整实现 |

缺失能力必须显示为 `NOT_OBSERVABLE`；不得根据普通日志文本推断隐藏的 Harness 内部状态。

## 现有代码的定位

现有代码不推倒重写：

- `src/contracts/rollout_record.py`、`RolloutBatch`、`TrainingReadyBatch` 和 `ResampleRequest` 保留为训练兼容层；
- Polar/JSONL Source Adapter、capability、checksum 与 lineage 逻辑继续复用；
- 新增 `AgentEpisode`/`TraceEvent` 上游层；
- 新增 Episode → `RolloutRecord` 的可选 exporter；
- 新项目主链路不依赖 Slime、Megatron 或 GRPO。

## 第一版明确非目标

- 不自动生成或修改 Harness 源码；
- 不做 Harness 自进化平台或搜索算法；
- 不重写完整 Agent loop；
- 不复制 Orchard 的 Kubernetes 环境平台；
- 不做生产级多租户、权限、计费或大规模调度；
- 不以 RL 训练、模型涨点或 SWE-Bench SOTA 作为完成条件；
- 不依赖 LLM Judge 替代确定性规则和 verifier；
- 不声称看到了未被采集的 chain-of-thought 或 Harness 内部决策；
- 不把 UI 做成核心逻辑或在线编辑器。

## 一周完成标准

第一版必须形成下面的证据闭环：

```text
2 个 Harness 或 Harness 版本
× 同一模型
× 同一组任务、环境、工具和 Verifier
→ 统一 AgentEpisode
→ 可回放 Trace 与统一指标
→ 定位一个 Harness 行为问题
→ 实施一个有边界的 Harness 策略修改
→ 对比 v1/v2
→ Regression Gate 给出可解释结论
```

具体要求：

1. 至少两个可比较的 Harness manifest；
2. 每条 Episode 具有稳定事件顺序、parent/child span、artifact lineage 和 capability 声明；
3. 至少覆盖一次成功、真实任务失败和基础设施无效；
4. 诊断结果包含 reason code、evidence、confidence 和 rule version；
5. 同任务对比能够展示 outcome、行为、成本、延迟和失败切片；
6. Regression Gate 能拒绝回归并在样本不足时返回证据不足；
7. UI 能从本地标准化数据展示 Episode、时间线和 A/B 结论；
8. 一个真实改进案例能从原始事件追溯到最终 Gate 判断；
9. 现有单元测试继续通过，新增核心逻辑无需安装 Polar/Slime；
10. README、演示和简历描述不超过实际 artifact 证据。

## 项目完成后的准确表述

> 我们实现的不是 Harness Optimizer，而是一个 Multi-Harness Agent Execution Data Plane。它把不同 Harness 的模型调用、工具、Sandbox、Harness decision 和验证事件统一成可审计 Episode，并在受控变量下完成失败归因、版本对比和回归验证，从而为 Harness 改进提供可复现的数据证据；训练数据只是可选下游视图。

## V2 实际验收快照

- 真实 Harness：Pi 0.84.2；固定模型：`opencode-go/gpt-5.6-luna`；无 silent fallback；
- 3 对真实 control/candidate，paired coverage 100%，无 compatibility mismatch；
- exact-verifier success：0/3 → 3/3；`OBSERVED_TOOL_ERROR_LOOP`：3 → 0；
- token +43.54%、latency +51.15%，超过冻结阈值，Gate 为 `REJECT`；
- 3 条 successful teacher Episode 可作 off-policy SFT candidate，0 条满足 on-policy RL 条件；
- 样本只证明 reference mechanism，不声称 benchmark 普遍提升。
