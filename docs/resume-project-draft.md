# Agent Improvement Data Plane：简历项目稿与反向学习清单

> 使用说明：本稿面向 Agent Infra / RL Infra / Python 后端方向。方括号中的内容必须在投递前补充或确认；没有亲自负责的工作，不要使用“主导”。

## 简历版项目经历

### Agent Improvement Data Plane｜Agent 可信执行与持续改进基础设施

**技术栈：** Python、Docker、vLLM、Model Proxy、Pi、Polar、LoRA/SFT、JSONL/WAL

- 设计 Agent 可信执行证据模型，以统一 `ExecutionIdentity` 贯穿 Harness、模型、工具、环境与 Verifier，基于不可变事件、确定性 Episode 和内容寻址 Bundle 建立端到端 lineage，使每项分析或训练结论均可回溯至同一次真实执行。
- 构建面向不同消费者的 fail-closed 认证机制，严格区分任务成败、基础设施有效性与轨迹完整性，并校验采集能力、Verifier 证据、policy fingerprint 及原生 token/logprob 语义，防止“格式合法但语义不可信”的数据进入 SFT、Preference 或 on-policy RL。
- 建立 Harness 与模型两个受控改进闭环：通过版本化 Manifest 冻结非实验变量，以逐任务配对比较和 Regression Gate 评估 Harness；以 Dataset Manifest、任务级防泄漏切分及固定 Holdout 评估模型，candidate 无提升时自动输出 `REJECT / NO_IMPROVEMENT`。
- 落地 Local Docker、Pi Direct、Polar Live 等执行来源及 Model Proxy、事件采集、Verifier、artifact 存储和恢复编排；在 200 条 MBPP rollout 中认证 140 条 SFT 合格样本，并完成 14B 模型 rollout → LoRA SFT → Holdout 的真实单卡链路，核心训练指标 `[待补充]`。

> 投递前：最后一条指标必须替换为真实实验结果。如果最终没有提升，应改成“定位模型输出格式退化问题，并由 Gate 阻止无收益版本晋升”，不要填写虚假提升。

## 三条精简版

- 设计 Agent 可信执行数据平面，通过统一 identity、不可变事件、确定性 Episode 与内容寻址 Bundle 关联 Harness、模型、工具、环境和 Verifier，使改进结论可回溯至原始执行证据。
- 构建消费者级 fail-closed 认证，分离任务结果、基础设施有效性与轨迹完整性，并校验 capability、policy lineage 及 token/logprob 语义，阻止不可信轨迹进入 SFT/Preference/RL。
- 建立 Harness 配对回归与模型训练评估双闭环，落地 Local/Pi/Polar 执行及 14B 单卡链路；在 200 条 MBPP rollout 中认证 140 条训练样本，模型指标 `[待补充]`。

## 30 秒口述版

我做的是一个 Agent 执行数据平面。Agent 轨迹不能因为格式正确就直接拿去训练，因为它可能不完整、来自错误 policy、缺少 token/logprob 语义，或者把基础设施故障当成模型失败。这个项目从 Local Docker、Pi、Polar 等执行来源采集不可变事实，把事件、模型证据和 verifier 结果绑定成可追溯的 Episode 与 ExecutionBundle，再针对 Harness 分析、SFT、Preference、RL 分别做 fail-closed 认证。最后只有满足特定消费者证据要求的数据才能进入数据集或回归 Gate。

## 90 秒口述版

项目背景是：真实 Agent 执行会产生模型调用、工具使用、sandbox 状态和 verifier 结果，但普通日志缺少稳定 identity、完整性判断和 policy lineage，不能安全地用于 Harness 归因或模型训练。

我的方案分成四层。第一层通过 Local Docker Launcher、Model Proxy 和 Harness ingress 采集 append-only 的执行事件与 artifact；第二层使用统一 ExecutionIdentity，把事件组装为 AgentEpisode，并通过内容寻址的 ExecutionBundle 严格关联 policy trace 和 verifier evidence；第三层进行消费者级认证，例如 SFT 要求有效、完整且 verifier 通过，on-policy RL 还要求原生 token IDs、对齐 logprobs 和一致的 policy fingerprint；第四层把合格数据编译成版本化 dataset manifest，或者进入 Harness paired comparison 与 regression gate。

可靠性方面，我把任务失败、基础设施有效性和数据完整性拆开，使用幂等事件、冲突 quarantine、checksum lineage、原子写和 CAS 防止损坏或过期状态被静默接受。在实际验证中完成了 Pi、受控模型、Verifier 和 ExecutionBundle 的真实链路；SFT candidate 在固定 holdout 上没有提升，系统按设计拒绝晋升。这个结果也验证了系统不会把训练 loss 或单次成功包装成模型能力提升。

## 投递前必须确认的真实性边界

以下内容只有确实由你承担时才能保留“设计并实现”：

- canonical contract 与 schema 设计；
- Local Execution Orchestrator / Model Proxy / capture 实现；
- certification 与 dataset gate；
- 单 GPU workflow 与 CAS；
- Pi、Polar、Slime 或 A6000 实验操作。

若某部分主要由团队其他成员实现，应改为“参与”“负责其中的……模块”或删除。面试官通常会沿简历动词判断你的 ownership 深度。

当前不得写成已完成成果：

- Slime 端到端正式模型更新；
- policy-v1 已通过 unseen DEV gate；
- 已实现大规模 Agentic RL；
- 已证明 benchmark 泛化提升；
- Polar crash/timeout/retry 的真实服务集成测试全部完成。

## 简历内容 → 反向学习路线

### 主张 1：统一可信执行契约

**必须能回答：**

- 为什么 `task_id` 不能代替 `ExecutionIdentity`？
- `TraceEvent`、`AgentEpisode`、`ExecutionBundle` 各自解决什么问题？
- 为什么 bundle 保存 checksum 引用，而不是复制原始数据？
- 为什么 certification 不写回 bundle？

**必须定位的代码：**

- `src/contracts/execution_identity.py`
- `src/contracts/trace_event.py`
- `src/contracts/agent_episode.py`
- `src/contracts/execution_bundle.py`
- `src/assembly/episode_assembler.py`
- `src/assembly/execution_bundle_assembler.py`

**必须做的实操：** 检查一条真实 raw event、episode 和 bundle，手工追踪 identity 与 checksum。

### 主张 2：fail-closed certification

**必须能回答：**

- task status、execution validity、integrity 为什么分开？
- `REJECTED` 与 `INSUFFICIENT_EVIDENCE` 有什么区别？
- 为什么外部 API 文本不能重新 tokenize 后冒充 behavior tokens？
- SFT、Preference 和 on-policy RL 的证据要求有何不同？

**必须定位的代码：**

- `src/validation/episode_semantics.py`
- `src/certification/engine.py`
- `src/training/eligibility.py`
- `src/training/policy_fingerprint.py`

**必须做的实操：** 修改或构造一条缺 verifier、缺 token evidence 或 policy mismatch 的 fixture，观察认证结果。

### 主张 3：本地事务化执行与可靠性

**必须能回答：**

- 为什么长时间 Agent 执行采用 append-only event log？
- 幂等重放与同 ID 内容冲突应如何区别？
- orchestrator 为什么统一拥有 prepare、proxy、Harness、verifier、finalize 和 cleanup？
- Ctrl-C、timeout、artifact 缺失时系统留下什么证据？

**必须定位的代码：**

- `src/orchestration/local_execution.py`
- `src/capture/event_writer.py`
- `src/capture/model_proxy.py`
- `src/capture/harness_http.py`
- `src/assembly/local_run_finalizer.py`
- `src/storage/event_store.py`

**必须做的实操：** 阅读一次 local smoke artifact，画出 raw events 到 finalized bundle 的时间线。

### 主张 4：数据集和评估 Gate

**必须能回答：**

- 为什么按 logical task 做 split，而不是按 trajectory 随机切分？
- Preference pair 为什么必须严格成对且 identity 可比？
- 为什么 train loss 下降不能证明 agent capability？
- Harness A/B 需要冻结哪些变量？

**必须定位的代码：**

- `src/learning/dataset_compiler.py`
- `src/learning/canonical_sft_dataset.py`
- `src/analysis/compare.py`
- `src/analysis/regression_gate.py`
- `src/evaluation/layered_eval.py`

**必须做的实操：** 构造 task split leakage 或不完整 preference pair，验证数据集构建被阻断。

### 主张 5：单 GPU 闭环

**必须能回答：**

- 为什么 rollout、train、evaluate 需要分时？
- 状态机阻止哪些非法跃迁？
- checksum CAS 防止什么并发/恢复问题？
- policy fingerprint 如何随 checkpoint handoff？

**必须定位的代码：**

- `src/orchestration/single_gpu.py`
- `src/orchestration/workflow.py`
- `src/recovery/`

**必须做的实操：** 运行 workflow 状态测试，并故意使用 stale checksum 推进状态。

## 建议学习优先级

面试时间紧时按以下顺序：

1. 主张 1：可信执行契约；
2. 主张 2：认证与失败语义；
3. 主张 3：本地执行链路；
4. 主张 4：数据集与评估；
5. 主张 5：单 GPU 与外部集成。

前两项决定你能否讲清项目最独特的价值；后三项决定面试官深入工程实现时你能否接住。
