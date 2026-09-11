# Agent Improvement Data Plane：项目范围

> 状态：Local Docker 闭环与 A6000 上真实 Pi/14B live rollout 已验证；APPS clean-v2 SFT 与三轮单卡 on-policy GRPO cycle（apps-rl-cycle-001/002/003）已执行，官方三基准与固定 APPS holdout 显示 candidate 未取得稳定提升；正式 Slime trainer 尚未接入。

## 一句话定义

一个 Harness-neutral、trainer-neutral 的 Agent 改进数据基础设施：从真实执行中同时保存执行证据与 policy evidence，经版本化 verifier 和 fail-closed certification 后，编译为 Harness 分析、SFT、Preference、RL 和 Evaluation 所需的不同数据视图。

## 两个一等闭环

### Harness 闭环

冻结模型、任务和环境，只改变 Harness，通过逐任务配对比较和回归 Gate 决定是否晋升。

### 模型闭环

冻结 Harness、任务和环境，只改变 policy；只有满足目标 consumer profile 的认证数据才能进入训练和评估。

## 核心拥有的能力

- `ExecutionIdentity`：贯穿 execution、policy trace、verifier、certification 和 dataset；
- Local Docker Launcher：默认启动任意 Harness，注入执行身份并施加容器隔离；
- `RolloutProducer`：隔离 Local、Pi Direct、Polar Live 与未来 producer；
- Model Proxy evidence：区分可观测文本与 token/logprob/policy 训练证据；
- Local Execution Orchestrator：一次性拥有 prepare、proxy、Harness、verifier、finalize 和 cleanup；
- Harness event ingress：为任意 Harness 提供带执行身份和鉴权的可插拔事件入口；
- ArtifactStore：内容寻址保存 stdout/stderr、workspace diff/patch 和 verifier report；
- immutable raw evidence、`TraceEvent`、`AgentEpisode`；
- verifier report、Episode certification 和 consumer eligibility；
- Harness attribution/comparison/regression gate；
- versioned dataset manifest/compiler；
- Polar/Slime adapters，不复制训练框架。

## 外部组件职责

- Local Docker Launcher 是默认执行入口；
- Polar 是可选的远程/批量 rollout producer；
- Slime 是可选的 RL trainer consumer；
- 当前项目不重写分布式 trainer、Megatron、SGLang 或权重同步。

## 有限资源约束

开发拓扑采用分时同步循环：

```text
Mac local sandbox + cloud policy-vN rollout → freeze certified batch → unload serving
→ train-only update → policy-vN+1 → reload serving → next rollout
```

先验证少量真实任务的正确性和学习信号，不追求生产级并发或复现上游多卡拓扑。

## 明确非目标

- 不把普通 Trace 自动宣称为 RL rollout；
- 不让 infrastructure failure 变成 reward 0；
- 不在一次因果实验里同时修改 Harness 与模型；
- 不训练 intended evaluation split；
- 不继续扩建归档的旧自定义 GRPO（当前单卡 RL 更新使用独立极简 GRPO LoRA trainer，正式训练仍以 Slime 接入为目标）；
- 不把历史 demo/release note 当作当前运行入口。

## 当前验收目标

1. Local Launcher 能在容器隔离中启动真实 Harness 并注入统一 identity；
2. 一次真实 rollout 同时关联 `AgentEpisode` 与 token-faithful policy trace；
3. Local/Polar/Pi producer capability 可机器检查；
4. 统一 certification 决定 Harness/SFT/Preference/RL/Evaluation 资格；
5. 认证数据集拥有 manifest、lineage、dedup 和 split 证据；
6. 单张租用 GPU 上跑通一次 policy-v0 → rollout → adapter update → policy-v1 → evaluation；
7. 所有晋升结论可以回溯到不可变原始证据。
