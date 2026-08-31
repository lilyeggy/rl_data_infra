# ADR-0002：一个可信证据底座，两个隔离改进闭环

- 状态：Accepted
- 日期：2026-08-22

## 决策

项目从“训练视图可选的 Harness observability”升级为 Agent Improvement Data Plane。Harness 改进和模型改进是同等重要的下游，但必须共享不可变执行事实，并通过不同认证 profile 获取使用资格。

## 边界

- Local Docker Launcher 是默认执行入口；Polar 是可选远程/批量 producer，Pi Direct 是受限 producer；
- Model Proxy 可以转发到云端受控模型，也可以观察外部 API；只有原生 token/logprob 和不可变 policy identity 才形成 RL 证据；
- 本项目拥有 identity、evidence、verification、certification、dataset 和 experiment lineage；
- Slime 拥有正式训练、SGLang serving、Megatron 和权重同步；
- 旧自定义 GRPO/SFT 不再是受支持入口。

## 因果隔离

- Harness 实验冻结 policy；
- 模型实验冻结 Harness；
- 两者都必须使用不可变 task/environment/evaluator manifests；
- 联合发布只能组合已经分别通过 Gate 的版本。

## 单 GPU 决策

有限资源环境采用 Mac Docker sandbox + 单张租用 GPU、batch-synchronous、time-sliced workflow。目标是验证少量真实任务的可信闭环，不复现上游多卡拓扑。

## 后果

- 当前外层主链路需要重构；
- Trace/AgentEpisode/storage/analysis 内核继续保留；
- 所有训练和评估入口最终必须消费统一 certification decision；
- archive 代码不得被生产模块导入。
