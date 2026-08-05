# Rollout-Agnostic Agentic RL Data Pipeline

> 状态：项目边界已更新，等待按新阶段实施
> 实施顺序：先复现 Polar rollout，再实现独立数据模块，最后自定义双卡 Agentic RL 训练
> 硬件基线：2 × NVIDIA RTX PRO 6000 96GB

## 1. 项目定义

本项目构建一个可插拔的 Agentic RL rollout 数据处理模块：从任意 Rollout Producer 接收 Agent trajectory，通过统一数据契约完成失败分类、训练信号筛选和 policy-consistent GRPO group 构建，再交给任意 Trainer Adapter。

项目名称：

> **Rollout-Agnostic Agentic RL Data Pipeline**

中文：

> **与 Rollout 框架解耦的 Agentic RL 数据处理流水线**

核心问题：

> 如何把不同 Harness/Rollout 系统产生的执行结果，稳定转换为有效、有训练信号、满足当前 policy 与 GRPO 分组语义的训练批次，同时把无效基础设施结果挡在 Trainer 之外？

## 2. 为什么先复现 Polar

Polar 已经提供真实 Coding Agent rollout 所需的关键路径：任务调度、Harness/runtime、模型 API proxy、token trace、trajectory builder 和 verifier。先复现 Polar 的 rollout 部分可以让我们基于真实 artifact 定义 contract，而不是凭空发明字段。

复现 Polar 的目的：

1. 理解一次真实 Agent rollout 的生命周期和产物；
2. 获取 Calculator 与 SWE/Coding 的 golden fixtures；
3. 确认 token、tool、termination、verifier 和 reward 的实际来源；
4. 实现第一个 `PolarSourceAdapter`；
5. 为后续处理器和训练闭环提供真实输入。

Polar 不是核心依赖。核心包在没有安装或启动 Polar 时也必须可测试和运行。

## 3. 项目边界

### 3.1 Rollout Producer 负责

- task sampling 与 rollout 执行；
- Agent/Harness loop、工具和环境交互；
- 模型生成请求；
- 原始状态、termination 和 verifier/reward 产物；
- 响应 `ResampleRequest`（如果上层选择自动补采）。

Reference Provider 是 Polar，但也可以是 VeRL rollout、自定义 Harness 服务或离线 JSONL。

### 3.2 核心数据模块负责

- Source Adapter contract 与 capability；
- canonical `RolloutRecord` / `RolloutBatch`；
- failure classification；
- high-signal/reward-variance filtering；
- policy-consistent GRPO group building；
- `TrainingReadyBatch`、`ProcessingReport` 和 `ResampleRequest`；
- 拒绝原因、配置版本和 checksum；
- Trainer Adapter contract。

### 3.3 Trainer Consumer 负责

- advantage、GRPO/PPO loss；
- forward/backward 和 optimizer；
- checkpoint；
- 将新 policy 提供给下一轮 rollout。

Reference Consumer 是 Slime/Megatron，但核心模块不直接依赖它。

### 3.4 明确不负责

- 不重新实现 Polar 或其他 Rollout Framework；
- 不重新实现 Coding Agent Harness；
- 不发明新的 GRPO/PPO 算法；
- 不做多 Session、Session Graph 或信用分配；
- 不做通用生产级数据湖、复杂 Dashboard 或大而全 Quality Gate；
- 不训练 Process Reward Model；
- 不实现通用异步 off-policy correction；
- 第一版不深度适配多个 Trainer；
- 不复现 Polar 官方 8-GPU 训练拓扑。

## 4. 系统架构

```text
                         Rollout Producers
              ┌──────────────┼──────────────┐
              │              │              │
            Polar       Custom Harness    JSONL
              │              │              │
              └────── Source Adapters ──────┘
                             │
                             ▼
                 Canonical Rollout Batch
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
      FailureClassifier  SignalFilter  PolicyConsistent
                                      GroupBuilder
              └──────────────┼──────────────┘
                             │
             ┌───────────────┴────────────────┐
             ▼                                ▼
     TrainingReadyBatch                ResampleRequest
             │                                │
      Trainer Adapters                Rollout Controller
       ┌─────┴─────┐                  （项目外或 example）
       ▼           ▼
     Slime        VeRL/other
       │
       ▼
  Megatron GRPO → checkpoint → next policy rollout
```

## 5. 核心数据契约

### 5.1 RolloutRecord

第一版最小字段：

```text
schema_version
source_type
source_record_id
trajectory_id
task_id
group_id
policy_version

token_ids
action_mask or loss_mask
old_logprobs                 # 按目标算法可选/必需
reward

rollout_status
termination_reason
verifier_status
verifier_evidence_ref

source_payload_ref
metadata
```

Adapter 不得静默伪造缺失的 token、logprob、reward 或 policy version。

### 5.2 SourceCapabilities

```text
TOKEN_IDS
ACTION_MASK
OLD_LOGPROBS
REWARD
GROUP_ID
POLICY_VERSION
VERIFIER_EVIDENCE
TOOL_EVENTS
```

每个 Source Adapter 声明它能可靠提供的 capability。每个 Processor/Trainer Adapter 声明其 required capabilities；不满足时必须显式拒绝或降级。

### 5.3 ValidityDecision

第一版固定三类：

```text
VALID_SUCCESS
VALID_FAILURE
INVALID_INFRASTRUCTURE
```

`VALID_FAILURE` 是模型在正常环境下真实失败，可以作为 reward=0 数据。`INVALID_INFRASTRUCTURE` 包括 runtime、Harness、模型服务和 verifier 非任务性失败，不能直接作为模型负奖励。

### 5.4 TrainingReadyBatch

```text
batch_id
task_groups
policy_version
group_size
records
reward_statistics
required_capabilities
processing_config_version
input/output checksums
```

### 5.5 ResampleRequest

```text
request_id
task_id
group_id
policy_version
required_count
reason
constraints
```

核心模块只输出补采意图，不直接依赖或调用 Polar。

## 6. 第一版三个可插拔 Processor

### 6.1 FailureClassifier

输入 rollout 状态、termination、verifier 和来源 metadata，输出三类 validity 与 versioned reason code。

首批 reason code：

```text
TASK_SUCCEEDED
TASK_FAILED
RUNTIME_PREP_FAILED
HARNESS_CRASHED
MODEL_BACKEND_FAILED
VERIFIER_FAILED
VERIFIER_TIMED_OUT
ROLLOUT_TIMED_OUT
TRACE_INCOMPLETE
TRAINING_FIELDS_INVALID
```

### 6.2 SignalFilter

按 task/group 检查：

- 至少包含目标数量的 valid records；
- reward 有可用的组内差异；
- 不是全部基础设施失败；
- trainable token 非空；
- 输入能力满足目标 Trainer Adapter。

默认只输出确定性规则和统计，不训练 reward/process model。

### 6.3 PolicyConsistentGroupBuilder

按以下 key 分组：

```text
task_id + group_id + policy_version
```

职责：

- 不混合 policy version；
- 过滤 `INVALID_INFRASTRUCTURE`；
- 构建固定 group size；
- 不足时生成 `ResampleRequest`；
- 超量时使用显式、可复现的 selection policy；
- 记录接受、拒绝和补采数量。

## 7. 插件接口

第一版使用简单 Python Protocol/ABC 与配置 registry，不引入微服务或复杂插件平台。

```python
class SourceAdapter:
    def capabilities(self) -> set[str]: ...
    def convert(self, source_payload) -> list[RolloutRecord]: ...

class RolloutProcessor:
    required_capabilities: set[str]
    def process(self, batch: RolloutBatch) -> ProcessResult: ...

class TrainerAdapter:
    required_capabilities: set[str]
    def convert(self, batch: TrainingReadyBatch): ...
```

处理顺序由配置声明：

```yaml
processors:
  - failure_classifier
  - signal_filter
  - policy_consistent_group_builder
```

## 8. Polar Reference Reproduction

### 8.1 Calculator

使用 1 个 Rollout Server、1 个 Gateway、1 个 SGLang backend、1 个 Harness 和 1 个 task。保存 request、response、summary、服务日志、模型 token metadata 和 runtime/evaluator 产物。

### 8.2 Coding/SWE Rollout

选择少量稳定任务，运行完整：

```text
task → Harness → multi-turn model/tool calls → patch → verifier → reward
```

保存至少成功/真实失败/基础设施失败三类 fixture。如果某一类别难以自然产生，可以在 fixture 层做可控故障注入，但必须标记为 synthetic fault。

### 8.3 复现不包含

- 官方 4 train + 4 serve GPU 拓扑；
- 官方完整长程训练参数；
- 为了匹配官方硬件而扩大模型或上下文；
- 把 Slime 当作 Polar rollout 的必要组成。

## 9. 自定义双卡 Agentic RL

Reference 训练采用 staged synchronous loop：

```text
policy_k
→ rollout phase
→ process phase
→ train phase
→ checkpoint policy_k+1
→ reload SGLang
→ rollout phase
```

### 9.1 Rollout Phase

- 使用一张或两张 GPU 运行 SGLang；
- Polar/Harness 在 CPU/runtime 上执行；
- 每个 task 采样 group size 2，稳定后增加到 4；
- 所有 records 标记当前 `policy_version`。

### 9.2 Process Phase

- 释放或停止 SGLang；
- Polar Adapter 转换 raw results；
- 三个 Processor 产生 TrainingReadyBatch 或 ResampleRequest；
- 如果需要补采，在训练前继续使用同一 policy。

### 9.3 Train Phase

- 两张 GPU 运行 Megatron TP2 BF16；
- 从 4K context、512 max output、小 prompt batch 开始；
- Slime Adapter 只消费 TrainingReadyBatch；
- 每个 policy iteration 只做配置允许的少量 update；
- 旧 rollout 不被无限复用。

### 9.4 Reload Phase

- 保存 checkpoint/checksum；
- 加载到 SGLang；
- 分配新 `policy_version`；
- 产生至少一条新 policy rollout，证明闭环成立。

## 10. 对照实验

### Baseline

```text
Source Adapter
→ minimal format validation
→ Slime Adapter
```

### Processed Pipeline

```text
Source Adapter
→ FailureClassifier
→ SignalFilter
→ PolicyConsistentGroupBuilder
→ Slime Adapter
```

### 指标

```text
invalid_infrastructure_to_trainer_rate
valid_group_completion_rate
resample_count
reward_variance
accepted_trainable_tokens
processing_latency
rollout_cost_per_ready_group
successful_optimizer_steps
checkpoint/reload success
held-out reward before/after training
```

能力提升不是唯一成功标准，但必须进行真实训练和独立评估，不能只做一个格式转换 demo。

## 11. 交付物

### 核心代码

- contracts 与 capability；
- Polar/JSONL Source Adapter；
- 三个 Processor；
- TrainingReadyBatch 与 ResampleRequest；
- Slime Trainer Adapter；
- pipeline 配置与 CLI；
- baseline/processed 两套配置。

### 测试

- core 无 Polar/Slime 依赖的单元测试；
- golden fixture roundtrip；
- capability 缺失；
- 三类 validity；
- reward variance；
- policy version 隔离；
- group selection 与 resample；
- Polar integration；
- Slime Sample contract；
- staged rollout/train/reload E2E。

### 证据

- Polar Calculator 和 Coding raw artifacts；
- canonical fixture；
- ProcessingReport；
- baseline vs processed 指标；
- optimizer/checkpoint/reload 记录；
- new policy rollout；
- held-out evaluation；
- 复现 runbook 与限制说明。

## 12. 完成定义

只有以下全部满足，项目才算完成：

1. Polar rollout 参考路径独立跑通并产生真实 fixture；
2. 核心模块不依赖 Polar/Slime；
3. 至少两个 Source Adapter 通过统一 contract；
4. 三个 Processor 可通过配置启用/关闭；
5. 基础设施失败不会静默作为 task failure 进入训练；
6. GRPO group 不混合 policy version；
7. group 不足能输出可执行的 ResampleRequest；
8. Slime Adapter 可重复转换 TrainingReadyBatch；
9. 双卡 staged loop 完成真实 GRPO update、checkpoint reload 和新 rollout；
10. baseline/processed 对照和故障注入可重复；
11. 文档明确区分 Reference Provider、Core 和 Reference Consumer；
12. 项目描述不超出实际证据。

## 13. 实施文档

- [项目范围](PROJECT_SCOPE.md)
- [Day 1–Day 7 实施索引](IMPLEMENTATION_PLAN.md)
- [Day 1：环境与版本基线](plans/day-01-environment.md)
- [Day 2：Polar Calculator rollout](plans/day-02-polar-smoke.md)
- [Day 3：Polar Coding/SWE rollout](plans/day-03-swegym-rollout.md)
- [Day 4：数据契约与 Source Adapter](plans/day-04-data-contract.md)
- [Day 5：可插拔 Processor](plans/day-05-processors.md)
- [Day 6：自定义双卡 Agentic RL](plans/day-06-custom-grpo.md)
- [Day 7：对照实验与包装](plans/day-07-packaging.md)

## 14. 官方参考

- [Polar / ProRL-Agent-Server](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server)
- [Polar Calculator](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server/tree/stable/examples/calculator)
- [Polar SWE-Gym + Slime GRPO Reference](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server/tree/stable/examples/swegym_slime_grpo)
- [Slime](https://github.com/THUDM/slime)
- [Agent-R1](https://github.com/AgentR1/Agent-R1)
- [PRIME-RL](https://github.com/PrimeIntellect-ai/prime-rl)
- [RAGEN-2](https://arxiv.org/abs/2604.06268)
- [rStar](https://github.com/microsoft/rstar)
