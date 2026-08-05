# Day 5：可插拔 Processor、有效性分类与 GRPO Group Builder

> 本阶段只实现三个 Processor。新增功能必须证明直接服务于“生成有效、同-policy、有训练信号的 GRPO batch”。

## 1. 阶段目标

1. 定义 `RolloutProcessor` Protocol 和简单 registry；
2. 实现 FailureClassifier；
3. 实现 SignalFilter；
4. 实现 PolicyConsistentGroupBuilder；
5. 输出 TrainingReadyBatch、RejectedRecord 和 ResampleRequest；
6. 允许通过配置启用/关闭每个 Processor。

## 2. Processor Protocol

```python
class RolloutProcessor(Protocol):
    name: str
    version: str
    required_capabilities: set[Capability]

    def process(self, batch: RolloutBatch) -> ProcessResult: ...
```

`ProcessResult` 至少包含：

```text
accepted_records
rejected_records
training_ready_batches
resample_requests
metrics
warnings
input/output checksums
```

## 3. FailureClassifier

固定三类输出：

```text
VALID_SUCCESS
VALID_FAILURE
INVALID_INFRASTRUCTURE
```

第一版规则：

- verifier 正常完成且 reward 成功 → `VALID_SUCCESS`；
- verifier 正常完成且任务未通过 → `VALID_FAILURE`；
- runtime/Harness/model backend/verifier 非任务性 crash/timeout → `INVALID_INFRASTRUCTURE`；
- trace 或必要训练字段不完整 → `INVALID_INFRASTRUCTURE` 或 contract rejection，按 reason code 区分。

输出 versioned reason code 和所依据字段，不使用不可解释的总分。

## 4. SignalFilter

按 `task_id + group_id + policy_version` 聚合：

- valid record 数量；
- reward unique count/variance；
- trainable token 数量；
- capability completeness；
- success/failure/invalid 分布。

默认策略：

- valid 数量不足：不训练，交给 GroupBuilder 请求补采；
- 全部 invalid：拒绝 group；
- reward 无差异：标记 `NO_RELATIVE_SIGNAL`，第一版默认不进入 GRPO；
- 有正负 reward 且字段完整：允许进入 GroupBuilder。

配置必须允许保留 no-variance group 做分析，但不能静默改变 GRPO 语义。

## 5. PolicyConsistentGroupBuilder

### Group Key

```text
task_id + group_id + policy_version
```

### 行为

1. 移除/隔离 `INVALID_INFRASTRUCTURE`；
2. 不跨 policy version 合并；
3. valid 数量等于目标 group size 时输出 TrainingReadyBatch；
4. valid 数量不足时输出 ResampleRequest；
5. valid 数量超出时使用配置的 deterministic selection；
6. 记录每条记录被接受/拒绝/延迟的原因。

### ResampleRequest

必须包含：

```text
task_id
group_id
policy_version
required_count
reason
sampling constraints
source hint optional
```

核心模块不执行补采；example controller 可以把请求交回 Polar 或其他 Producer。

## 6. 配置

```yaml
processors:
  - name: failure_classifier
    version: v1
  - name: signal_filter
    require_reward_variance: true
  - name: policy_consistent_group_builder
    group_size: 2
    selection: stable_first
```

第一版不需要 setuptools entry points、远程插件市场或独立服务。Protocol + registry 足够证明可插拔性。

## 7. 基线模式

提供两套 pipeline：

```text
baseline: minimal_contract_check → trainer_adapter
processed: failure_classifier → signal_filter → group_builder → trainer_adapter
```

两者使用相同 source fixture 和训练配置，为 Day 7 对照做准备。

## 8. 故障测试

至少覆盖：

- task success；
- task valid failure；
- runtime prep failed；
- Harness crash；
- model backend failed；
- verifier timeout；
- trace incomplete；
- group 混合 policy；
- group 少 1/多 1 条；
- reward 全同；
- reward 有差异；
- capability 缺失；
- duplicate trajectory。

## 9. 指标

```text
valid_success/failure/invalid_count
invalid_to_trainer_rate
ready_group_count
incomplete_group_count
no_signal_group_count
resample_request_count
accepted_trainable_tokens
processing_latency
```

## 10. 产物

```text
src/processors/
src/pipeline.py
src/registry.py
tests/processors/
configs/pipelines/baseline.yaml
configs/pipelines/processed.yaml
docs/processors.md
artifacts/day-05/
```

## 11. 验收门

- [ ] 三个 Processor 均通过统一 Protocol；
- [ ] 配置可启用/关闭 Processor；
- [ ] 三类 validity 与 reason code 测试通过；
- [ ] 同 group 不混合 policy version；
- [ ] 不足 group 产生正确 ResampleRequest；
- [ ] reward 无差异 group 被显式处理；
- [ ] baseline/processed 两套输出可重复；
- [ ] 核心没有调用 Polar 或 Slime。

## 12. 执行记录

```text
状态：NOT_STARTED
Processor版本：
规则数量：
fixture数量：
ready/incomplete/no-signal groups：
resample结果：
未解决问题：
```
