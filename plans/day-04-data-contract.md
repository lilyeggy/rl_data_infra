# Day 4：Producer-Agnostic 数据契约与 Source Adapter

> 从本阶段开始实现项目核心。核心代码不能 import Polar 或 Slime。

## 1. 阶段目标

1. 定义最小、版本化的 rollout 数据契约；
2. 实现 capability 声明与前置条件检查；
3. 实现 `PolarSourceAdapter` 和 `JsonlSourceAdapter`；
4. 确保相同 fixture 经不同入口得到相同 canonical record；
5. 建立不可伪造训练字段的 adapter 规则。

## 2. 目录边界

```text
src/
├── contracts/
│   ├── rollout_record.py
│   ├── rollout_batch.py
│   ├── training_batch.py
│   ├── resample_request.py
│   └── capabilities.py
├── sources/
│   ├── base.py
│   ├── polar.py
│   └── jsonl.py
└── errors.py
```

`contracts/` 不得依赖任何上游/下游框架。`sources/polar.py` 是可选 integration package 或 extra。

## 3. RolloutRecord v1

### Identity

```text
trajectory_id
task_id
group_id
policy_version
source_type
source_record_id
```

### Training Payload

```text
token_ids
action_mask or loss_mask
old_logprobs
reward
```

字段可以按 capability 缺失，但 Adapter 必须明确声明，不能补造。

### Execution Status

```text
rollout_status
termination_reason
runtime_status
harness_status
model_backend_status
verifier_status
```

### Evidence/Metadata

```text
verifier_evidence_ref
source_payload_ref
model/tokenizer revision
timestamps
opaque metadata
```

## 4. Capability

定义 enum 和校验器：

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

Adapter 返回 `AdapterResult(records, capabilities, warnings, errors)`。Processor/Trainer Adapter 在运行前比较 required capabilities。

## 5. Source Adapter Protocol

```python
class SourceAdapter(Protocol):
    name: str
    version: str

    def capabilities(self) -> set[Capability]: ...
    def convert(self, payload: object) -> AdapterResult: ...
```

转换要求：

- 纯函数或在相同输入/版本下确定性；
- 保留 source payload checksum；
- 不覆盖 raw source；
- 不猜测未知 status；
- 不将 text 重新 tokenize 后冒充 sampled token IDs；
- 不将缺失 old logprobs 填零；
- warnings/errors 可结构化查询。

## 6. PolarSourceAdapter

基于 Day 2–3 fixture 做显式字段映射。所有 Polar 特有字段只存在于该 Adapter；canonical contract 不出现 Gateway 内部对象。

对 success、valid failure、invalid infrastructure fixture 做 golden tests。未知 Polar 字段可以保存在 opaque source metadata，但不能泄漏进核心决策逻辑。

## 7. JsonlSourceAdapter

定义一个简单、公开的 JSONL interchange format。将 Day 3 的 canonical record 导出为 JSONL 后重新读入，结果除 source envelope 外应等价。

它同时用于：

- core unit tests；
- 无 Polar 环境下的 demo；
- 保存训练前 batch fixture；
- 未来接入其他 Producer 的最低成本入口。

## 8. 校验与错误语义

区分：

```text
ADAPTER_ERROR          源格式无法解析
CAPABILITY_MISSING     目标处理器所需字段不存在
CONTRACT_INVALID       canonical 字段内部不一致
SOURCE_WARNING         可保留但需告警
```

`CAPABILITY_MISSING` 不等同于 trajectory task failure。

## 9. 测试

- schema/model validation；
- success/failure/invalid fixture；
- JSONL roundtrip；
- Polar → canonical → JSONL → canonical 等价；
- missing token/logprob/policy/reward；
- checksum 稳定；
- opaque metadata roundtrip；
- source payload 不被覆盖；
- core package 在未安装 Polar/Slime 时导入和测试成功。

## 10. 产物

```text
src/contracts/
src/sources/
tests/contracts/
tests/sources/
docs/data-contract.md
examples/jsonl/
artifacts/day-04/
```

## 11. 验收门

- [ ] RolloutRecord/RolloutBatch/TrainingReadyBatch v1 完成；
- [ ] capability 前置检查完成；
- [ ] Polar 和 JSONL 两个 Adapter 通过测试；
- [ ] Adapter 不伪造训练字段；
- [ ] golden roundtrip checksum 稳定；
- [ ] 核心测试无需 Polar/Slime；
- [ ] 数据契约文档包含 required/optional/error 语义。

## 12. 执行记录

```text
状态：IN_PROGRESS（contract、capability、JSONL Adapter、PolarSourceAdapter 已完成；task-level success/coding golden coverage 等待 Day 3 fixture）
schema版本：rollout-record/v1、rollout-batch/v1、training-ready-batch/v1、resample-request/v1、rollout-jsonl/v1
capability数量：8
Polar fixture覆盖：Day 2 execution-complete VALID_FAILURE 与 synthetic runtime fault 已覆盖；Adapter 不按目录名改写 reward
JSONL roundtrip：已完成，除 source envelope 外语义等价
core独立测试：已完成，不 import Polar/Slime
未解决问题：task-level success、policy/group 缺失；Day 3 multi-turn/verifier mapping 待真实 artifact
```
