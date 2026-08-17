# Day 4 已实现代码深度讲解：从不可信 Rollout Artifact 到受约束的训练数据

> 历史基础说明（2026-08-12）：本文准确描述已实现的 `RolloutRecord v1` 训练视图。它不再定义项目的 universal contract；新 `TraceEvent/AgentEpisode` 契约见 [`../data-contract.md`](../data-contract.md)。

> 本文只解释目前已经实现的代码，不提前讲尚未实现的 Day 5 Processor、Day 6 Trainer Adapter 或训练实验。
> 对应核心目录：`src/errors.py`、`src/contracts/`、`src/sources/` 以及相应测试。
> 阅读目标：读完后，你不仅知道每个类“是干什么的”，还应能解释它维护了什么不变量、阻止了什么训练错误、为什么不能用更省事的实现，以及它仍然缺少什么。

## 1. 先从真正的问题开始：上游给我们的不是“训练数据”

Polar 产生的是 rollout artifact。它记录一次 Agent 执行过程中发生了什么，例如：

```text
任务是什么
模型收到了哪些 token
模型生成了哪些 token
调用了什么工具
Harness 是否结束
Verifier 是否运行
Verifier 给出了什么结果
```

Trainer 需要的却不是“发生过什么”的任意记录，而是一组满足数学和系统约束的数据：

```text
token 序列真实来自采样过程
mask 与 token 精确对齐
reward 是可信 evaluator 结果
同组样本属于同一个 task/group
同组样本来自同一个 policy version
无效基础设施故障没有伪装成 reward=0
```

两者之间存在一个语义鸿沟：

```text
Artifact 是证据
Canonical Record 是受约束的事实表达
TrainingReadyBatch 是通过训练前置条件的数据
```

Day 4 实现的就是中间两层的结构基础。它还没有完成任务有效性分类和 group signal 判断——那属于 Day 5。

### 1.1 为什么不能让 Trainer 直接读取 Polar JSON

如果 Trainer 直接读取 Polar JSON，会产生四种耦合：

1. **字段耦合**：Polar 改一个 JSON path，Trainer 就失效；
2. **状态耦合**：Trainer 必须理解 runtime、Harness、verifier 的全部私有状态；
3. **依赖耦合**：核心测试必须安装 Polar；
4. **语义耦合**：其他 Producer 必须模仿 Polar，而不是实现一个稳定公共契约。

当前设计把依赖方向变成：

```text
Polar 私有 schema
        │
        ▼
PolarSourceAdapter
        │
        ▼
公共 RolloutRecord
        │
        ▼
公共 Processor / Trainer Adapter
```

公共核心完全不知道 Gateway node、Polar session 对象或上游 Python 类。

## 2. 已实现模块的依赖结构

核心代码的真实依赖方向如下：

```text
src/errors.py
    ▲
    │
src/contracts/_json.py
    ▲
    │
src/contracts/rollout_record.py ◄── src/contracts/capabilities.py
    ▲                                      ▲
    │                                      │
    ├── src/contracts/rollout_batch.py ─────┘
    ├── src/contracts/training_batch.py ────┘
    └── src/contracts/resample_request.py
                    ▲
                    │
            src/sources/base.py
               ▲          ▲
               │          │
     src/sources/jsonl.py  src/sources/polar.py
```

这里有两个重要设计点：

- `contracts/` 不依赖任何 Source Adapter；
- JSONL 与 Polar Adapter 依赖同一公共 contract，但互相不依赖。

这意味着删除 Polar Adapter 后，JSONL 和核心 contract 仍可完整工作；未来增加其他 Producer 也无需修改 `RolloutRecord` 的运行机制。

## 3. 第一层：错误不是一段日志，而是公共 API

代码文件：[`src/errors.py`](../../src/errors.py)

### 3.1 为什么错误分类属于数据契约

数据流水线遇到“不能训练”时，原因并不相同：

```text
JSON 语法坏了
canonical 字段互相矛盾
字段合法但不满足某个消费者要求
数据可以保留，只是存在语义警告
```

如果这些情况全部抛 `ValueError("bad data")`，系统将无法：

- 统计每类拒绝原因；
- 决定是否需要重新采样；
- 判断是修复 Producer、Adapter 还是 Trainer 配置；
- 在实验报告中证明 infrastructure failure 没有进入训练。

因此实现定义了稳定的 `ErrorCode`：

```python
class ErrorCode(str, Enum):
    ADAPTER_ERROR = "ADAPTER_ERROR"
    CAPABILITY_MISSING = "CAPABILITY_MISSING"
    CONTRACT_INVALID = "CONTRACT_INVALID"
    SOURCE_WARNING = "SOURCE_WARNING"
```

### 3.2 四个 code 的边界

#### `ADAPTER_ERROR`

表示 Adapter 无法理解源输入，例如：

- JSONL 不是 UTF-8；
- 某行不是合法 JSON；
- Polar fixture 路径不是目录；
- JSONL envelope 结构不受支持。

它描述源表示转换失败，不描述模型任务失败。

#### `CONTRACT_INVALID`

表示已经尝试形成 canonical record，但字段违反内部不变量，例如：

- mask 长度与 token 不同；
- reward 是 NaN；
- 同时给出 `action_mask` 和 `loss_mask`；
- manifest 声明的 summary checksum 与实际字节不一致。

这种数据不应继续传播，因为下游无法确定应该相信哪个字段。

#### `CAPABILITY_MISSING`

表示对象本身可以合法表示，但当前消费者需要的能力不存在。例如一条 fault record 合法地保存：

```text
token_ids=None
reward=None
```

它可以被 FailureClassifier 观察，却不能被要求 `TOKEN_IDS + REWARD` 的 Trainer 消费。

#### `SOURCE_WARNING`

表示记录可以保留，但存在需要人工或下游注意的事实。例如目录名是 `calculator_success`，实际 reward 却是 `0.0`。Adapter 不修改事实，而是发 warning。

### 3.3 为什么同时需要异常和 `AdapterIssue`

内部构造对象时，违反不变量应立即抛异常：

```text
ContractValidationError
CapabilityMissingError
```

但 Adapter 是一个批量边界，它不能遇到一行坏数据就丢掉所有其他行。因此 Adapter 将异常转换成结构化 `AdapterIssue`：

```text
code                 机器判断类型
message              人类可读原因
source_record_id     定位原记录
field                定位字段
details              附加结构化上下文
```

这构成了两层错误处理：

```text
对象内部：fail fast，抛异常
Adapter 边界：捕获并结构化，返回结果
```

### 3.4 `AdapterIssue` 为什么也不可变

它使用 `frozen=True`，并将 `details` 转成 `MappingProxyType`。原因是拒绝原因也是实验 lineage 的一部分。如果处理完成后某段代码还能修改 `details["missing"]`，同一个运行结果会变得不可审计。

## 4. 第二层：确定性 JSON 是所有 checksum 的地基

代码文件：[`src/contracts/_json.py`](../../src/contracts/_json.py)

### 4.1 普通 `json.dumps` 为什么不够

以下两个 Python dict 语义相同：

```python
{"b": 2, "a": 1}
{"a": 1, "b": 2}
```

如果序列化保留插入顺序，它们可能产生不同字节。若 checksum 直接基于这些字节，语义相同的 record 会得到不同身份。

所以实现统一使用：

```python
json.dumps(
    value,
    ensure_ascii=False,
    allow_nan=False,
    sort_keys=True,
    separators=(",", ":"),
)
```

由此建立：

```text
相同 canonical JSON 语义
→ 相同 canonical bytes
→ 相同 SHA256
```

### 4.2 `freeze_json` 不是普通类型检查

它递归执行三项工作。

#### 验证 JSON 可表示性

允许：

```text
null / string / bool / int / finite float / object / array
```

拒绝：

```text
NaN / Infinity / bytes / set / datetime / 自定义类 / 非字符串 object key
```

这防止 `to_dict()` 到最后一步才发现 metadata 不能序列化。

#### 冻结嵌套容器

```text
dict → MappingProxyType
list → tuple
```

仅仅把 dataclass 声明为 `frozen=True` 并不够，因为 frozen 只禁止重新赋值：

```python
record.opaque_metadata = other  # frozen 会拒绝
```

但如果内部仍是普通 dict，下面仍可能成功：

```python
record.opaque_metadata["policy"] = "changed"
```

递归冻结堵住了这个漏洞。

#### 固定 object 遍历顺序

`freeze_json` 按 key 排序构造冻结 mapping。即使调用者传入不同插入顺序，内部观察也保持稳定。

### 4.3 `thaw_json` 为什么存在

冻结容器适合内存安全，却不适合直接交给所有 JSON encoder。因此序列化前执行反向转换：

```text
MappingProxyType → dict
tuple → list
```

它不改变标量值，只恢复标准 JSON 容器。

### 4.4 checksum 的安全边界

当前 SHA256 用于内容寻址和 lineage：

```text
“这批数据的精确内容是什么？”
“处理前后是否发生变化？”
“batch 是否来自所声明的 source batch？”
```

它不是密码学签名，不能回答：

```text
“是谁生成了数据？”
“拥有仓库写权限的人是否同时改了内容和 checksum？”
```

### 4.5 当前实现的两个细微限制

第一，字符串没有做 Unicode normalization。视觉相同但编码组合不同的字符串可能得到不同 checksum。

第二，时间戳会验证时区并用于先后比较，但保留原字符串：

```text
2026-08-10T00:00:00Z
2026-08-10T08:00:00+08:00
```

两者表示同一时刻，但 record checksum 不同。这是 v1 的“保存原始表达”选择，并非时间语义归一化。

## 5. 第三层：`RolloutRecord` 是事实容器，不是成功判定器

代码文件：[`src/contracts/rollout_record.py`](../../src/contracts/rollout_record.py)

`RolloutRecord` 是当前实现最核心的对象。理解它的关键是：

> 它保证字段内部自洽，但不保证这条 rollout 有效，更不保证它可以训练。

### 5.1 为什么使用 `frozen=True, slots=True, kw_only=True`

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class RolloutRecord:
    ...
```

三个选项分别服务于不同目标：

- `frozen=True`：创建后不可重新赋值，保护 lineage；
- `slots=True`：禁止随手挂任意新属性，也减少大量记录的内存开销；
- `kw_only=True`：构造时必须写字段名，避免几十个相似位置参数错位。

### 5.2 字段被分成四个语义域

#### A. Identity

```text
trajectory_id
task_id
group_id
policy_version
source_type
source_record_id
```

这里必须区分两组身份。

第一组描述轨迹语义：

```text
trajectory_id / task_id / group_id / policy_version
```

第二组描述当前直接来源：

```text
source_type / source_record_id
```

同一条轨迹从 Polar 导出到 JSONL 再读入，第一组不变，第二组会变。

#### B. Training payload

```text
token_ids
prompt_token_count
action_mask or loss_mask
old_logprobs
reward
```

这些字段决定数据是否具备某类训练能力，但允许为 `None`。允许缺失不是放松标准，而是为了准确表示无 trace、无 reward 或上游没提供 logprob 的情况。

#### C. Execution status

```text
rollout_status
termination_reason
runtime_status
harness_status
model_backend_status
verifier_status
```

它们是 Day 5 validity classification 的证据输入。

#### D. Evidence and lineage

```text
verifier_evidence_ref
source_payload_ref
source_payload_sha256
model_id / model_revision / tokenizer_revision
started_at / ended_at
tool_events
opaque_metadata
schema_version
```

它们让结论可以回到原始 artifact 复查。

## 6. `RolloutRecord.__post_init__` 的完整逻辑

`__post_init__` 不是一堆随意的防御代码。它从最基础身份逐步建立更强不变量。

### 6.1 第一步：验证必须身份

以下字段必须是非空字符串：

```text
trajectory_id
task_id
source_type
source_record_id
```

注意 `group_id` 和 `policy_version` 不在这里，因为 record 层需要表示还不能训练的数据。

如果连 `task_id` 都没有，后续不能分组、补采或审计，所以它不是 optional。

### 6.2 第二步：锁定 schema 和 enum

只接受：

```text
schema_version = rollout-record/v1
```

status 必须已经是对应 Enum，而不是任意字符串。`from_dict()` 负责把 JSON 字符串转换成 Enum；直接构造 Python 对象时，调用者必须显式遵守类型。

这避免状态中出现：

```text
"Success"
"success"
"SUCCEEDED"
"done"
```

等不可控同义词。

### 6.3 第三步：规范化 token、mask、logprob 容器

传入 list 或 tuple 都被转成 tuple。验证规则：

#### `token_ids`

- 每个元素必须是整数；
- `bool` 被拒绝，虽然 Python 中 `bool` 是 `int` 子类；
- token ID 必须非负。

#### mask

- 只能为 0 或 1；
- 不能同时出现 `action_mask` 和 `loss_mask`；
- mask 存在时 token 必须存在；
- mask 与完整 token 序列必须等长。

#### `old_logprobs`

- 必须是有限数；
- 没有 token 时不能存在；
- 长度必须满足允许的某一种对齐。

### 6.4 `old_logprobs` 长度约束的数学表达

设：

```text
N = len(token_ids)
P = prompt_token_count
M = sum(mask)
L = len(old_logprobs)
```

当前 contract 接受：

```text
L ∈ {N, N-P, M}
```

对应：

```text
N     全 token 对齐
N-P   response 段对齐
M     只与 trainable token 对齐
```

为什么不强制唯一形式？因为 contract 要接纳不同 Producer 的真实表达，而不应在 Source Adapter 中伪造或丢弃信息。

为什么这种宽容仍然安全？因为长度必须落在明确集合中；后续具体 Trainer Adapter 仍可要求更窄的对齐类型。

### 6.5 一个容易忽视的歧义

如果恰好 `N-P == M`，仅靠长度无法区分 response-aligned 与 trainable-only logprobs。当前 v1 保存了数据，但没有单独的 `logprob_alignment` enum。

对当前 Polar Day 2 fixture，response 的 99 个 token 全部 trainable，因此二者长度相同。我们依靠 Polar builder 的来源语义和 opaque provenance 判断它是 response-aligned。

这是一个真实的 v1 设计限制。未来 Trainer Adapter 若必须区分，可以：

- 在 contract v2 加显式 alignment；或
- 在特定 Adapter metadata 中读取可靠 provenance，并在 Trainer Adapter 入口显式校验。

不能仅凭长度猜。

### 6.6 `prompt_token_count` 为什么是 Day 2 后新增的重要字段

假设 response 包含 100 个 token，但其中只有 60 个参与 loss：

```text
response_length = 100
trainable_token_count = 60
```

如果用 `sum(mask)` 推断 response length，会把 40 个 response 内的非训练 token 误当成 prompt。Slime Sample 需要的 response boundary 与 loss mask 是两个不同概念。

所以 contract 明确保存：

```text
prompt_token_count
```

并验证：

```text
0 <= prompt_token_count <= len(token_ids)
```

### 6.7 reward 校验为什么拒绝 NaN 和 Infinity

NaN 有一个危险性质：

```python
float("nan") != float("nan")
```

它会破坏排序、均值、variance、checksum JSON 和下游 loss。Infinity 同样会污染归一化。因此 record 层直接拒绝非有限 reward。

### 6.8 时间戳校验维护什么

- 必须是 ISO-8601；
- 必须包含 UTC offset；
- `ended_at >= started_at`。

没有 offset 的本地时间无法跨服务器比较，因而拒绝。

### 6.9 tool events 与 opaque metadata

`tool_events` 必须是 object 数组，每个 object 递归冻结。

`opaque_metadata` 必须是 object。其设计原则是：

```text
Producer 特有但值得保留的信息
→ opaque_metadata

所有 Producer 都应理解且下游要依赖的信息
→ canonical 顶层字段
```

这防止 canonical schema 无限吸收 Polar 私有细节。

## 7. `RolloutRecord` 的三个“身份视图”

### 7.1 `to_dict()`：完整可持久化内容

它包括当前 source envelope，因此适用于：

- 完整记录 checksum；
- JSONL 导出；
- batch checksum；
- 审计当前直接来源。

### 7.2 `semantic_dict()`：排除运输外壳

它删除：

```text
source_type
source_record_id
source_payload_ref
source_payload_sha256
```

保留其他轨迹内容。

因此可以验证：

```text
Polar record
→ JSONL export
→ JSONL import

semantic_dict 前后相同
```

### 7.3 `with_source_envelope()`：不可变替换来源

它使用 `dataclasses.replace()` 返回新对象，不修改原对象。

这很重要：导出再导入不应改变内存中的原始 Polar record，也不能让旧 checksum 在对象背后漂移。

### 7.4 当前 record checksum 的可移植性限制

`checksum` 基于完整 `to_dict()`，而 `source_payload_ref` 可能是绝对路径。因此同一语义 record 在两台机器上：

```text
/data/a/summary.json
/mnt/b/summary.json
```

可能得到不同 record checksum。

这符合当前“完整 lineage 身份”的含义，但不适合用作跨机器纯语义 ID。跨入口等价测试使用 `semantic_dict()`，不是强求完整 checksum 相同。

未来如需要，可以同时引入：

```text
content_checksum   完整来源与内容
semantic_checksum  排除运输 envelope
```

当前代码没有伪装这两者相同。

## 8. 第四层：Capability 把 Optional 字段变成可执行前置条件

代码文件：[`src/contracts/capabilities.py`](../../src/contracts/capabilities.py)

### 8.1 为什么仅用 `Optional` 不够

如果一个 Trainer 直接访问：

```python
record.policy_version
```

它只能在运行到这一条时才发现值是 `None`。这会造成晚失败，甚至前几条已经被处理。

Capability 提供批量运行前检查：

```text
源数据实际提供什么
vs.
消费者运行需要什么
```

### 8.2 record capability 是字段事实的投影

定义函数可以形式化为：

```text
C(record) = {
    TOKEN_IDS           if token_ids is not None,
    ACTION_MASK         if action_mask or loss_mask exists,
    OLD_LOGPROBS        if old_logprobs is not None,
    REWARD              if reward is not None,
    GROUP_ID            if group_id is not None,
    POLICY_VERSION      if policy_version is not None,
    VERIFIER_EVIDENCE   if verifier_evidence_ref is not None,
    TOOL_EVENTS         if tool_events is not None
}
```

Adapter 不传一个布尔声明给 record；系统从真实字段计算。

### 8.3 batch capability 是集合交

对 records `r1...rn`：

```text
C(batch) = C(r1) ∩ C(r2) ∩ ... ∩ C(rn)
```

而不是集合并。

原因是 batch capability 的语义是：

> 消费者可以假设每一条记录都具备这些能力。

如果取并集，它只能表达“至少某条有”，不能作为安全前置条件。

### 8.4 空 batch 为什么返回空 capability

数学上空集合交集有时会定义为全集，但工程语义中，空 batch 不能保证能消费任何训练字段。返回空集更保守，也避免空输入通过 capability gate 后才失败。

### 8.5 默认训练 capability 为什么没有 `OLD_LOGPROBS`

默认集合是：

```text
TOKEN_IDS
ACTION_MASK
REWARD
GROUP_ID
POLICY_VERSION
```

它表达 core 层的最低结构训练资格。

`OLD_LOGPROBS` 是否必需取决于具体 loss 和 Trainer Adapter。如果某个 Slime 配置需要 importance ratio 或 PPO-style clipping，它必须在自己的 required set 中加入 `OLD_LOGPROBS`。

这种设计防止 core contract 假装所有 Trainer 都需要完全相同的输入。

### 8.6 format capability 与 observed capability

`adapter.capabilities()` 表示格式/实现理论上可以承载什么。

`AdapterResult.capabilities` 表示本次具体 records 共同实际具有什么。

举例：JSONL schema 可以保存 policy version，但一个具体 JSONL 文件可以没有它。不能因为格式支持，就声称这批数据拥有 `POLICY_VERSION`。

## 9. 第五层：三个批次对象表达三个不同生命周期

### 9.1 `RolloutBatch`：Processor 的原始 canonical 输入

代码文件：[`src/contracts/rollout_batch.py`](../../src/contracts/rollout_batch.py)

它只维护四类不变量：

```text
schema 版本正确
adapter 名称/版本非空
records 固定为 tuple
trajectory_id 不重复
```

它故意不检查：

- 所有 record task 相同；
- 所有 reward 有 variance；
- policy version 一致；
- rollout validity。

因为一批 Producer 输入天然可以混合多个 task 和 group，Processor 正是要处理这些差异。

#### duplicate 为什么在这里拒绝

如果同一 trajectory 被重复包含：

- reward 统计被加权两次；
- group size 看起来被错误补齐；
- Trainer 对同一采样做重复更新；
- 实验样本量虚高。

因此重复 identity 不是 warning，而是 contract invalid。

#### 当前重复检测复杂度

实现用 `list.count()` 查找重复，最坏是 O(n²)。对当前小规模 fixture 和 pilot batch 足够直观，但大批量生产输入应改成单次计数或 set 扫描。

这是性能优化点，不是当前语义错误。

### 9.2 `TrainingReadyBatch`：结构上可交给 Trainer Adapter

代码文件：[`src/contracts/training_batch.py`](../../src/contracts/training_batch.py)

可以把其构造条件写成：

```text
records 非空
AND trajectory_id 全部唯一
AND 对每条 r：
      r.task_id == batch.task_id
      r.group_id == batch.group_id
      r.policy_version == batch.policy_version
AND required_capabilities ⊆ C(r)
AND source_batch_checksum 合法
AND processing_versions 非空且合法
```

只有全部成立，构造才成功。

#### 为什么 task、group、policy 三者都要一致

`task_id` 相同保证同组样本在解决同一问题。

`group_id` 相同保证这些采样被上游/控制器定义为同一个相对比较单元。

`policy_version` 相同保证它们来自同一采样分布。

只检查其中两个都不够。

#### 为什么保存 `processing_versions`

未来 Day 5 规则可能更新：

```text
failure-classifier/v1
signal-filter/v2
group-builder/v1
```

训练结果必须能追溯使用了哪套数据决策规则。否则 checkpoint 指标变化无法归因。

#### 为什么 batch ID 来自 checksum

```python
batch_id = "training-" + checksum[:16]
```

相同内容和处理版本得到相同 ID，有利于：

- 避免同一 batch 被意外重复训练；
- 日志、checkpoint、manifest 之间交叉引用；
- crash recovery 时识别已经消费的 batch。

注意 records 顺序参与 checksum。因此未来 GroupBuilder 必须使用确定性排序；同一组 record 的不同排列会得到不同 batch ID。

#### 它为什么还不能证明“真的 training-ready”

当前类只维护结构 readiness，没有检查：

- record 是否 `VALID_SUCCESS/VALID_FAILURE`；
- infrastructure-invalid 是否已剔除；
- group size 是否等于配置值；
- reward 是否有 variance；
- trainable token 数是否超过阈值。

这些由尚未实现的 Day 5 Processor 负责。因此当前不应绕过 Day 5，手工把合法 record 塞进 `TrainingReadyBatch` 后就声称完成数据 pipeline。

### 9.3 `ResampleRequest`：表达缺口，不执行副作用

代码文件：[`src/contracts/resample_request.py`](../../src/contracts/resample_request.py)

它保存：

```text
task_id
group_id
policy_version
required_count
reason
sampling_constraints
source_hint
```

#### 为什么 `required_count` 必须为正整数

零条不需要请求；负数没有语义；`bool` 虽是 Python int 子类，也必须拒绝。

#### 为什么 request ID 也由 checksum 生成

同一个缺口重复计算会得到同一个 request ID，方便上游 controller 做幂等去重，避免无限重复提交。

#### 为什么 request 不直接调用 Polar

一旦核心对象负责网络请求，它就会：

- 依赖具体 Producer；
- 引入重试、认证、服务发现和副作用；
- 无法进行纯单元测试；
- 难以保证同 policy 补采发生在更新前。

所以 core 只表达需求，controller 负责执行。

## 10. 第六层：`AdapterResult` 是不可信输入与 canonical batch 的闸门

代码文件：[`src/sources/base.py`](../../src/sources/base.py)

### 10.1 为什么 Adapter 不直接返回 `RolloutBatch`

批量输入可能部分成功：

```text
line 1 合法
line 2 JSON 损坏
line 3 合法
```

如果只返回 batch：

- 要么因 line 2 丢掉 line 1 和 3 的诊断价值；
- 要么静默跳过 line 2，让不完整数据继续训练。

`AdapterResult` 同时返回 records 和 errors，使系统既能观察合法部分，又不会误把部分成功当全成功。

### 10.2 `ok` 的准确语义

```python
ok = not errors
```

warnings 不影响 `ok`。所以：

```text
有 warning、无 error → 可以转 batch
有任何 error        → 不可以转 batch
```

### 10.3 capability 声明的二次防线

`AdapterResult.__post_init__` 重新计算 records 的共同 capabilities，并要求：

```text
declared_capabilities ⊆ actual_common_capabilities
```

即使某个 Adapter 实现者写错了声明，结果对象也会拒绝“记录没有但 Adapter 声称有”的能力。

为什么允许声明为实际能力的子集，而不是必须完全相等？因为 Adapter 可以保守声明，不承诺某个它暂时不愿对下游保证的能力。

### 10.4 `to_batch()` 为什么遇到 error 必须整体拒绝

它抛 `AdapterConversionError`，而不是自动使用成功 records。

这是一个重要安全选择：

> 允许部分解析用于诊断，不允许部分解析在无显式决策的情况下成为 Processor 输入。

若未来确实需要 quarantine 模式，应该有显式 policy 和报告，而不是在 Adapter 中静默跳过。

### 10.5 `SourceAdapter` Protocol 的解耦意义

协议只要求：

```python
name
version
capabilities()
convert(payload)
```

它是结构类型，不要求继承公共基类。第三方 Adapter 只要实现相同接口，就能通过 `runtime_checkable` protocol 检查。

这比强制继承更适合插件生态，也避免核心类层次与某个框架绑定。

## 11. JSONL Adapter：最简单入口其实定义了重要的公共语义

代码文件：[`src/sources/jsonl.py`](../../src/sources/jsonl.py)

### 11.1 JSONL envelope 为什么再包一层 record

每行格式是：

```json
{
  "schema_version": "rollout-jsonl/v1",
  "record": {
    "schema_version": "rollout-record/v1"
  }
}
```

外层版本描述运输格式，内层版本描述 canonical object。未来可以更新 JSONL 压缩、附件引用等运输规则，而不必同步改变 record schema。

### 11.2 `dumps_jsonl()` 的确定性

对每条 record：

```text
record.to_dict()
→ 放入版本化 envelope
→ canonical_json_bytes
→ 一行 UTF-8 文本
```

多条记录按输入顺序排列，非空输出总以换行结束。

末尾换行是可重复 artifact 的一部分，可以避免不同写文件方式产生细微 diff。

### 11.3 `convert()` 与 `convert_file()` 的区别

`convert()` 接收内存中的 `str` 或 UTF-8 bytes，不知道稳定文件路径，因此 `source_payload_ref=None`。

`convert_file()` 读取文件并记录：

```text
<path>#line:<n>
```

两者最终进入同一个 `_convert_text()`，避免产生两套解析语义。

### 11.4 单行转换的精确控制流

对每个非空 raw line：

```text
1. 生成 source_record_id = line:N
2. json.loads(raw_line)
3. 验证 envelope 只含 schema_version 和 record
4. 验证 JSONL schema version
5. RolloutRecord.from_dict(record)
6. 生成当前 JSONL source envelope
7. 检查调用者要求的 capabilities
8. 保存 record 或结构化错误
```

### 11.5 为什么 source line checksum 不包含换行

实现对 `raw_line.encode("utf-8")` 计算 SHA256。`splitlines()` 已移除行终止符。

因此 checksum 标识该行 JSON 内容的精确字节，不受 `\n` 与 `\r\n` 文件行尾差异影响。

但要注意，它不是整个 JSONL 文件 checksum；每条 record 只引用自己的行。

### 11.6 Source envelope 为什么必须替换

假设 record 最初来自：

```text
source_type=polar
source_payload_ref=/server/summary.json
```

导出 JSONL 后，当前 Adapter 实际读取的是：

```text
source_type=jsonl
source_payload_ref=rollouts.jsonl#line:1
```

如果保留旧 envelope，就会谎称本次输入直接读取了 Polar artifact。旧来源仍可保存在 opaque lineage 中，但直接来源必须真实。

### 11.7 三种错误为何分开捕获

```text
JSONDecodeError             → ADAPTER_ERROR
ContractValidationError    → CONTRACT_INVALID
其他 envelope 类型错误     → ADAPTER_ERROR
```

这区分“连 JSON/外壳都不成立”和“JSON 成立但 canonical 字段矛盾”。两者修复责任不同。

### 11.8 空输入为什么是 warning，不是 error

一个空文件可以是合法的“当前没有 rollout”结果。它不包含损坏数据，因此没有 parse error；但下游大概率需要注意，于是返回：

```text
records=()
warnings=(SOURCE_WARNING,)
ok=True
```

后续 Processor 可以决定空 batch 的业务处理。

### 11.9 JSONL Adapter 当前没有做的事

- 不支持注释行；
- 不支持多行 JSON object；
- 不自动修复 schema；
- 不忽略未知字段；
- 不验证一个文件中 record 的业务分组；
- 不做流式 iterator，当前一次把文本读入内存。

这些限制让 v1 简单、严格、确定。超大规模数据时可以增加 streaming 版本，但不能改变现有单行语义。

## 12. Polar Adapter：把真实上游语义压缩到公共契约

代码文件：[`src/sources/polar.py`](../../src/sources/polar.py)

### 12.1 为什么它读取 fixture，而不是直接 import Polar

当前 Adapter 的输入是经过打包和审核的 fixture directory。这样做有四个好处：

- 核心测试不安装 Polar；
- artifact 可以固定 commit、模型 revision 和 checksum；
- Adapter 测试不会启动 GPU 服务；
- 上游升级不会在测试时改变历史输入。

这是一种 anti-corruption layer：Polar 私有 schema 在这一层结束。

### 12.2 `convert()` 的顶层状态机

```text
payload 不是 path
    → ADAPTER_ERROR

读取/解析/UTF-8 失败
    → ADAPTER_ERROR

manifest 或 canonical 不变量失败
    → CONTRACT_INVALID

转换成功但缺 required capability
    → records 保留 + CAPABILITY_MISSING errors

全部成功
    → records + common capabilities + warnings
```

### 12.3 `_load_fixture()` 建立的信任边界

它检查：

1. fixture path 必须是目录；
2. 根目录本身不能是 symlink；
3. `source-manifest.json` 和 `summary.json` 必须是 JSON object；
4. manifest schema 必须是 `polar-fixture-manifest/v1`；
5. manifest 必须声明 `summary.json` 且 role 为 `summary`；
6. summary 实际 bytes 和 SHA256 必须与 manifest 一致。

核心效果是：

```text
Adapter 后续映射的 summary
= manifest 审核时对应的精确 summary bytes
```

### 12.4 一个重要的当前限制：Adapter 只现场验证 summary

manifest 可能还列出 request、response、patch、replay 等文件。`PolarSourceAdapter._load_fixture()` 当前只现场读取并核验 `summary.json`。

其他文件的 checksum 被保存进 opaque metadata，但其完整验证由：

```text
scripts/verify_polar_fixture.py
```

负责。

所以两者的责任是：

```text
fixture verifier  验证整个 fixture 包
Polar Adapter     验证自己实际消费的 summary，并转换字段
```

不能只运行 Adapter 测试，就声称 patch/replay 等所有附件都验证过。

### 12.5 `_records_from_summary()` 为什么一条 session 可能产生多条 record

代码遍历 `trajectory.traces`：

```text
1 trace  → source_record_id = session_id
N traces → source_record_id = session_id:trace:index
```

这保留 trace 级 identity，避免多 trace 使用同一个 `trajectory_id` 后被 batch 去重拒绝。

如果完全没有 traces，则生成一条 empty failure record，使失败仍然可以进入 classifier，而不是消失在 Adapter 日志中。

### 12.6 fixture type 为什么不能支配 reward

当前代码只把 `fixture_type` 当审核标签，不当 ground truth。

例如：

```text
fixture_type = calculator_success
observed reward = 0.0
```

Adapter 行为：

```text
保留 reward=0.0
发 SOURCE_WARNING
不改成 reward=1
```

这体现 evidence-first 原则：目录命名可以错，源观察值不能为迎合命名而改写。

对 `coding_success`，如果任意 reward 不为 1，也会警告语义冲突。

### 12.7 `_trace_record()` 的映射方程

设 Polar trace 给出：

```text
P = prompt_ids
R = response_ids
Mr = response loss_mask
Lr = response_logprobs
```

canonical 映射为：

```text
T = P || R
M = 0^|P| || Mr
prompt_token_count = |P|
old_logprobs = Lr
```

其中 `||` 表示拼接。

映射前检查：

```text
len(Mr) == len(R)
len(Lr) == len(R)，如果 Lr 存在
```

映射后 `RolloutRecord` 再检查：

```text
len(M) == len(T)
mask 只含 0/1
token 非负
logprob 有限
```

这是两级校验：Adapter 检查上游局部结构，canonical contract 检查统一全局结构。

### 12.8 `response_logprobs → old_logprobs` 为什么不是随意改名

在 RL/PPO/GRPO 语境中，`old_logprobs` 必须代表生成 rollout 的行为 policy 对已采样 token 的 log probability。

我们根据锁定的 Polar `prefix_merging` trajectory builder 语义确认 `response_logprobs` 就是 sampled response logprobs，才映射到 canonical 字段。

同时保存：

```text
response_logprobs_provenance
```

指向原 JSON path。这样下游看到 canonical 名称时，仍能追溯语义证据。

如果未来另一个 Polar 字段只是 verifier score 或重新计算的 current-policy logprob，就不能因为数值形状相同而映射为 `old_logprobs`。

### 12.9 prompt token 的 mask 为什么补零

Polar trace 的 `loss_mask` 与 response 对齐。canonical contract 的 mask 与完整 `token_ids` 对齐，因此 prompt 部分补：

```text
[0] * len(prompt_ids)
```

这不是伪造训练数据，因为 prompt token 明确不参与 response loss；它只是把局部 response mask 提升为完整序列坐标系。

### 12.10 `verifier_status` 的保守映射

当前顺序：

```text
没有 evaluation                → NOT_RUN
没有 report object             → UNKNOWN
test_timeout=true              → TIMEOUT
error_eval/failed_apply_patch   → ERROR
resolved=true                  → PASSED
resolved=false                 → FAILED
其他                            → UNKNOWN
```

顺序很重要。例如 report 同时出现 timeout 与 resolved 字段时，timeout 优先，因为任务结论不可信。

### 12.11 `FAILED` 为什么不是 infrastructure-invalid

`VerifierStatus.FAILED` 的语义是 verifier 正常完成并判定答案错误。它是有效任务失败的候选。

`ERROR` 或 `TIMEOUT` 表示 verifier 没能给出可信判断，应由 Day 5 分类为 infrastructure-invalid。

这一区别正是项目价值所在。

### 12.12 tool events 当前提取了什么

代码只遍历：

```text
trace.response_messages[*].tool_calls
```

并收集其中的 object。

它没有把所有 tool result、stdout、runtime side effect 都正规化成统一 event schema。当前 `TOOL_EVENTS` capability 表示至少保存了结构化 tool call，不表示已完成完整工具因果链建模。

这是 v1 的有限覆盖，Day 3 多轮 Coding fixture 会帮助我们判断是否需要扩展。

### 12.13 为什么正常 trace 的 component status 仍是 `UNKNOWN`

当前 Day 2 summary 没有为 Adapter 提供足够稳定、统一的 runtime/harness/model-backend 独立状态 path。因此 `_trace_record()` 没有根据 `COMPLETED` 猜三个组件都成功，而是保留：

```text
runtime_status=UNKNOWN
harness_status=UNKNOWN
model_backend_status=UNKNOWN
```

这看起来不够“完整”，但比错误推断更安全。`rollout_status=COMPLETED` 不能逻辑推出每个组件都有独立成功证据。

### 12.14 finish reason 为什么没有写入 canonical termination reason

当前代码将 Polar `finish_reason` 放在 `opaque_metadata.polar.finish_reason`，没有直接映射为顶层 `termination_reason`。

原因是还没有用真实多轮 Coding artifact 冻结 finish reason 与 canonical termination 语义的完整映射。直接复制字符串虽然简单，却可能把模型 completion finish reason 和整个 Agent rollout termination 混为一谈。

这是等待 Day 3 证据校准的明确缺口。

### 12.15 `_empty_failure_record()` 的语义

没有 trace 时，代码仍构造 identity 和 lineage，但不构造训练 payload：

```text
token_ids=None
mask=None
old_logprobs=None
reward=None
verifier_status=NOT_RUN
```

如果 error 文本包含 `runtime`，设置：

```text
runtime_status=FAILED
harness_status=NOT_RUN
model_backend_status=NOT_RUN
```

否则 runtime status 保守为 `UNKNOWN`。

注意：当前实现对 runtime failure 的识别仍使用 error 文本包含关系。它适用于已冻结 Day 2 synthetic fault，但不是最终通用 FailureClassifier 规则。Day 5 应优先消费稳定 component status/reason code，而不是把这一启发式扩散到核心分类器。

## 13. 用真实 Day 2 数据完整走一遍

真实 execution-complete fixture 映射结果：

```text
prompt tokens            14846
response tokens             99
full token_ids           14945
full loss_mask           14945
response_logprobs           99
reward                     0.0
resolved                 false
```

### 13.1 形成 canonical payload

```text
token_ids = prompt_ids + response_ids
prompt_token_count = 14846
loss_mask = 14846 个 0 + 99 个 response mask
old_logprobs = 99 个 response_logprobs
reward = 0.0
verifier_status = FAILED
```

### 13.2 计算 capabilities

存在：

```text
TOKEN_IDS
ACTION_MASK
OLD_LOGPROBS
REWARD
VERIFIER_EVIDENCE
```

不存在：

```text
GROUP_ID
POLICY_VERSION
TOOL_EVENTS（若没有结构化 tool_calls）
```

### 13.3 为什么它可以进入 `RolloutBatch`

它是内部自洽且有唯一 identity 的 canonical record。`RolloutBatch` 接受混合 capability 和未分类记录。

### 13.4 为什么它不能进入默认 `TrainingReadyBatch`

默认训练门槛要求：

```text
GROUP_ID + POLICY_VERSION
```

两者缺失，所以构造时会抛 `CapabilityMissingError`。

即使我们手工给它填两个字符串，当前也不应该直接训练，因为 Day 5 尚未正式产出 validity decision、signal decision 和 processing versions。

### 13.5 它最终应如何分类

根据实际证据：

```text
rollout execution completed
verifier 正常判定 resolved=false
reward=0.0
```

应成为：

```text
VALID_FAILURE
```

但这个 decision 类型与 classifier 尚未在 Day 4 实现。Adapter 只保存足够证据并发出目录命名 warning。

## 14. 用真实 runtime fault 再走一遍

输入事实：

```text
status=ERROR
runtime prepare failed
no completion
no trace
no verifier outcome
```

Adapter 输出：

```text
rollout_status=FAILED
runtime_status=FAILED
harness_status=NOT_RUN
model_backend_status=NOT_RUN
verifier_status=NOT_RUN
token_ids=None
reward=None
```

Capabilities 是空集。

这里最重要的不是“记录不完整”，而是这种不完整准确表达了执行从未到达模型任务评价阶段。

如果把它转换成：

```text
reward=0.0
token_ids=[]
loss_mask=[]
```

会制造三个谎言：

1. evaluator 给出了零 reward；
2. 模型完成了一次零 token 的采样；
3. 这条记录拥有 token/mask capability。

当前实现明确阻止这种默认值污染。

## 15. 测试不是覆盖率装饰，而是可执行设计说明

### 15.1 `tests/contract_fixtures.py`

`make_record()` 提供一条完整、可训练结构的合成 record。测试通过覆盖一个字段制造单一变量实验：

```python
make_record(policy_version=None)
make_record(old_logprobs=None)
make_record(loss_mask=(1, 1))
```

这比每个测试手写完整对象更容易看出被破坏的不变量。

### 15.2 `test_rollout_record.py`

#### 不可变性测试

确认 nested metadata 不能修改。它保护 checksum 和 lineage。

#### roundtrip/checksum 测试

确认：

```text
RolloutRecord → dict → RolloutRecord
```

语义相同，并且 metadata key 顺序不影响 checksum。

#### unknown field 测试

如果上游新增 `polar_gateway_node` 到 canonical 顶层，`from_dict()` 拒绝并要求放进 opaque metadata。

它防止 schema drift 静默扩散。

#### token/mask/logprob 测试

分别验证：

- mask 不能脱离 token；
- mask 必须等长；
- 两种 mask 不能同时存在；
- old logprob 可按三种长度对齐；
- prompt boundary 合法。

#### missing fields 测试

它确认缺失字段产生 capability absence，而不是由 constructor 填默认训练值。

### 15.3 `test_batches.py`

#### capability intersection

一条有 old logprobs、一条没有，batch 不声明 `OLD_LOGPROBS`。

#### duplicate rejection

同 trajectory 即使 source record ID 不同，也被拒绝。canonical identity 优先于运输身份。

#### policy consistency

batch 声明 `policy-v0`，record 是 `policy-v1`，直接 contract invalid。

#### missing capability

合法但无 token 的 record 不能进入 TrainingReadyBatch。

#### deterministic resample request

相同输入产生相同 checksum/request ID，零 required count 被拒绝。

### 15.4 `test_jsonl.py`

最重要的不是“能读 JSON”，而是以下语义：

```text
roundtrip 保留 semantic_dict
source envelope 必须更新
单行损坏不吞相邻合法行
存在 error 时不能 to_batch
capability 必须是 record 交集
unknown envelope 与 invalid record 分开报错
```

### 15.5 `test_polar.py`

#### 不 import Polar

确认协议实现不需要上游包进入 `sys.modules`。这证明 core/integration fixture 测试真正离线。

#### token 拼接与 logprob 对齐

合成：

```text
prompt=[10,11]
response=[20,21,22]
mask=[1,0,1]
```

期望：

```text
tokens=[10,11,20,21,22]
mask=[0,0,1,0,1]
```

#### pre-run failure

确认没有 trace 时 reward 和 token 保持 `None`。

#### execution success 与 task reward

reward=0 的 `calculator_success` 产生 warning，而不是被改成 1。

#### required capability

要求 policy/group 时，Adapter 显式返回缺失错误。

#### manifest tamper

修改 summary 而不更新 manifest，checksum mismatch 被拒绝。

#### cross-adapter roundtrip

Polar canonical record 经 JSONL 后 `semantic_dict()` 相同，证明两个 Source Adapter 汇入同一公共语义。

## 16. 昨天同时完成的非核心支撑代码

除了 `src/` 核心，昨天还准备了 Day 3 的执行与证据工具。它们不属于训练数据核心逻辑，但保证真实 Coding fixture 能被安全带回仓库。

### 16.1 `scripts/select_swebench_candidates.py`

作用：从已缓存数据中确定性筛选少量候选任务，优先测试较少、prompt 较短，并可优先不同 repository。

它不运行模型、不修改 dataset，也不声称候选一定成功。它只是降低服务器 pilot 成本并冻结任务选择规则。

### 16.2 `configs/polar/coding/capture-policy.yaml`

作用：定义 Coding fixture 必须保存的证据和三类 outcome：

```text
coding_success         → VALID_SUCCESS
coding_valid_failure   → VALID_FAILURE
coding_invalid_infra   → INVALID_INFRASTRUCTURE
```

同时规定 patch、verifier evidence、clean replay、fault injection 等文件角色。

### 16.3 `scripts/package_polar_fixture.py`

作用：将服务器 staging 目录打包为安全、大小受限、带 SHA256 manifest 的仓库 fixture。

它维护的是 artifact provenance，不负责转换成 `RolloutRecord`。

### 16.4 `scripts/verify_polar_fixture.py`

作用：验证：

- manifest/path/size/checksum；
- 必需文件；
- coding outcome 与 verifier/replay 是否一致；
- patch checksum；
- infrastructure failure 是否错误地填了 reward；
- 明显敏感字段和私钥模式。

它验证 fixture 包，不等于 Day 5 FailureClassifier。

### 16.5 Day 3 runbook

[`docs/runbooks/day-03-polar-coding.md`](../runbooks/day-03-polar-coding.md) 冻结了：

- 官方 Polar commit 与 SWE example；
- 单 GPU、顺序采样方式；
- success/failure 和两类 infrastructure fault；
- exact PID teardown；
- staging 和打包命令；
- 服务器执行完成标准。

这些支撑代码已经完成，但真实 GPU rollout 尚未因为 Mac 端没有服务器控制通道而执行。

## 17. 当前实现刻意没有完成什么

深度理解一个系统，也要知道它没有做什么。

### 17.1 尚无 `FailureClassifier`

当前只有 status/evidence 字段，没有：

```text
ValidityDecision
VALID_SUCCESS
VALID_FAILURE
INVALID_INFRASTRUCTURE
versioned reason code
```

不能把 Polar Adapter 的 warning 当 classifier 输出。

### 17.2 尚无 `SignalFilter`

当前没有检查：

- reward variance；
- 全成功/全失败 no-signal group；
- 最小 trainable token；
- signal threshold。

`TrainingReadyBatch` 本身也不做这些判断。

### 17.3 尚无真正的 GroupBuilder

`TrainingReadyBatch` 会拒绝 mixed policy，但没有代码自动：

- 按 `(task_id, group_id, policy_version)` 分桶；
- 选择固定数量样本；
- 确定性排序；
- 缺样本时生成 request；
- 记录 rejected records。

### 17.4 尚无 Trainer Adapter

没有 Slime Sample 映射，也没有 optimizer step。当前实现证明的是数据边界，不是训练闭环。

### 17.5 Day 3 真实 Coding golden fixture 尚未回来

因此仍待验证：

- 真实 task-level success；
- 多轮 tool call/result；
- patch 与 clean replay；
- verifier timeout/error 的完整映射；
- canonical termination reason；
- policy/group identity 应从哪个真实上游位置取得。

## 18. 当前设计中值得保留的原则

### 18.1 Preserve absence

```text
不知道 ≠ 0
没有采样 ≠ 空序列
没有 reward ≠ reward 0
没有 policy identity ≠ 默认 policy
```

### 18.2 Evidence before label

目录名、运行名和人类备注都不能覆盖 trace、verifier 和 reward 的真实观察。

### 18.3 Represent first, qualify later

`RolloutRecord` 先表示各种成功与失败；capability、classifier、filter、builder 再逐步收紧资格。

如果在 Adapter 入口就丢弃所有无训练字段记录，系统将失去 infrastructure failure 统计和补采依据。

### 18.4 Fail before side effects

capability 和 contract gate 应在 Trainer 开始处理前失败，而不是 optimizer step 中途才发现某条记录缺字段。

### 18.5 Core stays framework-agnostic

Polar 字段只在 `src/sources/polar.py`；未来 Slime 字段只应在 trainer adapter。公共 Processor 只读取 canonical 语义。

## 19. 对当前实现的技术评价

### 19.1 已经做对的部分

1. **边界清楚**：contract 不 import Polar/Slime；
2. **缺失语义正确**：没有用零值伪造训练字段；
3. **lineage 完整**：schema、checksum、source ref、版本信息都有位置；
4. **能力门控明确**：合法 record 与可训练 record 分离；
5. **执行与任务 outcome 分离**：真实 Day 2 命名冲突被正确保留；
6. **测试表达业务风险**：不是只测 happy path；
7. **跨 Adapter 语义可比**：JSONL 提供 producer-independent 入口。

### 19.2 需要后续增强但不阻塞当前阶段的部分

1. 显式 `logprob_alignment`；
2. semantic checksum 与 lineage checksum 分离；
3. RolloutBatch duplicate 检测改为 O(n)；
4. 大型 JSONL 的 streaming 解析；
5. 更完整的 tool result/event schema；
6. 真实 component status 与 termination mapping；
7. manifest 所有附件的消费时验证策略；
8. 对 Adapter 意外异常的统一防护边界。

这些问题应该记录并按下游需求处理，而不是现在一次性扩张 scope。

## 20. 你应该怎样亲自验证已经理解

不要先背类定义。按下面四个实验推演。

### 实验一：删除 policy version

从 `make_record()` 得到完整记录，设置：

```python
policy_version=None
```

回答：

1. record 能否构造？能；
2. 是否拥有 `POLICY_VERSION` capability？否；
3. 能否进入 `RolloutBatch`？能；
4. 能否进入默认 `TrainingReadyBatch`？不能；
5. 这是不是 task failure？不是。

### 实验二：runtime failure 填 reward=0

思考为什么 contract 本身允许 `rollout_status=FAILED, reward=0.0`，而业务上仍可能错误。

答案：record contract 只检查内部类型和结构，无法仅凭这两个字段知道 reward 是否伪造。Day 5 classifier 必须联合 component/verifier evidence 判断。这说明结构 validation 不能替代业务 classification。

### 实验三：JSONL 中间一行损坏

结果会同时包含：

```text
合法 records
+ 一个 ADAPTER_ERROR
+ ok=False
```

回答为什么保留 records 却禁止 `to_batch()`：为了诊断可见性与训练完整性同时成立。

### 实验四：两条 record reward 都是 1

它们可以通过当前 `TrainingReadyBatch` 结构检查，只要 task/group/policy/capability 一致。

但它们可能没有 GRPO 组内相对信号。这个反例说明 Day 4 完成了结构 contract，却没有替代 Day 5 SignalFilter。

## 21. 对当前阶段最准确的总结

截至现在，我们完成的不是一个完整 data infra pipeline，而是它最重要的可信地基：

```text
1. 一个可以表示成功、任务失败和基础设施失败的 canonical record
2. 一套保证 token/mask/logprob/reward 内部自洽的结构不变量
3. 一套把真实字段存在性提升为可执行前置条件的 capability 系统
4. 三个区分原始 batch、训练结构 batch 和补采请求的版本化 contract
5. 一个允许部分诊断、禁止部分数据静默训练的 AdapterResult 闸门
6. 一个确定性、与 Polar 无关的 JSONL 入口
7. 一个依据真实 fixture 显式映射且不 import Polar 的 Polar Adapter
8. 一组把上述设计转成可执行规范的单元测试
9. 一套等待服务器执行的 Coding fixture 打包、验证和 runbook 工具
```

它已经解决了“上游 artifact 如何被忠实、稳定、可审计地表达”这个问题。

它还没有解决“哪些记录最终有效、哪些 group 有信号、如何自动组成固定大小的同 policy batch”。这正是下一阶段三个 Processor 要建立在这套地基之上的原因。
