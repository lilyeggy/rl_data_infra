# 昨晚 Stage 1–3 完整实现深度讲解

> 历史快照说明（2026-08-12）：本文解释项目转向前的 Day 1–4 实现基础，其中关于未来 Day 5/6 GRPO Processor/Trainer 的规划已不再生效。当前路线见 [`project-reading-order.md`](project-reading-order.md)。

> 本文严格按照昨晚的实际完成顺序讲解，不按照 Day 1、Day 2、Day 3 的自然编号重排。
> Stage 1：收口 Day 2 Polar Calculator 真实执行结果。
> Stage 2：基于真实证据实现 Day 4 canonical contract 与 Source Adapter。
> Stage 3：实现 Day 3 Coding/SWE 最小真实 rollout 的服务器执行包。
> 这里的“Stage 3 已实现”指执行工具、证据契约、打包验证和 runbook 已实现；真实 GPU Coding rollout 仍需要服务器执行，不能把准备完成写成实验结果完成。

## 0. 为什么昨晚的顺序是 Day 2 → Day 4 → Day 3

通常实施计划按 Day 1–7 顺序推进，但昨晚为了加速，我们采用了证据驱动的压缩顺序：

```text
Stage 1
审计服务器已经跑出的 Day 2 Calculator artifact
        │
        │ 提供真实 token/status/reward 字段
        ▼
Stage 2
立即实现不依赖 Polar 的 canonical contract 与 Adapter
        │
        │ 暴露仍缺少的 success/group/policy/tool 证据
        ▼
Stage 3
实现 Day 3 Coding/SWE 真实采集、故障注入、clean replay 和验证工具
```

这个顺序不是计划混乱，而是在压缩等待时间：

- Day 2 服务器结果已经存在，先把它变成可信证据；
- Day 4 大部分核心结构可以依据 Day 2 立即实现，不必等 Coding GPU 运行；
- Day 3 是昂贵服务器操作，所以先把所有本地可实现的选择器、契约、验证器和 runbook 准备到可执行状态；
- 服务器执行 Day 3 时，本地可以继续 Day 5，而不是所有工作串行等待。

### 0.1 三个 Stage 分别回答什么问题

| Stage | 核心问题 | 产出类型 |
|---|---|---|
| Stage 1 | Polar 实际产生了什么，执行成功和任务成功是否相同？ | 真实 fixture、manifest、语义审计 |
| Stage 2 | 如何把异构 artifact 变成不撒谎的公共数据对象？ | contract、capability、Adapter、测试 |
| Stage 3 | 如何取得 Coding success/failure/infra 三类更完整证据？ | capture policy、候选选择、打包验证、runbook |

三个 Stage 不是三个孤立功能，而是：

```text
观察事实 → 固化事实 → 补齐事实
```

---

# Part I：Stage 1——收口 Day 2 Polar Calculator 真实结果

## 1. Stage 1 的输入不是代码猜想，而是服务器 artifact

Stage 1 的原始对象是服务器提交中的两套 fixture：

```text
tests/fixtures/polar/calculator_success/
tests/fixtures/polar/calculator_fault/
```

以及本次执行的记录：

```text
artifacts/day-02/20260809T164046Z-calculator/
```

相关服务器提交包括：

```text
cb456c02960cc1dc0a91e09b97ee675ae6036620
    Complete Day 2 Polar calculator fixtures

bdfa2bdafd80fc37030c52edc4f8f228601c4b5e
    Close Day 2 Polar execution evidence
```

这里最重要的原则是：

> 先接受服务器保存下来的实际字段，再决定 canonical schema；不能先写一个理想 schema，再强迫 artifact 看起来符合它。

## 2. Day 2 实际运行拓扑

Stage 1 复现和收口的是下面这条链：

```text
Calculator task
    │
    ▼
Polar Rollout Server
    │ 调度 session
    ▼
Polar Gateway
    │ 代理模型 completion
    ▼
SGLang + Qwen3-4B
    │
    ▼
qwen_code / Calculator runtime
    │
    ▼
trajectory builder
    │
    ▼
evaluator / reward
```

没有运行：

```text
Slime
Megatron
GRPO update
checkpoint reload
```

所以 Day 2 证明的是 rollout 数据路径，不是训练路径。

### 2.1 为什么这一边界重要

如果看到 Polar 项目中有 Slime/Megatron example，就误以为跑 Calculator 必须复现官方八卡训练拓扑，会产生巨大无关工作量。

我们的目的只是获取 Adapter 所需的真实来源数据：

```text
token
mask
logprob
reward
status
verifier evidence
lineage
```

## 3. Stage 1 固定了哪些上游身份

Day 2 runbook 锁定：

```text
Polar commit
f0e8343a7870abf6ec2366890f685881ceab92cb

SGLang commit/version
28b095c01005d4a3a2a5b637b7d028b07fba31b2 / 0.5.13

Model
Qwen/Qwen3-4B-Instruct-2507

Model/tokenizer revision
cdbee75f17c01a7cc42f958dc650907174af0554

Calculator runtime image
sha256:0a967a5c33ee206f7524181cdd4ff866389016651645336eee93de1a7d970f49
```

这些不是普通环境说明。它们是数据语义的一部分：

- tokenizer revision 改变，token ID 解释可能改变；
- model revision 改变，sampled logprobs 不再属于同一 policy；
- Polar builder commit 改变，trace 字段语义可能改变；
- runtime image 改变，工具和 evaluator 结果可能改变。

所以 fixture manifest 保存的是“这条轨迹由哪套确定系统产生”，不是只保存三份 JSON。

## 4. 两套 fixture 的角色

### 4.1 `calculator_success`

这里的 `success` 原意是：

```text
rollout 和 evaluator 执行路径完成
```

它不是：

```text
模型完成了 Calculator 任务
```

真实观察为：

```text
summary.status = COMPLETED
evaluation.outcome_reward = 0.0
evaluation.report.resolved = false
evaluation.report.empty_generation = true
trace_count = 1
```

模型生成了一次文本形式 tool call，然后 `qwen_code` 退出；没有形成成功 task outcome。

正确分类是：

```text
执行链完成 + verifier 正常判错
= VALID_FAILURE
```

### 4.2 `calculator_fault`

这是显式、非破坏性的 synthetic infrastructure failure：

```text
runtime prepare 失败
session status = ERROR
run_ms = 0
无 completion
无 reward
```

正确分类是：

```text
INVALID_INFRASTRUCTURE
```

它绝不能变成：

```text
reward = 0
```

因为模型根本没有获得一次可以被 evaluator 判错的尝试。

## 5. Stage 1 最重要的语义发现

### 5.1 `COMPLETED` 只描述生命周期，不描述任务正确性

一次 session 可以：

```text
正常启动
正常调用模型
正常结束 Harness
正常运行 evaluator
最终 reward=0
```

所以一个统一的 `status=success` 不足以训练。后续 contract 必须拆开：

```text
rollout_status
runtime_status
harness_status
model_backend_status
verifier_status
reward
```

### 5.2 `0` 与 `None` 是不同事实

```text
reward=0.0
    evaluator 正常完成并给出失败结果

reward=None
    没有可信 evaluator outcome
```

这一区别直接决定 task failure 是否可以成为 GRPO 负样本。

### 5.3 Polar 真实提供 native token arrays

真实 trace 中存在：

```text
prompt_ids
response_ids
loss_mask
response_logprobs
```

因此后续 Adapter 不需要 tokenizer，也不应该重新 tokenize 文本。

### 5.4 字段名不存在，不等于语义不存在

服务器字段审计最初说没有字面名为 `old_logprobs` 的字段，这在字面上正确。

但锁定 Polar `prefix_merging` builder 定义 `response_logprobs` 为采样行为 policy 对 response token 的 logprob。因此：

```text
Polar response_logprobs
→ canonical old_logprobs
```

是有上游代码语义依据的映射，不是看到 shape 相同就猜字段。

### 5.5 真实数据迫使 contract 增加 prompt boundary

Polar 分开保存 prompt 与 response：

```text
prompt_ids
response_ids
```

而完整训练序列需要拼接。只保存拼接结果和 mask 仍不够，因为 response 内可以有 mask=0 的 token。

因此 Stage 2 加入：

```text
prompt_token_count
```

用来稳定表示 response 边界。

## 6. Fixture manifest 为什么存在

每个 fixture 不是随意复制的目录，而是一个版本化证据包：

```text
source-manifest.json
request.json
response.json
summary.json
可选 logs/evidence
```

manifest 至少记录：

```text
fixture_id/type
synthetic_fault
created_at_utc
Polar/model/tokenizer/runtime/harness identity
每个文件的 role/media_type/bytes/SHA256
redactions
known_missing_fields
notes
```

它解决三个问题：

1. 当前文件是不是审核时的精确字节？
2. fixture 是真实 outcome 还是显式 fault injection？
3. 哪些字段本来就缺失，而不是打包时丢了？

## 7. 通用打包器的实现逻辑

代码文件：[`scripts/package_polar_fixture.py`](../../scripts/package_polar_fixture.py)

### 7.1 为什么输入必须先经过 staging

打包器不直接遍历 Polar 的全部 raw runtime 目录。服务器先将需要的文件复制、审阅、脱敏到 staging：

```text
raw artifacts
→ reviewed/redacted staging
→ package_polar_fixture.py
→ committed fixture
```

raw 目录可能包含：

- 大日志；
- 内部路径；
- API header；
- Docker/主机信息；
- 不属于固定 contract 的临时文件。

打包器只处理明确 allowlist。

### 7.2 Required file 按 fixture type 变化

公共必需文件：

```text
request.json
response.json
summary.json
```

Coding fixture 额外要求：

```text
verifier-evidence.json
patch.diff
```

有效 Coding success/failure 再要求：

```text
replay.json
```

这使同一个工具支持 Stage 1 Calculator 和 Stage 3 Coding，而不是复制两套打包脚本。

### 7.3 `discover_source_files()` 的安全策略

它不是简单 `rglob` 全复制，而是：

```text
验证 source 是真实目录且不是 symlink
→ 验证 required file 存在、是 regular file、不是 symlink
→ JSON 必须是 UTF-8 且能解析
→ 文本必须是 UTF-8
→ 加入明确 optional evidence/log
→ 扫描 staging 全部文件
→ 任何未分类文件导致拒绝
→ 按 relative path 排序
```

“未知文件导致失败”是刻意设计。自动打包未知文件虽然方便，却最容易把 secret 或巨大日志带入 Git。

### 7.4 为什么禁止 symlink 与逃逸路径

如果 staging 内允许 symlink：

```text
staging/runtime.log → /etc/secret
```

普通复制可能越过 staging 边界。因此实现验证：

- root 不能是 symlink；
- 每个发现路径不能是 symlink；
- relative path 不能包含 `..`、绝对路径、空 segment。

### 7.5 为什么拒绝覆盖输出

fixture 是 golden evidence。覆盖已有文件会导致：

- 旧 test 突然指向新运行；
- review diff 难以区分删除与替换；
- artifact lineage 被破坏。

所以输出目录只允许预先存在一个 README placeholder；其他已有文件一律拒绝覆盖。

### 7.6 临时目录 + 双重验证

打包流程为：

```text
复制到 output 同级临时目录
→ 生成 manifest
→ 在临时目录运行 verifier
→ 全部通过后逐文件写入最终输出
→ 对最终输出再运行 verifier
```

第一遍防止生成坏 fixture；第二遍防止写入过程或目标目录状态产生差异。

这比直接向最终目录边生成边写更接近原子发布。

## 8. Fixture verifier 的实现逻辑

代码文件：[`scripts/verify_polar_fixture.py`](../../scripts/verify_polar_fixture.py)

### 8.1 Verifier 是只读的

它不修复 manifest、不重写 checksum、不删除敏感字段。验证失败只返回：

```text
PASS / WARN / FAIL
```

如果验证器自动修复输入，就会让“不可信 artifact”修改自己的证明。

### 8.2 manifest contract 检查

它检查：

- root/source/file/redaction 是否缺字段或多未知字段；
- fixture type 是否在支持集合；
- synthetic fault 与 fixture type 是否一致；
- UTC 时间戳是否以 `Z` 表达；
- Polar commit 是否为完整 40 位 SHA；
- model/runtime/harness 是否不是 placeholder；
- file path/role/media type/bytes/SHA256 是否合法；
- policy version 允许为 null，但不能是假空字符串。

这里拒绝 `TBD`、`unknown`、`<placeholder>`，因为 manifest 的目的就是冻结事实，不是保存待办。

### 8.3 文件完整性检查

对 manifest 每个 entry：

```text
路径必须留在 fixture root
不能是 symlink
文件必须存在
实际 bytes 必须匹配
实际 SHA256 必须匹配
```

同时检查实际目录里有没有 manifest 未列出的文件。

### 8.4 仓库体积约束

默认单文件最大：

```text
2 MiB
```

超过限制的 raw artifact 应留在服务器，仓库只保存外部引用和 checksum。这样 golden fixture 可 review、可 clone，不会逐渐变成数据湖。

### 8.5 明显敏感信息检查

JSON key 扫描包括：

```text
api_key
authorization
cookie
private_key
refresh_token
secret
```

文本扫描包括 private key header。

它只能发现“明显模式”，不是完整 DLP 系统，所以人工 staging 审查仍不能省略。

### 8.6 Calculator outcome 检查

对 `calculator_success`：

```text
summary.status 必须 COMPLETED
必须有 evaluator evidence/report
outcome reward 必须为数值
evaluator 不能 error/timeout/failed apply patch
```

它不要求 reward=1。

这是 Stage 1 审计后修正的关键：目录的 success 指 execution path success。

对 `calculator_fault`：

```text
terminal status 必须 error/timeout/failed
reward 必须 null/不存在
必须有显式 error message
```

### 8.7 Verifier 与模型 evaluator 不是同一个东西

这里容易混淆三层 verifier：

```text
任务 evaluator
    判断模型解答/patch 是否解决任务

fixture verifier
    判断证据包是否完整、自洽、安全

FailureClassifier（尚未实现）
    将 canonical record 分类为 valid success/failure/invalid infra
```

Stage 1 实现的是第二层，并保存第一层结果；第三层属于 Day 5。

## 9. Stage 1 的实际结论

### 已完成

- 两套 Calculator fixture 的 manifest、path、size、SHA256 检查通过；
- success 路径有完整 trace 和 evaluator evidence；
- fault 路径没有 completion/reward，也没有把 infra failure 写成 reward 0；
- 实际 token/logprob 字段已观察；
- `calculator_success` 的命名误导已在审计文档中澄清；
- Day 2 接受为 `COMPLETED_WITH_NOTES`。

### 没有完成

- 没有 task-level `VALID_SUCCESS`；
- 没有多轮 Coding tool/patch/replay；
- 没有 `group_id`；
- 没有 `policy_version`；
- 没有训练闭环。

### Stage 1 如何直接决定 Stage 2

```text
观察到 prompt_ids + response_ids
→ contract 加 prompt_token_count

观察到 response_logprobs
→ 在有 builder 语义依据下映射 old_logprobs

观察到 COMPLETED + reward=0
→ execution status 与 task outcome 分离

观察到 runtime fault 无 reward
→ missing 必须保持 None

观察到 group/policy 缺失
→ capability gate 不能允许训练
```

---

# Part II：Stage 2——立即实现 Day 4 核心 Contract 与 Adapter

## 10. Stage 2 的核心设计目标

Stage 2 要建立一个稳定中间层：

```text
Polar fixture / JSONL
        │
        ▼
Source Adapter
        │
        ▼
RolloutRecord
        │
        ▼
RolloutBatch
        │
        ▼
未来 Day 5 Processors
        │
        ├── TrainingReadyBatch
        └── ResampleRequest
```

核心约束：

- 不 import Polar；
- 不 import Slime/Megatron；
- 不重新 tokenize；
- 不给缺失 logprob/reward/policy 填零或默认值；
- 上游私有字段不进入公共顶层 schema；
- 所有版本、错误与 checksum 可审计。

Stage 2 的逐类、逐方法深解已经单独记录在：

[`docs/learning/day04-implementation-deep-dive.md`](day04-implementation-deep-dive.md)

下面从三阶段总链路角度解释它为什么这样实现。

## 11. `RolloutRecord` 如何接住 Stage 1 的两种相反样本

### 11.1 execution-complete task failure

可以表示：

```text
rollout_status=COMPLETED
verifier_status=FAILED
reward=0.0
token_ids=(...)
loss_mask=(...)
old_logprobs=(...)
```

### 11.2 pre-run infrastructure failure

也可以表示：

```text
rollout_status=FAILED
runtime_status=FAILED
verifier_status=NOT_RUN
token_ids=None
loss_mask=None
old_logprobs=None
reward=None
```

同一个 schema 能表示二者，但不会让二者拥有相同训练资格。

这就是 contract 和 validity decision 分离的价值：

```text
contract 问：这条事实表达是否自洽？
classifier 问：这条轨迹是否是有效任务样本？
trainer gate 问：它是否满足具体训练输入？
```

## 12. Training payload 的结构不变量

设：

```text
N = token 总数
P = prompt_token_count
M = mask 中 1 的数量
L = old_logprobs 数量
```

当前 contract 维护：

```text
0 <= P <= N
len(mask) = N
mask[i] ∈ {0,1}
L ∈ {N, N-P, M}
reward ∈ 有限实数，或 None
```

三种 logprob 对齐分别是：

```text
N     全 token
N-P   response token
M     trainable token
```

Stage 1 Polar 数据采用 response-aligned logprobs。

### 12.1 为什么不把所有上游强制转换成一种 logprob shape

强制转换需要知道每个 logprob 的准确 token 对应。如果源信息不足，Adapter 只能猜或丢数据。

core contract 选择表达真实差异；具体 Trainer Adapter 将来再声明自己接受哪一种对齐。

### 12.2 当前存在的 alignment 歧义

如果：

```text
N-P = M
```

仅靠长度无法区分 response-aligned 和 trainable-only。真实 Day 2 response token 恰好全部 trainable，所以二者同为 99。

当前通过 Polar builder provenance 判断；未来可能需要显式 `logprob_alignment` 字段。

## 13. Capability 是 Stage 2 的资格代数

对单条 record：

```text
C(r) = 从真实非 None 字段派生的 capability 集合
```

对 batch：

```text
C(B) = ∩ C(r)
```

必须取交集，因为 batch capability 表示每条记录都能保证的能力。

### 13.1 Day 2 成功路径实际 capability

```text
TOKEN_IDS
ACTION_MASK
OLD_LOGPROBS
REWARD
VERIFIER_EVIDENCE
```

缺失：

```text
GROUP_ID
POLICY_VERSION
```

### 13.2 Day 2 fault capability

无 token、mask、reward、group、policy 和 verifier evidence，因此共同 capability 是空集。

### 13.3 为什么缺 capability 不是 task failure

`CAPABILITY_MISSING` 表示当前消费者不能运行。它可能来自：

- Producer 没有保存字段；
- fault 在相应阶段前结束；
- Trainer 配置要求比源格式更强。

它不说明模型答案对错。

## 14. 三个版本化输出对象

### 14.1 `RolloutBatch`

用于接收未经 Processor 分类的 canonical records：

- 允许混 task/group/policy；
- 允许缺训练字段；
- 拒绝重复 trajectory；
- 保存 Adapter 名称/版本和 batch checksum。

### 14.2 `TrainingReadyBatch`

构造时强制：

```text
所有 task_id 相同
所有 group_id 相同
所有 policy_version 相同
trajectory 不重复
每条记录满足 required capabilities
```

它保存 source batch checksum 和 processing versions。

但目前它只保证结构 readiness，不保证 reward variance、固定 group size 或 validity。这些尚待 Day 5。

### 14.3 `ResampleRequest`

表达：

```text
哪个 task/group/policy
缺几条
为什么缺
补采约束是什么
建议找哪个 source
```

它不执行 Polar 请求，保持核心无副作用和 producer-agnostic。

## 15. `AdapterResult` 为什么允许“部分可见、整体不放行”

JSONL 可能出现：

```text
line 1 valid
line 2 malformed
line 3 valid
```

结果保存 line 1/3 records 和 line 2 error，便于诊断；但：

```text
result.ok = false
result.to_batch() 拒绝
```

这同时满足：

- 合法部分不被诊断系统吞掉；
- 不完整输入不会静默进入训练。

## 16. JSONL Adapter 为什么是重要实现，不只是 demo

JSONL 提供一个与 Polar 无关的公共入口：

```text
rollout-jsonl/v1 envelope
→ RolloutRecord.from_dict
→ 更新当前 source envelope
→ capability gate
```

它证明核心 pipeline 并不要求所有 Producer 模仿 Polar Python API。

任何 Producer 最低成本接入方式都可以是：

```text
输出公共 JSONL
→ 使用相同 Processor
```

### 16.1 Roundtrip 的准确不变量

Polar record 导出 JSONL 再导入：

```text
semantic_dict 相同
source envelope 不同
```

因为直接来源从 Polar summary 变成 JSONL line。

## 17. Polar Adapter 如何编码 Stage 1 发现

核心映射：

```text
token_ids = prompt_ids + response_ids
prompt_token_count = len(prompt_ids)
loss_mask = [0] * len(prompt_ids) + source loss_mask
old_logprobs = source response_logprobs
reward = source trace.reward
```

### 17.1 两级信任检查

首先验证 fixture：

```text
目录安全
manifest schema
summary role
summary bytes
summary SHA256
```

然后由 `RolloutRecord` 验证 canonical 对齐：

```text
token 非负
mask 二值并等长
logprob 有限且长度合法
reward 有限
```

### 17.2 名称与事实冲突时如何处理

`calculator_success + reward=0`：

```text
保留 reward=0
返回 SOURCE_WARNING
不改写事实
```

### 17.3 为什么 component status 仍可能 UNKNOWN

Day 2 artifact 没提供足够稳定的 runtime/harness/backend 独立成功路径。Adapter 不根据 `COMPLETED` 猜三个组件全成功。

这是一种保守性：字段不漂亮，但不会伪造证据。

## 18. Stage 2 的测试结果与真实数据验证

在包含服务器真实 fixture 的干净 clone 中：

```text
python3 -m unittest discover -s tests -v
→ 77 tests OK
```

验证内容包括：

- record 不可变性；
- schema/enum/token/mask/logprob/reward/time 校验；
- checksum 确定性；
- capability 交集；
- mixed policy 拒绝；
- JSONL partial failure；
- Polar manifest tamper；
- Polar → JSONL 语义 roundtrip；
- 真实 Day 2 token 数；
- 真实 fault 不产生训练 capability。

真实 success-path 映射得到：

```text
tokens = 14945
prompt_token_count = 14846
response/logprobs = 99
reward = 0.0
```

本地脏 checkout 缺服务器真实 fixture 文件时，真实 fixture 测试会 skip；这不是远端没完成，而是本地 checkout 的 artifact 不完整。

## 19. Stage 2 的实现边界

已经实现：

```text
错误语义
确定性 JSON/checksum
RolloutRecord
Capability
RolloutBatch
TrainingReadyBatch
ResampleRequest
SourceAdapter Protocol
AdapterResult
JsonlSourceAdapter
PolarSourceAdapter
单元与真实 fixture 测试
```

尚未实现：

```text
FailureClassifier
SignalFilter
PolicyConsistentGroupBuilder
ProcessingReport/RejectedRecord
Slime Adapter
optimizer step
```

这就是为什么 Day 4 核心实现已完成，但整个项目不能称为完成。

---

# Part III：Stage 3——实现 Day 3 Coding/SWE 最小服务器执行包

## 20. Stage 3 为什么必须在 Stage 2 后继续补证据

Stage 1 只有：

```text
Calculator VALID_FAILURE
runtime INVALID_INFRASTRUCTURE
```

缺少：

```text
真实 VALID_SUCCESS
多轮 Coding trajectory
文件读取与修改
patch
clean replay
verifier timeout
复杂 tool event
```

如果此时直接实现 Day 5 classifier，我们只能对 success 和多轮 Coding 语义做大量假设。

Stage 3 的任务是产生三类 golden evidence：

```text
VALID_SUCCESS
VALID_FAILURE
INVALID_INFRASTRUCTURE
```

## 21. 为什么选择 Polar 官方 SWE-bench Verified 路径

锁定 commit 中已经提供：

```text
examples/swebench_verified/dataset.py
examples/swebench_verified/build_images.py
examples/swebench_verified/submit_swebench_tasks.py
src/polar/trajectory/evaluator/swebench_harness.py
src/polar/trajectory/evaluator/_patch_utils.py
```

该路径支持：

- `qwen_code` Harness；
- 官方 `build_task_request()`；
- SWE runtime image；
- patch extraction；
- `swebench_harness` evaluator；
- fresh runtime clean replay。

因此我们不自己重写 Coding Agent loop 或 evaluator。

### 21.1 Stage 3 的边界

实现/维护：

```text
候选任务确定性选择
运行前门槛
单 GPU topology/runbook
故障注入规则
staging evidence schema
fixture 打包与验证
字段审计模板
pilot report 模板
```

不实现：

```text
Polar Rollout Server
qwen_code Agent loop
SWE evaluator
Slime/Megatron
benchmark 全量评估
```

## 22. Candidate selector 的算法

代码文件：[`scripts/select_swebench_candidates.py`](../../scripts/select_swebench_candidates.py)

### 22.1 输入校验

每条候选必须有：

```text
非空 instance_id
完整 40 位 base_commit
非空 problem_statement
非空 FAIL_TO_PASS
可解析 PASS_TO_PASS
repository identity
```

`FAIL_TO_PASS`/`PASS_TO_PASS` 可以已经是 list，也可以是 JSON string；解析失败的任务直接排除。

### 22.2 为什么必须有 FAIL_TO_PASS

Coding task 的关键不是“测试能运行”，而是至少有一个测试在 base commit 失败，应用正确 patch 后应通过。

没有 FAIL_TO_PASS 的任务不能稳定证明修复行为。

### 22.3 排名公式

```text
test_count = len(FAIL_TO_PASS) + len(PASS_TO_PASS)
problem_chars = len(problem_statement)

ranking_score
= test_count * 1,000,000
+ min(problem_chars, 999,999)
```

这使测试数量拥有一级优先权，prompt 长度只在测试数量相同时近似打破平局。

原因：

- 测试越少，verifier 成本通常越低；
- prompt 越短，context 压力通常越低；
- 最后再按 `instance_id` 排序，确保稳定结果。

它不是预测模型成功率，只是构造低成本 pilot shortlist。

### 22.4 `distinct_repos`

启用后每个 repository 最多选择一个候选，避免前三个任务都被同一 repo 的构建问题阻塞。

它牺牲某些全局最低 score，换取环境多样性和风险分散。

### 22.5 为什么拒绝覆盖 output

候选 shortlist 是一次运行 provenance 的一部分。覆盖会让同一 run ID 的选择过程改变，因此输出已存在时直接失败。

### 22.6 selector 不证明什么

它不检查：

- runtime image 已存在；
- base commit 在容器中正确；
- tests 稳定；
- 模型能解题。

这些仍需 runbook 中的 runtime baseline。

## 23. Coding capture policy 是结果契约，不是普通配置

文件：[`configs/polar/coding/capture-policy.yaml`](../../configs/polar/coding/capture-policy.yaml)

### 23.1 三类 outcome 的严格定义

#### `coding_success`

```text
canonical_class = VALID_SUCCESS
synthetic_fault = false
verifier_completed = true
resolved = true
reward = 1
```

#### `coding_valid_failure`

```text
canonical_class = VALID_FAILURE
synthetic_fault = false
verifier_completed = true
resolved = false
reward = 0
```

#### `coding_invalid_infra`

```text
canonical_class = INVALID_INFRASTRUCTURE
必须有 timeout/crash/error signal
resolved = null
reward = null
```

它可以是自然故障，也可以是明确记录的 synthetic fault，但不能借 infrastructure fault 制造 reward 0 负样本。

### 23.2 Valid outcome 为什么必须有 replay

模型 trajectory 中 verifier 通过/失败可能依赖：

- 被 Agent 污染的 workspace；
- 未记录的临时文件；
- 前一次测试残留；
- runtime 中未提交副作用。

所以 success/failure 都要求：

```text
fresh clean workspace
→ apply captured patch
→ run verifier
→ replay outcome 与原 outcome 一致
```

成功需要 replay resolved=true；有效失败需要 replay resolved=false。

### 23.3 为什么 failure 也需要 clean replay

如果只 replay success，reward=0 可能来自偶发 evaluator 环境问题，却被当成稳定任务失败。clean replay 能证明相同 patch 在干净环境中仍然无法解决任务。

## 24. Coding verifier evidence schema

`verifier-evidence.json` 必须包含：

```text
schema_version
outcome_class
task_id / instance_id
base_commit
runtime_image_identity
synthetic_fault
test.command / timeout_seconds
result.verifier_completed
result.timed_out
result.crashed
result.exit_code
result.resolved
result.reward
provenance.source_file / json_path
patch_sha256
```

### 24.1 为什么保存 command 和 timeout

只保存 `resolved=true` 不足以复验。不同 test command 或 timeout 可以产生不同结果。

### 24.2 为什么保存 base commit 和 runtime image

同一 patch 对不同代码版本可能应用失败或得到不同测试结果。base commit 与 image identity 定义 replay 的起点。

### 24.3 为什么保存 provenance path

`verifier-evidence.json` 是从 raw Polar artifact 派生的规范化文件。必须能回答每个结论来自哪份源文件和 JSON path，避免派生摘要变成不可追踪的第二份 truth。

## 25. Stage 3 对通用 Packager 的扩展

昨晚没有另写 `package_coding_fixture.py`，而是扩展通用打包器支持：

```text
calculator_success
calculator_fault
coding_success
coding_valid_failure
coding_invalid_infra
```

### 25.1 Coding required roles

新增：

```text
verifier_evidence
patch
replay_evidence
fault_injection
task_metadata
```

### 25.2 为什么 valid 与 invalid 的文件要求不同

有效 outcome 要证明任务判断可重复，所以需要 replay。

infrastructure invalid 没有可信任务 outcome，强制 replay resolved true/false 反而会伪造结论；它需要 fault evidence，而 resolved/reward 保持 null。

## 26. Stage 3 对 Fixture Verifier 的扩展

### 26.1 Outcome class 与目录类型交叉校验

```text
coding_success       ↔ VALID_SUCCESS
coding_valid_failure ↔ VALID_FAILURE
coding_invalid_infra ↔ INVALID_INFRASTRUCTURE
```

目录名、manifest 和 evidence 三方必须一致。

### 26.2 Success 条件

```text
verifier_completed=true
timed_out=false
crashed=false
resolved=true
reward=1
```

### 26.3 Valid failure 条件

```text
verifier_completed=true
timed_out=false
crashed=false
resolved=false
reward=0
```

### 26.4 Infrastructure invalid 条件

必须至少有：

```text
timed_out=true
OR crashed=true
OR 非空 error_type
```

同时强制：

```text
resolved=None
reward=None
```

### 26.5 Patch checksum

`verifier-evidence.patch_sha256` 必须是 64 位小写 SHA256，并与实际 `patch.diff` 字节匹配。

这防止 summary 说测试的是 patch A，fixture 中却保存 patch B。

### 26.6 Replay 一致性

对 valid outcome：

```text
schema = coding-clean-replay/v1
clean_workspace = true
patch_applied 为 bool
verifier_completed 为 bool
resolved 与 fixture type 一致
```

当前 verifier 强制 clean workspace 和 resolved 一致；后续真实 fixture 还应人工审计 patch applied/verifier completed 的失败语义。

### 26.7 敏感信息扫描扩展到文本

Coding patch 和日志不是 JSON，因此除了 JSON key 扫描，还检查文本中的 private key header。

这仍是最低安全门，不代替人工审查 patch 是否含业务 secret。

## 27. 两类基础设施故障设计

Stage 3 要记录至少两种不同 failure plane。

### 27.1 Verifier timeout

只修改：

```text
evaluator.config.test_timeout
```

改为稳定的小正数，使 evaluator 明确超时。

它验证：

```text
Agent 可能完成了 rollout
但 verifier 没有给出可信 task outcome
→ 不能写 reward=0
```

### 27.2 Runtime prepare failure

只在 `runtime.prepare` 末尾增加非破坏性：

```text
exit 42
```

它验证：

```text
任务在 Agent 开始前失败
→ 无 model completion / verifier outcome
```

### 27.3 为什么需要两种 infra fault

它们发生在不同生命周期位置：

```text
runtime prepare failure
    rollout 前失败

verifier timeout
    rollout 后、任务判定阶段失败
```

一个 classifier 如果只会识别“没有 token”的 fault，就可能漏掉 verifier timeout 这种已有 trajectory 但 outcome 无效的情况。

### 27.4 Fault provenance

必须保存：

```text
synthetic_fault=true
原 payload SHA256
修改后 payload SHA256
唯一修改 JSON path
预期失败阶段
实际失败阶段
terminal status/error
```

这证明故障是可控实验，而不是服务器随机坏掉。

## 28. Stage 3 Runbook 的系统设计

文件：[`docs/runbooks/day-03-polar-coding.md`](../runbooks/day-03-polar-coding.md)

### 28.1 为什么只做 3 个候选的 pilot

目标不是 SWE-bench 分数，而是得到数据基础设施所需的三类 evidence。

策略：

```text
3 个 runtime baseline 候选
→ 每个 1–2 samples
→ 只对有希望产生 reward variance 的任务补采
→ 总补采上限 4
```

这样限制 GPU 成本，并避免为了得到 success 无限采样、最终无法复现预算。

### 28.2 为什么 success 必须来自真实模型

禁止：

- 人工 patch；
- gold patch；
- 把 Calculator execution success 冒充 Coding success。

否则 fixture 只能测试数据格式，不能证明真实 rollout 成功路径。

### 28.3 为什么至少一条必须是多轮 trajectory

数据基础设施最终面向 Agentic RL，而不是单轮 completion。多轮样本应覆盖：

```text
多个 Gateway completion
文件读取
代码修改
测试执行
最终 patch
```

这样才能观察 tool events、termination、patch 和 verifier 的真实边界。

### 28.4 为什么单卡、顺序提交

Stage 3 优先可诊断性：

- 只使用一张 GPU；
- `max_*_workers=1`；
- 候选顺序执行；
- 每个任务完成后立即保存 session/artifact 路径。

并发虽然更快，却会增加端口、日志、GPU OOM 和 session 对应关系的复杂度，不适合第一批 golden fixture。

### 28.5 为什么 context 设置与 Day 2 不完全相同

Day 2 实际 qwen_code request 输入约 14846 token、输出预算 8000，因此 8K/16K 配置曾遇到不足，最终需要 32768 context 才覆盖 Calculator 路径。

当前 Day 3 runbook 示例仍写 8192，服务器执行时必须以候选实际 prompt/token budget 做 preflight；如果官方 task request 超过配置，不能截断后声称正常结果，应调整并记录实际 context。

这也是 runbook 在真实执行前需要服务器 AI审计请求长度的原因。

### 28.6 精确 PID teardown

只停止本 run PID 文件记录的：

```text
Gateway
Rollout Server
SGLang
```

顺序发送 TERM，并先核对命令行。禁止 `pkill`，因为服务器可能有其他用户的同名进程。

Day 2 已遇到 8080 被其他 `server` 用户进程占用，因此“未知端口占用不杀，改用安全端口/停止本次运行”是实际经验，不是形式主义。

## 29. Stage 3 字段审计模板为什么仍然写 `missing`

文件：[`notes/polar-coding-field-map.md`](../../notes/polar-coding-field-map.md)

字段包括：

```text
identity/group/policy
model/tokenizer revision
prompt/output token IDs
mask/logprobs
tool call/result
patch
rollout/component/verifier status
test command/exit code/resolved/reward
runtime image/base commit
```

当前尚未运行真实 Coding rollout，因此表中写 `TBD/missing` 是正确状态。

模板的规则是：

```text
direct   源 artifact 直接存在
derived  有明确可复算算法
missing  未观察到
```

不能因为 Stage 2 contract 有某字段，就在 Stage 3 field map 中先宣称 Polar Coding artifact 一定提供它。

## 30. Pilot report 将来回答什么

文件：[`notes/task-pilot-report.md`](../../notes/task-pilot-report.md)

至少记录：

```text
3 个候选的 runtime baseline
实际 rollout 数量
success/failure/invalid 数量
token/turn/tool 数量
verifier latency
失败原因
是否适合 Day 6 训练
```

它不是 benchmark leaderboard，而是训练任务选择报告。

一个适合 Day 6 的任务应当：

- verifier 稳定；
- 运行时长可控；
- 基础模型在多次采样中能产生 success/failure 差异；
- reward variance 可形成；
- runtime 可重复构建。

## 31. Stage 3 当前准确完成状态

### 已实现并推送

- 官方 Polar Coding 路径和 commit 已审计；
- qwen_code/evaluator/patch/clean replay 路径已确认；
- 确定性候选选择器已实现并测试；
- Coding capture policy 已冻结；
- 通用 packager 已支持三类 Coding fixture；
- fixture verifier 已支持 outcome、patch、replay、infra null 语义；
- Day 3 runbook 已达到服务器可执行状态；
- 字段表和 pilot report 模板已准备；
- 两类非破坏性 infrastructure fault 已设计；
- 相关本地测试已纳入 77-test suite。

### 尚待服务器执行

- 3 个候选 runtime baseline；
- 真实 qwen_code 多轮 rollout；
- 真实 `VALID_SUCCESS`；
- 真实 `VALID_FAILURE`；
- verifier timeout 与 runtime prepare fault 实际运行；
- clean replay；
- 三套 Coding fixture 文件；
- 真实 field map 和 pilot report 填写。

因此准确表述是：

```text
Stage 3 implementation package = completed
Stage 3 GPU experiment = pending server execution
```

不能写成：

```text
Coding rollout 已跑通
```

---

# 代码伴读：把三个 Stage 与实际实现逐段对应起来

前面的 Part I–III 解释了系统为什么这样设计。本部分直接阅读昨晚提交的实际代码。每一小节都按相同结构展开：

```text
实际代码
→ 代码逐步做了什么
→ 它维护的工程不变量
→ 哪个测试证明这一行为
→ 它如何连接真实 Stage 结果
```

## A. Stage 1 实际代码：Packager 不是复制脚本

### A.1 fixture 类型如何决定证据要求

实际实现位于 `scripts/package_polar_fixture.py`：

```python
BASE_REQUIRED_FILES = {
    "request.json": "request",
    "response.json": "response",
    "summary.json": "summary",
}
CODING_REQUIRED_FILES = {
    "verifier-evidence.json": "verifier_evidence",
    "patch.diff": "patch",
}
CODING_REPLAY_FILES = {"replay.json": "replay_evidence"}


def _required_files_for_type(fixture_type: str) -> dict[str, str]:
    required = dict(BASE_REQUIRED_FILES)
    if fixture_type.startswith("coding_"):
        required.update(CODING_REQUIRED_FILES)
    if fixture_type in {"coding_success", "coding_valid_failure"}:
        required.update(CODING_REPLAY_FILES)
    return required
```

这段代码并不是按目录随意找文件，而是先从 outcome 类型推导最小证据集合。

对 Calculator：

```text
request + response + summary
```

对所有 Coding：

```text
公共三件套 + verifier evidence + patch
```

对有效 Coding outcome：

```text
再增加 clean replay
```

为什么 `coding_invalid_infra` 不强制 replay？因为它没有可信 task outcome。强行要求 `resolved=true/false` 会制造本应为 null 的任务结论。

### A.2 allowlist 是怎样真正执行的

`discover_source_files()` 最后不是直接返回发现结果，而是重新扫描 staging：

```python
actual_files: set[str] = set()
for path in source_dir.rglob("*"):
    if path.is_symlink():
        raise PackageError(f"symlink is not allowed in staging: {path}")
    if path.is_file():
        actual_files.add(_safe_relative_path(path, source_dir))
unexpected = sorted(actual_files - allowed_paths)
if unexpected:
    raise PackageError(
        "unexpected staging files; explicitly classify or remove them: "
        + ", ".join(unexpected)
    )
```

集合含义是：

```text
actual_files   staging 实际存在的一切
allowed_paths  代码明确识别并赋予 role 的文件

actual_files - allowed_paths
= 没有被证据契约解释的文件
```

只要差集非空就失败。这避免了下面这种危险实现：

```python
for path in source_dir.rglob("*"):
    shutil.copy(path, output)
```

后者可能把 token、内部日志、模型输出缓存或 secret 一起提交。

对应测试不是只看返回码，而是主动放入未知文件：

```python
def test_rejects_unclassified_staging_file(self):
    ...
    (source / "mystery.bin").write_bytes(b"unknown")

    with self.assertRaisesRegex(PackageError, "unexpected staging files"):
        package_fixture(...)
```

这证明 allowlist 不是文档约定，而是运行时强制行为。

### A.3 为什么先写临时目录，再写最终 fixture

实际打包核心：

```python
with tempfile.TemporaryDirectory(
    prefix=f".{output_dir.name}-package-", dir=output_dir.parent
) as temp_dir:
    package_root = Path(temp_dir) / "fixture"
    package_root.mkdir()
    for source_file in source_files:
        source_path = source_dir / source_file.relative_path
        destination = package_root / source_file.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)

    manifest = build_manifest(package_root, source_files, metadata)
    (package_root / "source-manifest.json").write_text(...)

    verification = verify_fixture(package_root, schema_path, ...)
    failures = [result for result in verification if result.status == FAIL]
    if failures:
        raise PackageError(...)

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(package_root.rglob("*")):
        ...
        if destination.exists():
            raise PackageError(f"refusing to overwrite: {relative.as_posix()}")
        shutil.copy2(path, destination)
```

控制流不是：

```text
边复制边验证
```

而是：

```text
完整建立临时 package
→ 生成 manifest
→ 验证临时 package
→ 只有全通过才发布到最终目录
```

发布后又执行：

```python
final_results = verify_fixture(output_dir, schema_path, ...)
final_failures = [result for result in final_results if result.status == FAIL]
if final_failures:
    raise PackageError(...)
```

这是双重验证：第一次证明将要发布的内容正确；第二次证明最终目录中的内容正确。

### A.4 打包器确实不修改原始 payload

对应测试保存源文件原始 bytes：

```python
original_response = (source / "response.json").read_bytes()

manifest = package_fixture(...)

self.assertEqual((output / "response.json").read_bytes(), original_response)
self.assertEqual((source / "response.json").read_bytes(), original_response)
self.assertNotIn(
    "old_logprobs",
    json.loads((output / "response.json").read_text()),
)
```

这个测试保护两个事实：

1. 打包过程不修改 source；
2. 打包过程不为了让数据“更完整”而注入 `old_logprobs`。

### A.5 manifest 的实际生成方式

```python
file_entries.append(
    {
        "path": source_file.relative_path,
        "role": source_file.role,
        "media_type": source_file.media_type,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
)
```

以及：

```python
"source": {
    "polar_commit": metadata.polar_commit,
    "model_id": metadata.model_id,
    "model_revision": metadata.model_revision,
    "tokenizer_revision": metadata.tokenizer_revision,
    "runtime_image_identity": metadata.runtime_image_identity,
    "harness": metadata.harness,
    "policy_version": metadata.policy_version,
},
```

文件 checksum 与运行环境 identity 同时存在，分别回答：

```text
这些 artifact 的精确字节是什么？
这些 artifact 由什么版本的系统产生？
```

只有前者不能复现实验；只有后者不能检测提交后的文件篡改。

## B. Stage 1 实际代码：Fixture Verifier 如何判断 Calculator

### B.1 success path 没有要求 reward=1

实际 `_validate_calculator_evidence()`：

```python
if fixture_type == "calculator_success":
    if status != "completed":
        errors.append("calculator_success requires summary.status=COMPLETED")
    if not isinstance(evaluation, dict) or not evaluation:
        errors.append("calculator_success requires evaluator evidence")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        errors.append("calculator_success requires a numeric evaluator outcome_reward")
    if not isinstance(report, dict) or not report:
        errors.append("calculator_success requires evaluation.report")
    elif (
        report.get("error_eval") is True
        or report.get("test_timeout") is True
        or report.get("failed_apply_patch") is True
    ):
        errors.append("calculator_success requires a normally completed evaluator path")
```

条件要求的是：

```text
session 完成
evaluator 实际运行
reward 是数值
report 存在
evaluator 没有 error/timeout/apply failure
```

没有：

```python
if reward != 1:
    fail
```

因此真实 `reward=0, resolved=false` 可以作为 execution-complete fixture 通过。

对应测试直接编码了昨晚真实发现：

```python
def test_accepts_completed_calculator_path_with_valid_task_failure(self):
    ...
    "outcome_reward": 0.0,
    "report": {
        "resolved": False,
        "empty_generation": True,
        "error_eval": False,
        "test_timeout": False,
    }
    ...
    self.assertEqual(semantic.status, PASS)
```

这不是 synthetic happy path，而是把真实 Day 2 语义变成 regression test。

### B.2 evaluator timeout 为什么不能混入 success path

相反测试把：

```python
"test_timeout": True
```

放入同样 `status=COMPLETED` 的 summary，并断言：

```python
self.assertEqual(
    result_for(results, "calculator_evidence").status,
    FAIL,
)
```

这证明顶层 `COMPLETED` 不会覆盖 evaluator failure。

### B.3 fault path 真实代码如何保护 null reward

```python
else:
    if status not in {"error", "timeout", "failed"}:
        errors.append("calculator_fault requires terminal infrastructure error status")
    if reward is not None:
        errors.append("calculator_fault must not map infrastructure failure to reward")
    if not _is_nonempty_string(summary.get("error")):
        errors.append("calculator_fault requires an explicit error message")
```

这里检查的不是 reward 是否等于 0，而是必须完全不存在可信 reward：

```text
reward is None
```

## C. Stage 2 实际代码：Canonical JSON 与不可变性

### C.1 实际冻结函数

`src/contracts/_json.py`：

```python
def freeze_json(value: Any, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractValidationError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        if any(not isinstance(key, str) for key in value):
            raise ContractValidationError(f"{path} contains a non-string object key")
        for key in sorted(value):
            frozen[key] = freeze_json(value[key], f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            freeze_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise ContractValidationError(...)
```

逐分支解释：

- JSON 标量直接保留；
- float 额外拒绝 NaN/Infinity；
- mapping 先验证 key，再排序、递归冻结；
- sequence 转 tuple，但明确排除字符串与 bytes；
- 任何 Python 私有对象拒绝进入 metadata。

对应测试不是只检查 dataclass frozen，而是修改嵌套 mapping：

```python
with self.assertRaises(TypeError):
    record.opaque_metadata["new"] = "value"
```

### C.2 实际 canonical serialization

```python
def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        thaw_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
```

对应测试用不同 key 插入顺序构造记录：

```python
first = make_record(opaque_metadata={"b": 2, "a": 1})
second = make_record(opaque_metadata={"a": 1, "b": 2})
self.assertEqual(first.checksum, second.checksum)
```

这把“确定性”从文档要求变成了可运行断言。

## D. Stage 2 实际代码：`RolloutRecord` 校验是怎样写的

### D.1 真实 training payload 字段

```python
token_ids: tuple[int, ...] | None = None
prompt_token_count: int | None = None
action_mask: tuple[int, ...] | None = None
loss_mask: tuple[int, ...] | None = None
old_logprobs: tuple[float, ...] | None = None
reward: float | None = None
```

所有字段允许 `None`，但它们之间的组合受到严格约束。

### D.2 mask 约束原文

```python
if action_mask is not None and loss_mask is not None:
    raise ContractValidationError("provide action_mask or loss_mask, not both")
mask = action_mask if action_mask is not None else loss_mask
if mask is not None and token_ids is None:
    raise ContractValidationError("a mask cannot be present without token_ids")
if token_ids is not None and mask is not None and len(token_ids) != len(mask):
    raise ContractValidationError("token_ids and mask must have the same length")
```

这个控制流允许：

```text
token 有、mask 无
token 与一种 mask 同时有
token/mask 都无
```

拒绝：

```text
两种 mask 同时存在
mask 脱离 token
mask/token 错位
```

为什么 token 有而 mask 可以无？因为 record 层要忠实表示 Producer 只保存 token 的情况；缺 mask 会体现在 capability gate，而不是让整条 evidence 无法保存。

### D.3 prompt boundary 原文

```python
if self.prompt_token_count is not None:
    if isinstance(self.prompt_token_count, bool) or not isinstance(
        self.prompt_token_count, int
    ):
        raise ContractValidationError("prompt_token_count must be an integer")
    if token_ids is None:
        raise ContractValidationError(
            "prompt_token_count cannot be present without token_ids"
        )
    if not 0 <= self.prompt_token_count <= len(token_ids):
        raise ContractValidationError(
            "prompt_token_count must be between zero and token_ids length"
        )
```

特意拒绝 bool 是因为 Python 中：

```python
isinstance(True, int) is True
```

若不单独排除，`prompt_token_count=True` 会被错误解释为 1。

### D.4 logprob 对齐原文

```python
if old_logprobs is not None:
    if token_ids is None:
        raise ContractValidationError(
            "old_logprobs cannot be present without token_ids"
        )
    valid_lengths = {len(token_ids)}
    if mask is not None:
        valid_lengths.add(sum(mask))
    if self.prompt_token_count is not None:
        valid_lengths.add(len(token_ids) - self.prompt_token_count)
    if len(old_logprobs) not in valid_lengths:
        raise ContractValidationError(
            "old_logprobs length must match all tokens or trainable tokens"
        )
```

`valid_lengths` 实际就是前文公式：

```text
{N, M, N-P}
```

测试分别覆盖：

```python
full = make_record(old_logprobs=(-0.4, -0.3, -0.2, -0.1))
trainable = make_record(old_logprobs=(-0.3, -0.2, -0.1))
response_aligned = make_record(
    loss_mask=(0, 1, 0, 1),
    prompt_token_count=1,
    old_logprobs=(-0.3, 0.0, -0.1),
)
```

并用长度 1 的错误输入确认拒绝。

### D.5 missing 不被补造的测试原文

```python
record = make_record(
    group_id=None,
    policy_version=None,
    token_ids=None,
    prompt_token_count=None,
    loss_mask=None,
    old_logprobs=None,
    reward=None,
    verifier_evidence_ref=None,
    tool_events=None,
)

self.assertEqual(capabilities_for_record(record), frozenset())
self.assertIsNone(record.token_ids)
self.assertIsNone(record.old_logprobs)
```

这直接证明 constructor 没有把空数据变成：

```text
token_ids=()
old_logprobs=()
reward=0
```

## E. Stage 2 实际代码：Capability 与 Batch Gate

### E.1 record capability 是怎样派生的

```python
def capabilities_for_record(record: "RolloutRecord") -> frozenset[Capability]:
    available: set[Capability] = set()
    if record.token_ids is not None:
        available.add(Capability.TOKEN_IDS)
    if record.action_mask is not None or record.loss_mask is not None:
        available.add(Capability.ACTION_MASK)
    if record.old_logprobs is not None:
        available.add(Capability.OLD_LOGPROBS)
    if record.reward is not None:
        available.add(Capability.REWARD)
    if record.group_id is not None:
        available.add(Capability.GROUP_ID)
    if record.policy_version is not None:
        available.add(Capability.POLICY_VERSION)
    ...
    return frozenset(available)
```

注意每个判断都使用：

```python
is not None
```

而不是 truthiness：

```python
if record.reward:
```

否则合法 `reward=0.0` 会被误判为没有 REWARD capability。

### E.2 batch 交集原文

```python
def common_capabilities(records: Iterable["RolloutRecord"]) -> frozenset[Capability]:
    iterator = iter(records)
    try:
        common = set(capabilities_for_record(next(iterator)))
    except StopIteration:
        return frozenset()
    for record in iterator:
        common.intersection_update(capabilities_for_record(record))
    return frozenset(common)
```

实现从第一条 capability 开始，不断 `intersection_update`，正对应数学集合交。

测试实际构造第二条缺 old logprob 的记录：

```python
without_logprobs = make_record(..., old_logprobs=None)
batch = RolloutBatch.from_records((complete, without_logprobs), ...)
self.assertNotIn(Capability.OLD_LOGPROBS, batch.capabilities)
self.assertIn(Capability.TOKEN_IDS, batch.capabilities)
```

### E.3 TrainingReadyBatch 的 policy gate 原文

```python
for record in self.records:
    if record.trajectory_id in trajectory_ids:
        raise ContractValidationError(...)
    trajectory_ids.add(record.trajectory_id)
    if (
        record.task_id != self.task_id
        or record.group_id != self.group_id
        or record.policy_version != self.policy_version
    ):
        raise ContractValidationError(
            "all training records must match task_id, group_id, and policy_version"
        )
    require_capabilities(
        capabilities_for_record(record),
        self.required_capabilities,
        context=f"trajectory {record.trajectory_id}",
    )
```

校验发生在每条 record 上，不只是看 batch common capabilities。这让错误能带上具体 trajectory context。

对应 mixed-policy test：

```python
with self.assertRaisesRegex(ContractValidationError, "must match"):
    TrainingReadyBatch(
        records=(make_record(policy_version="policy-v1"),),
        policy_version="policy-v0",
        ...
    )
```

## F. Stage 2 实际代码：AdapterResult 的整体放行逻辑

### F.1 不允许 Adapter 虚报 capability

```python
guaranteed = common_capabilities(self.records)
if not self.capabilities.issubset(guaranteed):
    raise ValueError(
        "AdapterResult cannot declare capabilities missing from a record"
    )
```

即使具体 Adapter 实现错误地声称全部字段都有，结果对象仍会二次阻止。

### F.2 有 error 时拒绝 batch

```python
def to_batch(self, *, adapter_name: str, adapter_version: str) -> RolloutBatch:
    if self.errors:
        raise AdapterConversionError(
            f"{adapter_name} produced {len(self.errors)} error(s); "
            "refusing batch conversion"
        )
    return RolloutBatch.from_records(...)
```

JSONL partial failure test：

```python
result = JsonlSourceAdapter().convert("{not-json}\n" + valid_line + "\n")

self.assertFalse(result.ok)
self.assertEqual(len(result.records), 1)
self.assertEqual(len(result.errors), 1)
with self.assertRaises(AdapterConversionError):
    result.to_batch(adapter_name="jsonl", adapter_version="v1")
```

这就是“部分可见、整体不放行”的实际证据。

## G. Stage 2 实际代码：JSONL Source Envelope

### G.1 导出原文

```python
def dumps_jsonl(records: Iterable[RolloutRecord]) -> str:
    lines = []
    for record in records:
        envelope = {
            "schema_version": JSONL_SCHEMA_VERSION,
            "record": record.to_dict(),
        }
        lines.append(canonical_json_bytes(envelope).decode("utf-8"))
    return "" if not lines else "\n".join(lines) + "\n"
```

这里有三个确定性条件：

- record 顺序保持；
- 每行 canonical JSON；
- 非空文件统一以 newline 结尾。

### G.2 导入时替换 source envelope

```python
source_record_id = f"line:{line_number}"
...
record = record.with_source_envelope(
    source_type=self.name,
    source_record_id=source_record_id,
    source_payload_ref=source_payload_ref,
    source_payload_sha256=sha256_bytes(raw_line.encode("utf-8")),
)
```

对应 roundtrip test：

```python
self.assertEqual(restored.semantic_dict(), original.semantic_dict())
self.assertEqual(restored.source_type, "jsonl")
self.assertEqual(restored.source_record_id, "line:1")
self.assertNotEqual(
    restored.source_payload_sha256,
    original.source_payload_sha256,
)
```

测试明确要求语义相同、来源不同，而不是含糊地只比较对象是否相等。

## H. Stage 2 实际代码：Polar 字段映射

### H.1 summary 信任检查原文

```python
summary_bytes = summary_path.read_bytes()
summary_sha = hashlib.sha256(summary_bytes).hexdigest()
if summary_entry.get("bytes") != len(summary_bytes):
    raise ContractValidationError(
        "summary.json size does not match manifest"
    )
if summary_entry.get("sha256") != summary_sha:
    raise ContractValidationError(
        "summary.json checksum does not match manifest"
    )
```

对应 tamper test 先生成合法 fixture，再直接写坏 summary：

```python
(fixture / "summary.json").write_text("{}", encoding="utf-8")
result = PolarSourceAdapter().convert(fixture)
self.assertFalse(result.ok)
self.assertEqual(result.errors[0].code, ErrorCode.CONTRACT_INVALID)
```

### H.2 token 映射原文

```python
prompt_ids = self._int_list(trace.get("prompt_ids"), "prompt_ids")
response_ids = self._int_list(trace.get("response_ids"), "response_ids")
response_mask = self._int_list(trace.get("loss_mask"), "loss_mask")
if len(response_mask) != len(response_ids):
    raise ContractValidationError(
        "Polar loss_mask must match response_ids length"
    )
response_logprobs = self._float_list_or_none(
    trace.get("response_logprobs"), "response_logprobs"
)
if response_logprobs is not None and len(response_logprobs) != len(response_ids):
    raise ContractValidationError(
        "Polar response_logprobs must match response_ids length"
    )
token_ids = tuple(prompt_ids + response_ids)
loss_mask = tuple([0] * len(prompt_ids) + response_mask)
```

然后构造：

```python
RolloutRecord(
    token_ids=token_ids,
    prompt_token_count=len(prompt_ids),
    loss_mask=loss_mask,
    old_logprobs=response_logprobs,
    reward=trace.get("reward"),
    ...
)
```

对应 synthetic mapping test：

```python
self.assertEqual(record.token_ids, (10, 11, 20, 21, 22))
self.assertEqual(record.prompt_token_count, 2)
self.assertEqual(record.loss_mask, (0, 0, 1, 0, 1))
self.assertEqual(record.old_logprobs, (-0.1, 0.0, -0.2))
```

对应真实 fixture test：

```python
self.assertEqual(record.prompt_token_count, 14846)
self.assertEqual(len(record.token_ids or ()), 14945)
self.assertEqual(len(record.loss_mask or ()), 14945)
self.assertEqual(len(record.old_logprobs or ()), 99)
self.assertEqual(record.reward, 0.0)
```

所以文档中的 14846/14945/99 不是估计，而是实际 golden assertion。

### H.3 runtime fault 不造 reward 的实际构造

```python
return RolloutRecord(
    trajectory_id=session_id,
    task_id=task_id,
    ...
    rollout_status=self._rollout_status(summary.get("status")),
    runtime_status=runtime_status,
    harness_status=ComponentStatus.NOT_RUN,
    model_backend_status=ComponentStatus.NOT_RUN,
    verifier_status=VerifierStatus.NOT_RUN,
    source_payload_ref=str(fixture_dir / "summary.json"),
    source_payload_sha256=summary_sha,
    ...
)
```

构造调用没有传：

```text
token_ids
loss_mask
old_logprobs
reward
```

因此自然保留默认 `None`。真实 fault test 断言：

```python
self.assertEqual(record.runtime_status, ComponentStatus.FAILED)
self.assertIsNone(record.reward)
self.assertFalse(result.capabilities)
```

## I. Stage 3 实际代码：Candidate Ranking

### I.1 可复现性 gate 原文

```python
if not isinstance(instance_id, str) or not instance_id.strip():
    continue
if not isinstance(base_commit, str) or not FULL_SHA.fullmatch(base_commit):
    continue
if not isinstance(problem, str) or not problem.strip():
    continue
if fail_to_pass is None or not fail_to_pass or pass_to_pass is None:
    continue
```

这段代码没有尝试修补短 SHA、空 FAIL_TO_PASS 或坏 JSON test list。候选选择的目标是可重复 pilot，不是最大化保留数据集行数。

对应测试：

```python
invalid_sha = {**valid, "instance_id": "bad-sha", "base_commit": "abc"}
no_failing_test = {**valid, "instance_id": "no-fail", "FAIL_TO_PASS": []}

ranked = rank_candidates([invalid_sha, no_failing_test, valid])

self.assertEqual(
    [item["instance_id"] for item in ranked],
    ["owner__valid-1"],
)
```

### I.2 排名原文

```python
test_count = len(fail_to_pass) + len(pass_to_pass)
problem_chars = len(problem)
score = test_count * 1_000_000 + min(problem_chars, 999_999)
```

以及稳定排序：

```python
return sorted(
    candidates,
    key=lambda item: (item["ranking_score"], item["instance_id"]),
)
```

乘一百万确保 `test_count` 的一个单位差异大于 prompt 长度项的最大贡献。

测试输入刻意让一个 prompt 更长但 tests 更少，确认 tests 优先；tests 相同再按 prompt 长度：

```python
self.assertEqual(
    [item["instance_id"] for item in ranked],
    ["owner__shorter-1", "owner__short-1", "owner__large-1"],
)
```

### I.3 distinct repository 原文

```python
selected: list[dict[str, Any]] = []
seen_repos: set[str] = set()
for candidate in ranked:
    if candidate["repo"] in seen_repos:
        continue
    selected.append(candidate)
    seen_repos.add(candidate["repo"])
    if len(selected) == limit:
        break
```

这是贪心选择：按全局排名顺序取每个 repo 的第一个最佳候选。

## J. Stage 3 实际代码：Coding Outcome Verifier

### J.1 fixture type 与 outcome class 映射原文

```python
expected_outcome = {
    "coding_success": "VALID_SUCCESS",
    "coding_valid_failure": "VALID_FAILURE",
    "coding_invalid_infra": "INVALID_INFRASTRUCTURE",
}[fixture_type]
if evidence.get("outcome_class") != expected_outcome:
    errors.append(f"outcome_class must be {expected_outcome}")
```

测试主动把 success evidence 改成 failure label：

```python
evidence["outcome_class"] = "VALID_FAILURE"
...
self.assertEqual(result_for(results, "coding_evidence").status, FAIL)
```

### J.2 success/failure/infra 三分支原文

```python
if fixture_type == "coding_success":
    if (
        not result.get("verifier_completed")
        or result.get("timed_out")
        or result.get("crashed")
    ):
        errors.append("VALID_SUCCESS requires a normally completed verifier")
    if resolved is not True or reward != 1:
        errors.append("VALID_SUCCESS requires resolved=true and reward=1")
elif fixture_type == "coding_valid_failure":
    if (
        not result.get("verifier_completed")
        or result.get("timed_out")
        or result.get("crashed")
    ):
        errors.append("VALID_FAILURE requires a normally completed verifier")
    if resolved is not False or reward != 0:
        errors.append("VALID_FAILURE requires resolved=false and reward=0")
else:
    error_type = result.get("error_type")
    has_infra_signal = (
        bool(result.get("timed_out"))
        or bool(result.get("crashed"))
        or _is_nonempty_string(error_type)
    )
    if not has_infra_signal:
        errors.append(
            "INVALID_INFRASTRUCTURE requires timeout, crash, or error_type"
        )
    if resolved is not None or reward is not None:
        errors.append(
            "INVALID_INFRASTRUCTURE must keep resolved and reward null"
        )
```

这里的深层结构是二维判断：

```text
Verifier 是否正常完成？
任务是否 resolved？
```

不是只读 reward：

```text
正常完成 + resolved=true  → valid success
正常完成 + resolved=false → valid failure
未正常完成                 → invalid infrastructure
```

### J.3 防止 infra reward=0 的真实测试

```python
builder = FixtureBuilder(Path(temp_dir), "coding_invalid_infra")
evidence["result"]["reward"] = 0.0
...
self.assertEqual(evidence_result.status, FAIL)
self.assertIn("reward null", evidence_result.message)
```

这个测试是未来 FailureClassifier 最核心的 golden negative case 之一。

### J.4 patch SHA 与 replay 的实际校验

```python
expected_patch_sha = evidence.get("patch_sha256")
if not SHA256_PATTERN.fullmatch(str(expected_patch_sha or "")):
    errors.append(...)
elif patch_path.is_file() and sha256_file(patch_path) != expected_patch_sha:
    errors.append(
        "verifier evidence patch_sha256 does not match patch.diff"
    )
```

Replay：

```python
if replay.get("schema_version") != "coding-clean-replay/v1":
    errors.append(...)
if replay.get("clean_workspace") is not True:
    errors.append("replay.clean_workspace must be true")
if replay.get("resolved") is not (fixture_type == "coding_success"):
    errors.append("replay.resolved does not match fixture outcome")
```

测试同时修改实际 patch 和 replay outcome：

```python
(builder.root / "patch.diff").write_text("changed", encoding="utf-8")
replay["resolved"] = False
...
self.assertEqual(evidence_result.status, FAIL)
self.assertIn("patch_sha256", evidence_result.message)
```

这证明 verifier 不是只检查“文件存在”，而是在检查文件之间的语义引用关系。

## K. 三个 Stage 的代码—测试—真实结果对应表

| Stage | 实际函数/类 | 关键测试 | 当前真实证据 |
|---|---|---|---|
| 1 | `discover_source_files` | `test_rejects_unclassified_staging_file` | Day 2 fixture 只包含审核文件 |
| 1 | `package_fixture` | `test_packages_and_verifies_success_fixture_without_mutating_payload` | 原 artifact 未被补字段 |
| 1 | `_validate_calculator_evidence` | `test_accepts_completed_calculator_path_with_valid_task_failure` | reward 0 execution-complete 被正确接受 |
| 1 | `verify_fixture` | checksum/path/secret tests | 两套真实 fixture 通过验证 |
| 2 | `RolloutRecord.__post_init__` | token/mask/logprob tests | 14945/14846/99 真实映射成立 |
| 2 | `common_capabilities` | intersection test | group/policy 缺失被暴露 |
| 2 | `TrainingReadyBatch` | mixed policy/missing capability tests | Day 2 record 被正确挡在训练外 |
| 2 | `JsonlSourceAdapter` | partial error/roundtrip tests | 公共离线入口可用 |
| 2 | `PolarSourceAdapter._trace_record` | real fixture mapping test | response logprob 映射已验证 |
| 2 | `_empty_failure_record` | real fault test | infra fault reward 保持 None |
| 3 | `rank_candidates` | ranking/invalid row tests | 执行算法已准备，真实 shortlist 待服务器 |
| 3 | `_validate_coding_evidence` | all-three-outcomes/mismatch tests | 合成契约通过，真实 Coding fixture 待服务器 |
| 3 | Coding packager roles | coding package tests | 打包路径已实现，真实 artifact 待采集 |
| 3 | patch/replay validation | mismatch tests | 验证规则已实现，真实 replay 待执行 |

这张表体现准确完成边界：Stage 1/2 已有真实 Day 2 evidence 支撑；Stage 3 的代码和合成 contract test 已完成，但对应真实 Coding evidence 仍待服务器。

---

# Part IV：三个 Stage 如何组成一个完整工程闭环

## 32. 从 Stage 1 到 Stage 2：artifact 反向塑造 schema

```text
Stage 1 观察                     Stage 2 设计
────────────────────────────────────────────────────
prompt_ids + response_ids       prompt_token_count
response loss_mask              full-sequence mask
response_logprobs               old_logprobs + provenance
COMPLETED + reward=0             lifecycle/outcome 分离
runtime fault + no reward        missing 保持 None
group/policy 不存在              capability absence
fixture bytes + manifest         source checksum/lineage
```

这证明 schema 是从真实数据中长出来的，而不是先验想象。

## 33. 从 Stage 2 到 Stage 3：contract 反向暴露证据缺口

```text
Stage 2 需要                      Stage 3 补证据
────────────────────────────────────────────────────
VALID_SUCCESS                     coding_success
VALID_FAILURE                     coding_valid_failure
verifier ERROR/TIMEOUT            coding_invalid_infra
完整 tool behavior                多轮 qwen_code trajectory
可信 patch outcome                patch.diff + SHA256
可重复 verifier                   clean replay
稳定 policy/group 来源            Coding field audit
```

contract 不仅接收数据，也告诉我们下一次实验应该采什么。

## 34. 三层验证不能互相替代

```text
Stage 1/3 fixture verifier
    验证 artifact 包完整、安全、checksum 与 outcome evidence 自洽
            │
            ▼
Stage 2 contract validation
    验证 canonical token/mask/logprob/reward/status 结构自洽
            │
            ▼
未来 Day 5 Processor
    判断 validity、signal、group/policy consistency
            │
            ▼
未来 Day 6 Trainer Adapter
    判断具体 loss 所需字段并转换 Sample
```

一个 fixture checksum 正确，不代表它可以训练。

一个 RolloutRecord 结构合法，不代表它是有效任务样本。

一个 group policy 一致，不代表 reward 有 variance。

## 35. 昨晚真正完成的工程价值

如果只按文件数量看，会误以为昨晚只是写了 dataclass、JSON 校验和 runbook。实际上完成了三类更重要的工程边界。

### 35.1 证据边界

```text
raw server artifact
→ reviewed staging
→ allowlisted package
→ manifest/checksum fixture
```

任何提交到仓库的 evidence 都能定位来源、版本和精确字节。

### 35.2 语义边界

```text
Producer-private JSON
→ explicit Source Adapter
→ producer-agnostic RolloutRecord
```

Polar 字段不扩散到后续 Processor。

### 35.3 训练资格边界

```text
字段存在性
→ Capability
→ RolloutBatch
→ 更严格 TrainingReadyBatch
```

系统不再通过“字段看起来有默认值”判断可训练性。

## 36. 当前总体状态应该怎样表述

最准确的状态是：

```text
Stage 1：完成
    Day 2 真实 Calculator fixture 已取得、验证和语义审计

Stage 2：核心实现完成，阶段验收部分依赖 Stage 3
    contract/capability/JSONL/Polar Adapter 已实现并测试
    task-level success 与 Coding golden coverage 待真实 Stage 3 artifact

Stage 3：实现完成，实验执行待服务器
    所有本地代码、策略、验证和 runbook 已就绪
    真实 GPU Coding rollout 尚未执行
```

如果把“代码实现”与“服务器实验”分开，三个 Stage 的完成关系就不会混乱。

## 37. 你读完后应能回答的深度问题

### Stage 1

1. `calculator_success` 为什么实际上是 `VALID_FAILURE`？
2. 为什么 runtime prepare failure 的 reward 必须是 null？
3. 为什么字段名没有 `old_logprobs`，仍可存在 sampled-policy logprob 语义？
4. 为什么 manifest 要同时保存 model revision、tokenizer revision 和 Polar commit？
5. Packager 为什么拒绝未知 staging 文件，而不是全部复制？
6. 为什么 fixture verifier 不自动修复 checksum？

### Stage 2

7. `RolloutRecord` 为什么允许缺 group/policy？
8. `TrainingReadyBatch` 为什么不允许缺 group/policy？
9. 为什么 capability 取 batch records 的交集？
10. `reward=0` 与 `reward=None` 如何影响后续 classifier？
11. 为什么 JSONL roundtrip 要替换 source envelope？
12. 为什么有部分合法 records 的 AdapterResult 仍拒绝 `to_batch()`？
13. 为什么 response length 不能用 `sum(mask)` 推断？
14. 为什么当前 TrainingReadyBatch 仍不能替代 SignalFilter？

### Stage 3

15. 为什么 success 和 valid failure 都要 clean replay？
16. 为什么需要 verifier timeout 和 runtime prepare failure 两种 fault？
17. Candidate ranking 为什么让 test count 压倒 prompt length？
18. 为什么至少一条 fixture 必须来自真实多轮 Agent trajectory？
19. 为什么不能用 gold patch 制作 coding_success？
20. Stage 3“实现完成”和“实验完成”的边界是什么？

## 38. 一句话总结昨晚的三阶段工作

> Stage 1 用真实 Polar Calculator artifact 建立了“什么是可信事实”；Stage 2 把这些事实编码成不依赖 Polar、不会伪造缺失训练字段的 canonical contract 与 Adapter；Stage 3 则实现了取得 Coding success、valid failure 和 infrastructure-invalid 三类更完整证据的最小服务器执行系统。三者共同完成了从原始 rollout 证据到可进入后续数据处理器之前的可信基础层，但真实 Coding GPU 实验和 Day 5 Processor 仍是下一步。
