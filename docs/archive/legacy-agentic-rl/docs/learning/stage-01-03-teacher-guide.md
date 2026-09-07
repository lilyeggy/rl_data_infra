# 从零读懂昨晚的 Stage 1–3：老师带读版

> 历史快照说明（2026-08-12）：本文保留 Polar rollout、fixture 和旧 training contract 的教学细节；其中后续 GRPO 路线已被 Multi-Harness Execution Data Plane 替代。当前路线见 [`project-reading-order.md`](project-reading-order.md)。

> 这份文档是第一阅读材料。它不假设你已经理解 fixture、manifest、packager、verifier、Adapter 或 capability。
> 阅读目标不是记住所有类名，而是能够回答：“这段前置工作到底解决了什么问题？输入是什么？输出是什么？对应代码在哪里？它怎样服务于最终的数据基础设施？”
> 完整技术细节可在读懂本文后，再查看 `stage-01-03-implementation-deep-dive.md`。

## 第一课开始前：先看最终目标，再看为什么有这些前置工作

你已经知道我们最终要实现一个 data infra 模块：

```text
Polar 或其他系统产生 Agent rollout
        ↓
统一成公共数据格式
        ↓
剔除基础设施故障
        ↓
筛选有训练信号的数据
        ↓
组成同 policy 的 GRPO batch
        ↓
交给 Trainer
```

你现在困惑的是：

> 既然目标是做中间的数据模块，为什么前面在写 fixture、manifest、packager、verifier、runbook？这些看起来不像核心模块。

答案是：我们的模块要处理 Polar 产生的真实数据，但一开始我们甚至不知道 Polar 的真实数据到底长什么样。

如果没有真实数据就直接写 Adapter，我们很容易写出这样的假代码：

```python
record.old_logprobs = polar_result["old_logprobs"]
```

但实际 Polar artifact 里没有字面名为 `old_logprobs` 的字段，它叫 `response_logprobs`，而且只有在确认它代表采样 policy 的 response logprob 后，才能映射成 canonical `old_logprobs`。

再比如，我们可能看到一个目录叫 `calculator_success`，就写：

```python
record.reward = 1.0
```

但真实 artifact 里：

```text
reward = 0.0
resolved = false
```

所以这些“前置工作”的本质是：

```text
先取得真实样本
→ 证明样本没有被改坏
→ 看清每个字段的真实语义
→ 再写核心转换代码
```

这和训练一个视觉模型前先检查原始图片、标签和数据集版本是同一个道理。

## 一张图看懂昨晚三个 Stage

```text
Stage 1：取得并审计真实 Calculator 数据

服务器 raw artifact
    ↓ 只挑需要的文件并脱敏
staging 目录
    ↓ packager
带 manifest/checksum 的 fixture
    ↓ fixture verifier
可信、可重复使用的测试样本


Stage 2：用真实样本实现核心数据边界

Polar fixture
    ↓ PolarSourceAdapter
RolloutRecord
    ↓ capability 检查
RolloutBatch / TrainingReadyBatch


Stage 3：准备采集更完整的 Coding 数据

筛选便宜、可复现的 SWE 任务
    ↓
运行真实 qwen_code rollout
    ↓
取得 success / valid failure / infra failure
    ↓
保存 patch、verifier evidence、clean replay
```

昨晚的真实完成状态是：

```text
Stage 1：真实服务器结果已完成
Stage 2：核心代码已完成
Stage 3：执行代码和操作方案已完成，真实 GPU Coding rollout 待服务器执行
```

---

# 第一课：先把容易混淆的名词讲明白

## 1. Rollout 是什么

给 Coding Agent 一个任务：

```text
“修复仓库里的某个 bug”
```

Agent 可能经历：

```text
读题
→ 查看文件
→ 调用搜索工具
→ 修改代码
→ 运行测试
→ 再修改代码
→ 提交最终答案
```

这一整段执行轨迹就叫一个 rollout。

它不只包含模型最终回答，还可能包含：

- prompt token；
- response token；
- 多次模型 completion；
- tool call 和 tool result；
- 文件修改；
- patch；
- runtime 状态；
- verifier 结果；
- reward。

## 2. Artifact 是什么

Artifact 就是一次实际运行保存下来的文件。

例如：

```text
request.json      提交给 Polar 的任务请求
response.json     Polar 返回的终端响应
summary.json      trajectory、token、reward、状态摘要
gateway.log       Gateway 日志
runtime.log       runtime 日志
patch.diff        Agent 最终修改
```

Artifact 是“发生过什么”的证据。

## 3. Raw artifact 是什么

Raw 表示服务器原始保存的内容，没有为了提交 Git 而整理。

它可能：

- 非常大；
- 包含内部绝对路径；
- 包含主机名、用户名；
- 包含 authorization header；
- 包含大量与测试无关的日志；
- 文件命名和结构由 Polar 决定。

因此 raw artifact 通常留在服务器，不直接提交仓库。

## 4. Staging 目录是什么

Staging 是从 raw artifact 中挑出的、经过人工或脚本检查的待打包目录。

可以把它理解成“准备提交的候选文件夹”：

```text
raw/
├── 大量原始文件
├── 内部日志
├── 临时缓存
└── 可能有敏感信息

staging/
├── request.json
├── response.json
├── summary.json
└── 已审核的必要日志
```

为什么不让 packager 直接读取整个 raw 目录？因为 packager 无法知道所有未知文件是否安全、是否需要。

## 5. Fixture 是什么

Fixture 是测试使用的固定输入样本。

我们的单元测试不能每次都：

```text
启动 GPU
启动 SGLang
启动 Polar
运行 Agent
等待 evaluator
```

那样一次测试可能要几十分钟，而且结果会随采样变化。

所以我们把一次真实运行的关键产物固定下来：

```text
tests/fixtures/polar/calculator_success/
tests/fixtures/polar/calculator_fault/
```

以后测试直接读取这些文件，就能反复验证 Adapter，而不需要 GPU。

这就是 fixture 的作用：

> 把一次昂贵、随机、依赖服务器的真实运行，变成一个便宜、固定、可以反复使用的测试输入。

## 6. Golden fixture 是什么

Golden 表示它不是随便编造的 JSON，而是经过审核、确认语义、保存来源版本和 checksum 的参考样本。

它会成为测试的“标准答案输入”。

## 7. Manifest 是什么

Fixture 目录里有一个：

```text
source-manifest.json
```

它类似快递箱外面的装箱单，记录：

```text
这个 fixture 是什么类型
什么时候生成
来自哪个 Polar commit
使用哪个模型和 tokenizer revision
使用哪个 runtime image
包含哪些文件
每个文件多大
每个文件的 SHA256 是什么
做过哪些脱敏
哪些字段已知缺失
```

没有 manifest 时，我们只看到 `summary.json`，却不知道它从哪次运行来，也不知道是否被人改过。

## 8. Checksum 是什么

Checksum 是文件内容计算出来的指纹。

例如：

```text
summary.json 原始 bytes
    ↓ SHA256
64 位十六进制字符串
```

只要文件改一个字符，checksum 通常就会完全不同。

它用来回答：

> 现在读到的文件，是否还是打包时审核的那份精确内容？

它不能证明“谁写了文件”，所以不是数字签名。

## 9. Packager 是什么

Packager 是把 staging 目录变成正式 fixture 的工具。

输入：

```text
已经审核过的 staging 目录
+ Polar/model/runtime 等 metadata
```

输出：

```text
正式 fixture 目录
+ source-manifest.json
+ 每个文件的 bytes/SHA256
```

对应代码：

[`scripts/package_polar_fixture.py`](../../scripts/package_polar_fixture.py)

## 10. Fixture Verifier 是什么

Fixture verifier 检查打包好的 fixture 是否符合规则。

它检查的是：

- 文件是否齐全；
- manifest 是否完整；
- checksum 是否匹配；
- 路径是否安全；
- 是否有未登记文件；
- 是否有明显 secret；
- success/fault 的 evidence 是否自洽。

对应代码：

[`scripts/verify_polar_fixture.py`](../../scripts/verify_polar_fixture.py)

注意，它不是判断模型答案的那个 verifier。

## 11. Adapter 是什么

Adapter 是翻译器。

Polar 有自己的字段：

```text
prompt_ids
response_ids
response_logprobs
```

我们的核心模块使用公共字段：

```text
token_ids
prompt_token_count
old_logprobs
```

`PolarSourceAdapter` 负责将前者翻译成后者。

对应代码：

[`src/sources/polar.py`](../../src/sources/polar.py)

## 12. Canonical Record 是什么

Canonical 的意思是“统一、规范的表达”。

不管数据来自 Polar、JSONL 还是未来的自定义 Harness，进入核心 Processor 后都变成：

```python
RolloutRecord(...)
```

这样 FailureClassifier 不需要分别学习 Polar 字段、JSONL 字段和每个新 Producer 的字段。

对应代码：

[`src/contracts/rollout_record.py`](../../src/contracts/rollout_record.py)

## 13. Capability 是什么

Capability 可以理解为一条数据拥有的“能力标签”，但它不是人手写的标签，而是从实际字段计算。

例如：

```text
token_ids 不为 None
→ TOKEN_IDS capability

reward 不为 None
→ REWARD capability

policy_version 不为 None
→ POLICY_VERSION capability
```

它用来在运行 Trainer 前提前回答：

> 这批数据是否真的具有当前操作需要的字段？

对应代码：

[`src/contracts/capabilities.py`](../../src/contracts/capabilities.py)

---

# 第二课：Stage 1 到底做了什么

## 14. Stage 1 的一句话目标

> 把服务器已经跑出来的 Polar Calculator 结果，整理成可信、固定、可以给 Adapter 测试使用的真实样本。

Stage 1 不是在实现最终 Processor。它是在准备“真实教材”和“测试样本”。

## 15. Stage 1 的完整输入输出

### 输入

服务器上的一次真实 Polar 运行：

```text
task request
Polar session
Gateway completion
SGLang generation
trajectory
evaluator report
runtime error evidence
```

### 输出

两套 fixture：

```text
calculator_success/
calculator_fault/
```

以及一份语义审计：

[`notes/day-02-semantic-audit.md`](../../notes/day-02-semantic-audit.md)

## 16. 为什么需要两套 fixture

如果只有一条正常执行的数据，我们只能测试 Adapter 的 happy path。

我们还需要一条基础设施故障，用来测试：

```text
没有运行模型时
不能自动补 token
不能自动补 reward=0
不能让它看起来可以训练
```

所以：

```text
calculator_success
    测试有 trace、有 evaluator 的路径

calculator_fault
    测试 runtime 在模型运行前失败的路径
```

## 17. 第一步代码：规定允许打包哪些文件

具体代码位于：

```text
scripts/package_polar_fixture.py
```

其中：

```python
BASE_REQUIRED_FILES = {
    "request.json": "request",
    "response.json": "response",
    "summary.json": "summary",
}
```

这段代码的作用是规定 Calculator fixture 最少必须有三份证据。

为什么是这三份？

- `request.json`：证明提交了什么任务；
- `response.json`：证明上游终端返回了什么；
- `summary.json`：保存 trajectory、status、reward 等核心字段。

如果 `summary.json` 不存在，Packager 直接拒绝：

```python
if not path.is_file() or path.is_symlink():
    raise PackageError(f"missing required regular file: {filename}")
```

对应测试位于：

```text
tests/test_package_polar_fixture.py
```

测试名：

```text
test_rejects_missing_required_source_file
```

这个测试先删除 `summary.json`，然后确认打包失败。

## 18. 第二步代码：拒绝偷偷混进来的文件

Packager 先记录所有允许的路径，然后重新扫描 staging：

```python
unexpected = sorted(actual_files - allowed_paths)
if unexpected:
    raise PackageError(
        "unexpected staging files; explicitly classify or remove them: "
        + ", ".join(unexpected)
    )
```

这段代码起到的作用是：

> staging 中任何没有被我们明确分类的文件，都不能自动进入 fixture。

举例：如果 staging 多出：

```text
mystery.bin
```

Packager 不会想当然地复制它，而是失败并要求人检查。

对应测试：

```text
test_rejects_unclassified_staging_file
```

这一步主要防止把敏感文件或巨大无关日志提交到 Git。

## 19. 第三步代码：计算每个文件的指纹

具体函数：

```text
build_manifest()
```

核心代码：

```python
{
    "path": source_file.relative_path,
    "role": source_file.role,
    "media_type": source_file.media_type,
    "bytes": path.stat().st_size,
    "sha256": sha256_file(path),
}
```

这段代码为每个 fixture 文件生成一条装箱记录。

例如：

```json
{
  "path": "summary.json",
  "role": "summary",
  "media_type": "application/json",
  "bytes": 12345,
  "sha256": "..."
}
```

以后任何人修改 `summary.json`，`bytes` 或 `sha256` 就不再匹配。

## 20. 第四步代码：为什么先写临时目录

具体函数：

```text
package_fixture()
```

主要流程：

```python
with tempfile.TemporaryDirectory(...) as temp_dir:
    # 先复制到临时目录
    # 在临时目录生成 manifest
    # 运行 verify_fixture
    # 全部通过后再复制到正式 fixture
```

这段代码起到的作用是：

> 不让一个打包到一半、验证失败的 fixture 留在正式目录里。

如果我们直接向正式目录写：

```text
先写 request.json
再写 response.json
写 summary 时失败
```

目录里会留下半套 fixture，后面的人可能误以为它可用。

临时目录让整个过程更接近：

```text
先完整组装并验收
→ 再发布
```

## 21. 第五步代码：Verifier 检查 Calculator success

具体代码位于：

```text
scripts/verify_polar_fixture.py
```

函数：

```text
_validate_calculator_evidence()
```

它对 `calculator_success` 检查：

```python
if status != "completed":
    errors.append(...)
if not evaluation:
    errors.append(...)
if reward 不是数值:
    errors.append(...)
if evaluator timeout/error:
    errors.append(...)
```

请注意，它没有检查：

```python
reward == 1
```

因为这里的 success 是“执行链完成”，不一定是“模型答对”。

## 22. 真实结果为什么让我们修改理解

真实 `calculator_success` 中：

```text
status = COMPLETED
reward = 0.0
resolved = false
empty_generation = true
```

如果我们把 fixture 名字当作真相，就会误标为模型成功。

现在的测试专门固定这个行为：

```text
test_accepts_completed_calculator_path_with_valid_task_failure
```

测试输入明确写：

```python
"outcome_reward": 0.0,
"resolved": False,
```

然后要求 verifier 返回 PASS。

PASS 的意思是：

```text
这个 fixture 正确表达了一次执行完成的任务失败
```

不是：

```text
模型成功了
```

## 23. Fault fixture 的代码规则

同一个函数对 `calculator_fault` 检查：

```python
if status not in {"error", "timeout", "failed"}:
    errors.append(...)
if reward is not None:
    errors.append(
        "calculator_fault must not map infrastructure failure to reward"
    )
if not summary.get("error"):
    errors.append(...)
```

这段代码起到的作用是：

> 基础设施故障必须有明确错误证据，而且 reward 必须缺失。

为什么不是 reward=0？

因为 reward=0 表示 evaluator 正常判定模型失败；这里模型根本没开始。

## 24. 第一课小结：Stage 1 与最终模块有什么关系

Stage 1 自己不做训练数据分类，但它给后续模块提供两条关键教材：

```text
教材 A：有 token、有 reward、verifier 正常判错
→ 未来应成为 VALID_FAILURE

教材 B：runtime 失败、无 token、无 reward
→ 未来应成为 INVALID_INFRASTRUCTURE
```

如果没有这两条真实教材，我们写 FailureClassifier 时只能自己想象。

### 你现在应该能回答

1. Fixture 为什么不是普通 JSON 示例？
2. Packager 为什么不直接复制 raw 目录？
3. Manifest 为什么同时保存版本和 checksum？
4. `calculator_success` 为什么不是模型成功？
5. Infrastructure fault 为什么不能 reward=0？

如果这五点还不清楚，建议先不要继续读 Stage 2。

---

# 第三课：Stage 2 怎样把真实数据变成公共数据结构

## 25. Stage 2 的一句话目标

> 不管 rollout 来自 Polar 还是 JSONL，都把它翻译成同一种可信数据对象，而且缺失的训练字段必须保持缺失。

## 26. 为什么先写 `RolloutRecord`

最终的 FailureClassifier 不应该接收 Polar 私有 JSON：

```python
classifier.process(polar_summary)
```

因为换成其他 Producer 就要重写 classifier。

我们希望它接收：

```python
classifier.process(rollout_record)
```

所以先定义公共对象。

具体代码：

[`src/contracts/rollout_record.py`](../../src/contracts/rollout_record.py)

## 27. 先看一条最简单的完整记录

测试中的示例位于：

```text
tests/contract_fixtures.py
```

简化后：

```python
RolloutRecord(
    trajectory_id="trajectory-001",
    task_id="task-001",
    group_id="group-001",
    policy_version="policy-v0",
    source_type="fixture",
    source_record_id="fixture-record-001",
    token_ids=(101, 102, 103, 104),
    prompt_token_count=1,
    loss_mask=(0, 1, 1, 1),
    old_logprobs=(-0.3, -0.2, -0.1),
    reward=1.0,
    verifier_status=VerifierStatus.PASSED,
)
```

现在逐组解释。

## 28. Identity 部分在回答什么

```text
trajectory_id      这条轨迹是谁
task_id            它在做哪个任务
group_id           它属于哪个 GRPO 采样组
policy_version     它由哪个模型版本采样
source_type        当前直接从哪里读到
source_record_id   在当前来源中的编号
```

其中 `group_id`、`policy_version` 可以在普通 record 中缺失，因为 fault 或早期 fixture 可能根本没保存。

但进入训练 batch 前必须存在。

## 29. Training payload 在回答什么

### `token_ids`

模型采样时真实使用的 token 数字。

不能从文本重新 tokenize，因为 tokenizer revision、chat template 和 special token 可能不同。

### `prompt_token_count`

告诉我们完整 token 中前多少个属于 prompt。

例如：

```text
token_ids = [10, 11, 20, 21, 22]
prompt_token_count = 2

prompt   = [10, 11]
response = [20, 21, 22]
```

### `loss_mask`

告诉 Trainer 哪些 token 参与 loss：

```text
loss_mask = [0, 0, 1, 0, 1]
```

前两个 prompt token 不训练；response 中也可能有 mask=0 的 token。

### `old_logprobs`

采样 policy 当时给这些 response token 的 log probability。

缺失时必须是 `None`，不能填 0。

因为：

```text
logprob = 0
```

数学含义是 probability=1，不是“未知”。

### `reward`

Evaluator 的数值结果。

```text
0.0   正常判错
None  没有可信结果
```

## 30. `RolloutRecord` 为什么有大量检查代码

`RolloutRecord.__post_init__()` 会检查字段之间是否自洽。

### Mask 检查

```python
if action_mask is not None and loss_mask is not None:
    raise ContractValidationError("provide action_mask or loss_mask, not both")
```

作用：防止两套 mask 同时出现、下游不知道相信哪套。

```python
if token_ids is not None and mask is not None and len(token_ids) != len(mask):
    raise ContractValidationError(
        "token_ids and mask must have the same length"
    )
```

作用：防止第 i 个 mask 实际对应错 token。

### Prompt boundary 检查

```python
if not 0 <= prompt_token_count <= len(token_ids):
    raise ContractValidationError(...)
```

作用：防止 response 边界跑到 token 序列外。

### Logprob 长度检查

```python
valid_lengths = {len(token_ids)}
if mask is not None:
    valid_lengths.add(sum(mask))
if prompt_token_count is not None:
    valid_lengths.add(len(token_ids) - prompt_token_count)
```

它允许 logprob 对齐：

```text
全部 token
response token
trainable token
```

但不允许任意长度。

## 31. 为什么结构合法不等于业务有效

下面这条记录在类型上可以合法：

```python
RolloutRecord(
    rollout_status=FAILED,
    reward=0.0,
    ...
)
```

`RolloutRecord` 无法只凭这两个字段知道 reward 是否由 evaluator 真实给出。

所以：

```text
RolloutRecord
    负责结构自洽

FailureClassifier（Day 5）
    负责联合 status/verifier/evidence 判断业务有效性
```

不要把 contract validation 和业务 classification 混在一起。

## 32. 为什么数据对象要不可变

具体代码：

[`src/contracts/_json.py`](../../src/contracts/_json.py)

`RolloutRecord` 本身是 frozen dataclass，但内部 metadata 如果仍是 dict，仍可能被修改。

所以 `freeze_json()` 把：

```text
dict → MappingProxyType
list → tuple
```

作用是保证：

```text
record 创建后
→ 内容不会在某个 Processor 中途被偷偷改变
→ checksum 稳定
→ 实验可审计
```

对应测试：

```text
test_valid_record_is_immutable_and_exposes_capabilities
```

## 33. Capability 是怎样计算出来的

具体代码：

```text
src/contracts/capabilities.py
```

例如：

```python
if record.reward is not None:
    available.add(Capability.REWARD)
```

这里必须使用：

```python
is not None
```

不能写：

```python
if record.reward:
```

因为 reward=0 是合法 reward，但在 Python 中是 falsy。

同理：

```python
if record.policy_version is not None:
    available.add(Capability.POLICY_VERSION)
```

Capability 不是猜测，也不是 Adapter 自报，而是由真实字段派生。

## 34. 为什么一个 batch 的 capability 要取交集

假设两条记录：

```text
A：TOKEN_IDS + REWARD + POLICY_VERSION
B：TOKEN_IDS + REWARD
```

如果 batch 声明并集：

```text
TOKEN_IDS + REWARD + POLICY_VERSION
```

Trainer 会误以为每条都有 policy version。

所以实现：

```python
common.intersection_update(capabilities_for_record(record))
```

结果是：

```text
TOKEN_IDS + REWARD
```

对应测试：

```text
test_rollout_batch_uses_capability_intersection
```

## 35. 三个数据对象为什么不能合成一个

### `RolloutBatch`

具体代码：

[`src/contracts/rollout_batch.py`](../../src/contracts/rollout_batch.py)

它表示刚从 Adapter 得到的一批 canonical records。

这批数据可以：

- 混多个 task；
- 缺 policy；
- 包含 fault；
- 还没分类。

### `TrainingReadyBatch`

具体代码：

[`src/contracts/training_batch.py`](../../src/contracts/training_batch.py)

它要求每条记录的：

```text
task_id
group_id
policy_version
```

全部和 batch 声明一致，并满足训练 capability。

### `ResampleRequest`

具体代码：

[`src/contracts/resample_request.py`](../../src/contracts/resample_request.py)

当 group 需要 4 条，过滤后只有 3 条，它表达：

```text
这个 task/group/policy 还缺 1 条
```

它不亲自调用 Polar。

## 36. AdapterResult 为什么不直接返回 batch

具体代码：

[`src/sources/base.py`](../../src/sources/base.py)

假设 JSONL：

```text
第 1 行合法
第 2 行损坏
第 3 行合法
```

如果遇到第 2 行就抛异常，我们看不到第 1、3 行已经能解析。

如果静默跳过第 2 行，又可能拿残缺数据训练。

所以 `AdapterResult` 同时返回：

```text
records
warnings
errors
capabilities
```

但只要有 error：

```python
def to_batch(...):
    if self.errors:
        raise AdapterConversionError(...)
```

这叫：

```text
诊断时允许看见部分成功
训练时拒绝静默使用残缺输入
```

## 37. JSONL Adapter 是做什么的

具体代码：

[`src/sources/jsonl.py`](../../src/sources/jsonl.py)

它让我们不用 Polar，也能保存和读取 canonical records。

例如：

```json
{
  "schema_version": "rollout-jsonl/v1",
  "record": {
    "schema_version": "rollout-record/v1",
    "trajectory_id": "..."
  }
}
```

它的作用包括：

- 无 GPU 单元测试；
- 离线 demo；
- 其他 Producer 的最低成本接入；
- Polar record 的公共格式导出。

### 为什么读回来后 source 会改变

原来直接来源：

```text
Polar summary.json
```

导出再读入后，当前直接来源变成：

```text
rollouts.jsonl 第 N 行
```

所以代码调用：

```python
record.with_source_envelope(
    source_type="jsonl",
    source_record_id=f"line:{line_number}",
    ...
)
```

轨迹语义不变，运输来源改变。

## 38. Polar Adapter 才是真正读取 Stage 1 fixture 的代码

具体代码：

[`src/sources/polar.py`](../../src/sources/polar.py)

它的输入不是在线 Polar 服务，而是：

```text
tests/fixtures/polar/.../
```

### 第一步：确认 summary 没被改

```python
summary_sha = hashlib.sha256(summary_bytes).hexdigest()
if summary_entry.get("sha256") != summary_sha:
    raise ContractValidationError(...)
```

作用：Adapter 只转换经过 manifest 审核的精确 summary。

### 第二步：读取 Polar 原始数组

```python
prompt_ids = trace["prompt_ids"]
response_ids = trace["response_ids"]
response_mask = trace["loss_mask"]
response_logprobs = trace["response_logprobs"]
```

### 第三步：转成完整序列

```python
token_ids = tuple(prompt_ids + response_ids)
loss_mask = tuple([0] * len(prompt_ids) + response_mask)
```

举一个小例子：

```text
prompt_ids        = [10, 11]
response_ids      = [20, 21, 22]
response loss mask= [1, 0, 1]
```

转换后：

```text
token_ids         = [10, 11, 20, 21, 22]
prompt_token_count= 2
full loss mask    = [0, 0, 1, 0, 1]
```

为什么 prompt 补 0？因为 prompt token 不参与 response loss。

### 第四步：构造 canonical record

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

这就是 Stage 1 真实数据进入 Stage 2 公共 contract 的交界点。

## 39. 真实 Day 2 数据经过 Adapter 后是什么

测试断言的真实结果：

```text
prompt token 数       14846
response token 数        99
完整 token 数         14945
完整 mask 数          14945
old logprob 数           99
reward                  0.0
```

因此它拥有：

```text
TOKEN_IDS
ACTION_MASK
OLD_LOGPROBS
REWARD
VERIFIER_EVIDENCE
```

但没有：

```text
GROUP_ID
POLICY_VERSION
```

所以它能成为合法 `RolloutRecord`，却不能成为默认 `TrainingReadyBatch`。

这正是 Capability 系统的价值：

```text
不因为缺字段而丢失 evidence
也不因为能表示就误以为能训练
```

## 40. Fault fixture 经过 Adapter 后是什么

没有 traces 时，Polar Adapter 调用：

```text
_empty_failure_record()
```

它构造：

```text
rollout_status = FAILED
runtime_status = FAILED
harness_status = NOT_RUN
verifier_status = NOT_RUN
token_ids = None
reward = None
```

测试确认：

```python
self.assertIsNone(record.reward)
self.assertFalse(result.capabilities)
```

这里没有任何“补齐字段”的逻辑。

## 41. 第二课小结：Stage 2 到底实现了什么

Stage 2 不是实现了完整 pipeline，而是实现了可信的数据边界：

```text
Producer 私有 artifact
→ Source Adapter
→ 统一 RolloutRecord
→ Capability 检查
→ RolloutBatch / TrainingReadyBatch / ResampleRequest
```

还没有实现：

```text
FailureClassifier
SignalFilter
GroupBuilder
Trainer Adapter
```

### 你现在应该能回答

1. RolloutRecord 为什么允许字段为 None？
2. 为什么 None 不能统一换成 0 或空数组？
3. Capability 为什么由字段派生？
4. RolloutBatch 和 TrainingReadyBatch 有什么区别？
5. Polar Adapter 为什么不重新 tokenize？
6. Day 2 record 为什么合法但不能训练？

---

# 第四课：Stage 3 为什么又回去准备 Coding rollout

## 42. Stage 3 的一句话目标

> 准备一套服务器可以直接执行的最小 Coding 实验，用真实模型取得 success、valid failure 和 infrastructure failure 三类更完整样本。

## 43. 为什么 Calculator fixture 还不够

目前只有：

```text
一条有效任务失败
一条 runtime failure
```

我们还没有真实看到：

- 模型成功解决 Coding task；
- 多轮工具调用；
- 文件修改；
- patch；
- clean replay；
- verifier timeout；
- Coding 的 task/group/policy 字段来源。

如果直接写 FailureClassifier 的全部规则，会对这些情况做猜测。

所以 Stage 3 是在补充下一批“真实教材”。

## 44. 第一步：为什么要先筛选候选任务

SWE-bench Verified 有很多任务。每个任务的：

- repository 不同；
- runtime image 不同；
- 测试数量不同；
- prompt 长度不同；
- 执行时间不同。

我们不需要跑完整 benchmark，只需要少量适合生成 fixture 的任务。

对应代码：

[`scripts/select_swebench_candidates.py`](../../scripts/select_swebench_candidates.py)

## 45. Candidate selector 具体在做什么

### 先排除不可复现的任务

```python
if base_commit 不是完整 40 位 SHA:
    continue
if FAIL_TO_PASS 为空:
    continue
if problem_statement 为空:
    continue
```

为什么必须有完整 base commit？

因为我们需要在相同代码起点 replay patch。

为什么必须有 FAIL_TO_PASS？

因为至少要有测试能证明 base 版本存在 bug，修复后才能判断任务是否解决。

### 然后计算低成本排序

```python
test_count = len(fail_to_pass) + len(pass_to_pass)
problem_chars = len(problem)
score = test_count * 1_000_000 + min(problem_chars, 999_999)
```

这段代码的意思是：

```text
优先测试更少的任务
测试数相同时优先 prompt 更短的任务
```

为什么测试数量乘一百万？为了让少一个测试的优势一定大于 prompt 长度差异。

这不是在预测哪个任务模型最容易答对，只是在优先挑运行便宜、容易诊断的任务。

## 46. 第二步：Capture Policy 是什么

对应文件：

[`configs/polar/coding/capture-policy.yaml`](../../configs/polar/coding/capture-policy.yaml)

它不是模型超参数，而是证据保存规则。

它规定三类结果。

### `coding_success`

```text
verifier 正常完成
resolved = true
reward = 1
```

### `coding_valid_failure`

```text
verifier 正常完成
resolved = false
reward = 0
```

### `coding_invalid_infra`

```text
verifier timeout/crash 或 runtime error
resolved = null
reward = null
```

Capture policy 的作用是：

> 在服务器开始跑之前，先规定什么证据才有资格被称为 success、valid failure 或 infrastructure failure。

否则跑完后可能为了迎合结果临时改标准。

## 47. Patch 为什么必须单独保存

Coding Agent 最终说：

```text
“我已经修好了。”
```

这段文本不是证据。

真正需要的是：

```text
patch.diff
```

它记录到底改了哪些文件、哪些行。

同时保存：

```text
patch_sha256
```

保证 verifier evidence 引用的 patch 与 fixture 中的 patch 是同一份内容。

## 48. Clean replay 是什么

假设 Agent 在运行过程中：

- 创建了临时文件；
- 安装了额外包；
- 手工改了多个未进入 patch 的文件；
- 某次测试留下缓存。

原 workspace 的 verifier 结果可能依赖这些副作用。

Clean replay 的过程是：

```text
创建干净 runtime
→ 回到指定 base commit
→ 只应用保存的 patch.diff
→ 再运行 verifier
```

如果结果仍一致，说明 outcome 可以由保存的 patch 重现。

Replay 记录在：

```text
replay.json
```

成功与有效失败都要求 replay。

为什么失败也要 replay？

因为 reward=0 也可能是 evaluator 环境偶发错误。只有 verifier 正常完成、并且在干净环境中仍失败，才是可信的 valid failure。

## 49. Coding verifier 代码如何判断三类结果

具体函数：

```text
scripts/verify_polar_fixture.py
_validate_coding_evidence()
```

### Success 分支

```python
if not verifier_completed or timed_out or crashed:
    error
if resolved is not True or reward != 1:
    error
```

### Valid failure 分支

```python
if not verifier_completed or timed_out or crashed:
    error
if resolved is not False or reward != 0:
    error
```

### Infrastructure invalid 分支

```python
has_infra_signal = timed_out or crashed or error_type
if not has_infra_signal:
    error
if resolved is not None or reward is not None:
    error
```

这段代码不是只看 reward，而是先看 verifier 是否正常完成。

最简单的判断表：

| Verifier 是否正常 | resolved | reward | 类别 |
|---|---:|---:|---|
| 是 | true | 1 | `VALID_SUCCESS` |
| 是 | false | 0 | `VALID_FAILURE` |
| 否 | null | null | `INVALID_INFRASTRUCTURE` |

## 50. 为什么故意设计两种 infrastructure fault

### Runtime prepare failure

发生在 Agent 运行前：

```text
runtime.prepare 增加 exit 42
```

它验证：没有 completion、没有 verifier outcome 时如何处理。

### Verifier timeout

发生在 Agent rollout 后：

```text
把 evaluator timeout 调成很小的正数
```

它验证：即使已经有 token 和 patch，只要 verifier 没有正常完成，task outcome 仍然无效。

为什么要两种？

因为 FailureClassifier 不能只学会识别“没有 token”的 fault；verifier timeout 可能已经有完整 trajectory。

## 51. Runbook 是什么，为什么也算实现

对应文件：

[`docs/runbooks/day-03-polar-coding.md`](../runbooks/day-03-polar-coding.md)

Runbook 是服务器执行说明，但不只是命令清单。它冻结了：

```text
使用哪个 Polar commit
使用哪个模型 revision
如何选择 3 个候选
如何验证 runtime baseline
启动哪些服务
使用哪些端口
每个任务采多少次
何时允许补采
如何注入两种 fault
保存哪些文件
如何打包和验证
如何只停止本次运行的 PID
什么条件才算完成
```

为什么这些要写得精确？

因为服务器端 AI 如果自行发挥，可能：

- 换成更新模型；
- 并发跑任务导致 OOM；
- 把 reward 0 当 success；
- 杀掉其他用户进程；
- 没保存 raw evidence 就退出。

Runbook 把一次昂贵实验变成可重复执行流程。

## 52. Stage 3 当前到底完成了哪一部分

已经完成的代码：

```text
候选任务选择器
候选选择测试
Coding capture policy
Coding fixture required file/role
Coding success/failure/infra verifier
patch SHA 检查
clean replay 检查
敏感信息扫描
服务器 runbook
字段审计模板
pilot report 模板
```

尚未完成的外部执行：

```text
在服务器构建 3 个 runtime
运行真实 qwen_code rollout
取得真实 Coding success/failure
实际注入两种 fault
实际 clean replay
生成三套真实 Coding fixture
填写真实 field map/pilot report
```

所以不能说：

```text
“Coding Agent 已经跑通。”
```

应该说：

```text
“运行 Coding Agent 并收集可信 fixture 所需的本地实现已经完成，等待服务器执行。”
```

## 53. 第三课小结：Stage 3 与最终模块有什么关系

Stage 3 将为 Day 5 提供：

```text
真实 VALID_SUCCESS 教材
真实 VALID_FAILURE 教材
有 trajectory 的 verifier timeout 教材
无 trajectory 的 runtime failure 教材
```

FailureClassifier 之后就可以根据真实 failure plane 写规则，而不是只适配 Calculator。

### 你现在应该能回答

1. 为什么不直接随便选一个 SWE-bench 任务？
2. Capture policy 与模型配置有什么区别？
3. Patch 文本为什么还需要 SHA256？
4. 为什么 success 和 failure 都要 clean replay？
5. 为什么需要两种 infrastructure fault？
6. Stage 3 的实现完成和实验完成有什么区别？

---

# 第五课：把三个 Stage 连回最终 Data Infra

## 54. Stage 1 不是无关的数据整理

它解决：

```text
我们要用什么真实输入测试 Adapter？
```

没有它，Stage 2 只能用自己编造的 Polar JSON。

## 55. Stage 2 是目前完成的核心地基

它解决：

```text
如何忠实表示不同 Producer 的 rollout？
如何保证字段内部一致？
如何明确哪些训练能力真实存在？
```

没有它，Day 5 Processor 会直接耦合 Polar。

## 56. Stage 3 不是回头重复 Polar

它解决：

```text
Calculator 没覆盖的 Coding success、多轮 tool、patch 和 verifier fault 从哪里来？
```

没有它，Day 5 的 success 和复杂 failure 规则没有真实 golden test。

## 57. 下一步 Day 5 如何使用这些结果

### FailureClassifier

读取 canonical 字段：

```text
rollout_status
runtime_status
harness_status
model_backend_status
verifier_status
reward
evidence
```

用 Stage 1/3 fixture 验证：

```text
正常判对 → VALID_SUCCESS
正常判错 → VALID_FAILURE
runtime/verifier 故障 → INVALID_INFRASTRUCTURE
```

### SignalFilter

分类之后，对同组 valid records 检查：

```text
reward 是否有差异
是否有足够 trainable token
```

### GroupBuilder

最后按：

```text
task_id + group_id + policy_version
```

构建固定大小 batch；缺样本就产生 `ResampleRequest`。

## 58. 现在最推荐的真实阅读顺序

不要直接读 2771 行深度参考。按下面顺序学习。

### 第一次：只读本文 Stage 1

打开：

1. 本文第 1–24 节；
2. `scripts/package_polar_fixture.py` 中的常量和 `discover_source_files()`；
3. `scripts/verify_polar_fixture.py` 中的 `_validate_calculator_evidence()`；
4. `notes/day-02-semantic-audit.md`。

目标：能讲清 raw → staging → fixture → verifier。

### 第二次：只读本文 Stage 2

打开：

1. 本文第 25–41 节；
2. `tests/contract_fixtures.py`；
3. `src/contracts/rollout_record.py` 的字段和 `__post_init__()`；
4. `src/contracts/capabilities.py`；
5. `src/sources/polar.py` 的 `_trace_record()`。

目标：手工完成小数组的 Polar → RolloutRecord 映射。

### 第三次：只读本文 Stage 3

打开：

1. 本文第 42–53 节；
2. `scripts/select_swebench_candidates.py`；
3. `configs/polar/coding/capture-policy.yaml`；
4. `scripts/verify_polar_fixture.py` 的 `_validate_coding_evidence()`；
5. Day 3 runbook 的目标、outcome 和 staging 部分。

目标：能解释为什么 success/failure/infra 各自需要什么证据。

## 59. 最终请用这段话检查自己是否真正理解

> 我们最终要写的是 rollout 到训练 batch 之间的数据模块，但这个模块必须建立在真实上游语义上。Stage 1 先把 Polar Calculator 的真实服务器结果整理成带 manifest 和 checksum 的固定 fixture，并发现 execution completed 不等于 task success、infrastructure failure 不能写 reward 0。Stage 2 再根据这些事实实现统一 RolloutRecord、capability、batch contract、JSONL Adapter 和 Polar Adapter，让缺失字段保持缺失，同时阻止缺 group/policy 的数据进入训练。Stage 3 则准备采集 Calculator 没覆盖的真实 Coding success、valid failure、patch、clean replay 和 verifier/runtime fault，为下一步 FailureClassifier 提供更完整的 golden examples。

如果你能不用术语堆砌、用自己的话复述上面这段逻辑，那么这些“前置工作”就已经真正读懂了。
