# V1 Agent Infra 教学与复盘手册

> 这不是一份只告诉你“项目里有哪些文件”的目录，也不是背面试答案的提纲。
> 它是一份从零建立 Agent Infra 思维、理解 V1 为什么这样设计、能够亲手验证实现，并能在数周后重新捡起项目的教学文档。

---

## 0. 先知道你正在学习什么

### 0.1 项目的一句话定义

我们正在实现一个 **Multi-Harness Agent Execution Data Plane**：它接收不同 Agent Harness 执行任务时产生的事件，把事件可靠地记录、组装成可审计的 Episode，再计算指标、定位失败原因，并为 Harness 对比、数据筛选和后续 Agentic RL 提供可信数据。

先记住一个朴素版本：

> Agent 在“做题”时会发生很多事情；我们的系统负责把这些事情如实记下来、整理清楚、判断数据是否可信，并帮助人回答“它为什么成功或失败”。

### 0.2 V1 已经解决了什么

V1 建立的是一条最小但完整的数据链路：

```text
Harness / Model / Tool / Sandbox / Verifier
                    │
                    ▼
              TraceEvent 事件流
                    │
                    ▼
           append-only JSONL 原始日志
                    │
                    ▼
            EpisodeAssembler 组装
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
     AgentEpisode  Metrics  Diagnosis
          │
          ▼
 Harness 比较 / 数据筛选 / Agentic RL 上游数据
```

V1 的重点不是训练一个更强的模型，而是建立“数据事实层”。如果事实层不可信，那么后面的成功率、失败归因、训练样本筛选都会建立在错误数据上。

### 0.3 学完本文后，你应该能够做到

1. 用自己的话解释 Agent、Harness、Trace、Episode、Artifact 和 Verifier。
2. 解释为什么不能只保存最终答案或一段普通日志。
3. 沿着代码说清楚一次执行怎样从事件变成 Episode。
4. 区分任务失败、基础设施错误和数据不完整。
5. 解释为什么 `0`、`None`、`UNKNOWN` 和 `NOT_OBSERVABLE` 不能混为一谈。
6. 运行 V1 演示并读懂三个样例 Episode。
7. 解释 V1 和 Stage 1–3、Harness 改进以及 Agentic RL 的关系。
8. 面对面试官追问时，能说明一致性、幂等、血缘、能力边界和因果比较。

### 0.4 推荐使用方式

- 第一轮只读第 1–4 章，建立整体直觉。
- 第二轮读第 5–12 章，同时打开对应代码。
- 第三轮完成第 13 章实验，再用第 18–21 章复盘和自测。

如果某个术语暂时没懂，不要停下来背定义。先顺着“一个失败任务”的故事走完，后面再回来补概念。

---

## 1. 从一次 Agent 执行开始理解系统

### 1.1 一个贯穿全文的例子

假设任务是：

> 请读取仓库中的 `config.json`，找出 `timeout` 配置并告诉我它的值。

Agent 的一次执行可能如下：

1. 用户任务进入 Harness。
2. Harness 把任务和工具说明发给模型。
3. 模型决定调用文件读取工具，参数是 `config.json`。
4. 工具返回“文件不存在”。
5. Harness 把错误反馈给模型。
6. 模型没有调整策略，又调用完全相同的工具和参数。
7. 同样的错误再次发生，形成循环。
8. Harness 达到最大步数后终止。
9. Verifier 检查答案，判定任务失败。

如果只保存最终结果，数据库里可能只有：

```json
{"task_id": "task-42", "success": false}
```

它不能回答：

- 模型有没有请求正确的工具？
- 工具错误有没有反馈给模型？
- Harness 是否修改、截断或遗漏了错误信息？
- 模型是否看到了错误却重复同一动作？
- 是任务本身失败，还是 Sandbox 超时导致执行无效？
- 日志是否丢了一半，所以我们误以为模型没有恢复？

Agent Infra 的价值就出现在这些问题里：它不只保存结果，还保存能够解释结果的执行证据。

### 1.2 先区分五个参与者

#### 模型（Model）

模型接收上下文并产生下一步输出。输出可能是自然语言，也可能是结构化工具调用。模型本身通常不知道工具是否真的执行成功，只能依据 Harness 下一轮反馈继续决策。

本项目用 `MODEL_REQUEST` 与 `MODEL_RESPONSE` 事件表示模型调用。

#### Agent

Agent 不是单指模型。更准确地说，Agent 是“模型 + 上下文 + 工具 + 执行循环 + 终止条件”共同构成的任务执行体。

同一个模型放在两个不同 Harness 中，表现可能明显不同，因为工具描述、错误反馈、重试策略、上下文裁剪和终止逻辑都可能不同。

#### Harness

Harness 是驱动 Agent 执行的编排层，通常负责：

- 组织 prompt 和历史上下文；
- 把模型输出解析成工具调用；
- 执行工具并回传结果；
- 决定是否重试、何时停止；
- 管理 token、轮数和预算；
- 接入 Sandbox、Verifier 和观测钩子。

因此，Harness 不是无关紧要的“外壳”。我们比较 Harness，是为了研究同一模型在不同执行编排下为什么表现不同。

本项目通过 `HARNESS_DECISION`、`RETRY_SCHEDULED` 和 `TERMINATION_DECIDED` 等事件观测内部决策。

#### Tool 与 Sandbox

Tool 是 Agent 可以调用的能力，例如读取文件、运行命令、搜索代码或提交补丁。Sandbox 是工具运行时所在的受控环境，负责文件、进程、网络、资源和超时限制。

工具调用失败不一定是模型能力差：

- 模型给了错误参数，属于任务执行问题；
- 工具服务返回 500，可能是工具基础设施问题；
- Sandbox 不允许读取目标文件，可能是环境配置问题；
- Sandbox 自身崩溃，可能让整次执行无效。

V1 用 `TOOL_CALL` / `TOOL_RESULT`、`SANDBOX_STARTED` / `SANDBOX_FINISHED` 等事件保存证据。

#### Verifier / Evaluator

Verifier 负责判断任务是否完成，而不是判断过程是否漂亮。代码任务可能通过测试验证，问答任务可能通过规则、模型裁判或人工标注验证。

Verifier 也会变化或出错，所以我们不仅保存 `passed=true/false`，还保存评测器版本和状态。否则评测规则改变后，两批成功率不能直接比较。

### 1.3 Agent Infra 到底是什么

Agent Infra 是支撑 Agent 稳定运行、观测、调试、评测和迭代的基础设施，通常包含：

- 执行编排：会话、工具、重试、终止、预算；
- 运行环境：Sandbox、资源隔离、命令执行；
- Trace/Data：事件采集、存储、组装、版本和血缘；
- Observability：指标、诊断、查询、告警和可视化；
- Evaluation：任务、Verifier、实验对比；
- Training data：轨迹筛选、转换、回放和训练批次。

当前项目专注的是其中的 **Trace / Data / Observability 数据平面**。执行系统规模化后，难点往往不再是“能不能跑一次”，而是“数据能否解释、能否比较、出故障后能否恢复”。

---

## 2. Control Plane、Data Plane 与 Observability

### 2.1 Control Plane 与 Data Plane

可以把系统想象成机场：

- Control Plane 像塔台，决定航班怎样调度、配置什么规则、使用哪个版本。
- Data Plane 像实际跑道和飞行记录，承载一次次真实执行以及执行产生的数据。

Agent 系统的 Control Plane 可能管理 Harness、模型、任务、并发、预算和调度。V1 的 Data Plane 负责接收事件、保存事实、组装 Episode、检查完整性、计算指标和诊断。

V1 没有一次性完成调度平台、在线队列、分布式存储和训练集群，而是先冻结最关键的数据语义。

### 2.2 Logging、Tracing 与 Observability 的区别

| 概念 | 核心问题 | 例子 |
| --- | --- | --- |
| Logging | 某处打印了什么？ | `tool failed: file not found` |
| Tracing | 一次请求经过哪些步骤，如何关联？ | 模型请求 → 工具调用 → 工具结果 → 重试 |
| Metrics | 系统整体表现怎样？ | 成功率、P95 延迟、工具错误率 |
| Observability | 能否从外部证据理解内部状态和故障？ | 为什么 Harness B 的成功率下降？ |

普通日志是一串文本。结构化 Trace 有稳定字段、身份关联、顺序、类型和版本，所以机器能够聚合、校验和重放。

Observability 也不等于“多打日志”。它要求系统能够回答可操作的问题，并明确哪些问题当前没有足够证据。

### 2.3 Telemetry 是原材料，不是结论

“出现了两次相同的 `TOOL_CALL`”是事实；“Harness 丢失了工具错误反馈”是归因结论。两者必须分开，否则系统会把猜测伪装成事实。

```text
TraceEvent（事实）
  → AgentEpisode（结构化事实集合）
  → EpisodeMetrics（确定性计算结果）
  → Diagnosis（带证据和置信度的解释）
```

---

## 3. Event、Span、Trace、Episode、Run 与 Artifact

### 3.1 Event：最小事实单元

Event 表示“某个时间点发生了一件事”，例如模型请求发出、工具返回错误、Verifier 判定失败。V1 的 Event 对应 `TraceEvent`，它至少需要回答：

- 谁发生了什么：`component`、`event_type`；
- 属于哪次执行：`run_id`、`episode_id`、`trace_id`；
- 在什么顺序：`sequence`、`timestamp`；
- 结果怎样：`status`；
- 额外事实：`attributes`、`artifact_refs`。

### 3.2 Span：有开始和结束的一段操作

Span 表示一段有持续时间的工作，例如一次模型调用或工具调用。它通常由开始事件与结束事件组成，并用 `span_id` 关联。

`parent_span_id` 表达父子或调用关系。如果只有时间戳而没有 Span 关系，并发发生时就很难知道哪个结果属于哪个请求。

### 3.3 Trace：一次端到端调用链

Trace 把跨组件 Span 连成调用链。`trace_id` 让我们知道模型、Harness、工具和 Sandbox 的事件属于同一个端到端过程。

Episode 在 V1 中通常对应一条主要 Trace，但概念上不要强行等同；未来一个 Episode 可能包含子 Trace。

### 3.4 Episode：一次完整任务尝试

Episode 是分析和训练常用的单位：某个 Agent 在特定 Harness、模型、环境和评测器配置下，对一个任务进行的一次尝试。

它不只是 Event 数组，还包含 Manifest、Outcome、Termination、IntegrityReport、Capture Capability、Artifact 和 Lineage。

### 3.5 Run：一批相关执行

Run 通常表示同一次实验、批处理或采集作业。一个 Run 可以包含很多 Episode。例如，用 Harness A 和固定模型跑 1,000 个任务是一个 Run，每个任务尝试是一个 Episode。

### 3.6 Artifact：不适合直接塞进事件的大对象

命令完整输出、模型原始响应、补丁和测试报告可能很大。全部内嵌 Event 会导致日志膨胀、重复存储和治理困难。因此 Event 保存 `ArtifactRef`，真实内容进入 Artifact Store；引用包含 URI、内容摘要、大小和媒体类型。

### 3.7 ID 的职责表

| 字段 | 回答的问题 | 不能代替什么 |
| --- | --- | --- |
| `event_id` | 这是哪个唯一事件？ | 业务顺序 |
| `run_id` | 属于哪次批量运行？ | Episode 唯一身份 |
| `episode_id` | 属于哪次任务尝试？ | 全局事件身份 |
| `trace_id` | 属于哪条调用链？ | 任务结果 |
| `span_id` | 属于哪段操作？ | 父子关系 |
| `parent_span_id` | 谁直接触发了它？ | 逻辑顺序 |
| `sequence` | Episode 内的逻辑顺序？ | 墙上时间 |

### 3.8 为什么既要 sequence 又要 timestamp

分布式系统的机器时钟可能漂移，网络也会让后产生的事件先到达。只按 `timestamp` 排序可能错误重建过程。

`sequence` 是生产者承诺的逻辑顺序；`timestamp` 是观测时间，用于延迟和审计。V1 组装时使用稳定排序规则，并检查 sequence 缺口与冲突。

---

## 4. 为什么先保存 append-only 原始事件

### 4.1 不直接“修改最终 Episode”

每收到一个事件就更新数据库中的 Episode 行看似简单，但会带来：

- 写到一半崩溃，状态无法解释；
- 乱序事件覆盖新状态；
- 重试造成重复写入；
- 组装规则升级后无法从事实重算；
- 调试时不知道当初到底收到了什么。

V1 先写 append-only JSONL：已落盘的原始事件不在原位置修改，只在末尾追加，再由可重放的 Assembler 生成派生视图。这和数据库 Write-Ahead Log 的思路相似：先可靠记录事实，再构建状态。

### 4.2 append-only 带来的能力

- 可审计：查看原始到达内容；
- 可重放：Assembler 修复后重新生成 Episode；
- 易恢复：部分写入不会覆盖过去数据；
- 易做血缘：派生结果可指向原事件摘要；
- 易解耦：生产与消费可分开演进。

它不是无限扩展的最终存储方案。V2 仍可能使用对象存储、列式文件、消息队列或数据库。

### 4.3 至少一次投递与幂等

生产者发送事件后若没及时收到确认，可能重试发送，这就是 at-least-once delivery 的典型现象。消费端要具备幂等性：处理一次和多次，最终语义相同。

V1 用 `event_id` 和 checksum 区分：

1. **精确重复**：ID 相同、摘要相同，安全忽略重复副本。
2. **冲突重复**：ID 相同、内容不同，不能任选一个，必须隔离并标记损坏。

如果静默接受第二种情况，就等于允许同一个事实有两个版本。

### 4.4 为什么要 quarantine

Quarantine 是隔离区。坏数据不应直接丢弃、假装正常进入指标，也不应阻塞整个批次。V1 将无法归属、身份冲突等事件隔离并保留原因，实现“好数据继续处理，坏数据可以追查”。

### 4.5 `flush`、`fsync` 与持久性边界

写文件不等于数据已经安全到达磁盘。`flush` 把用户态缓冲推进操作系统，`fsync` 请求同步到存储。V1 的 `EventWriter` 在 durable 模式下执行这些操作。

但这不等于分布式 exactly-once，也不保证所有硬件和电源故障下绝对不丢数据。V1 是单进程本地持久化基线。

### 4.6 坏行为什么不能拖垮整个文件

JSONL 一行一个事件。如果中间某行损坏，Reader 记录 `JsonlIssue`，同时尽量保留其他合法行。这叫 failure isolation：局部坏数据不应自动变成全局不可用。

### 4.7 敏感信息为何必须在落盘前处理

API key、token、cookie 或密码一旦进入原始事件，即使展示时遮盖，也已存在于磁盘和备份。因此 `redact_secrets` 在写入前按敏感键名遮盖。生产系统还需内容分类、加密、访问控制和保留期限。

---

## 5. 数据契约、不可变对象与可复现性

### 5.1 什么是数据契约

数据契约需要明确字段、类型、必填性、枚举、编码、未知字段策略和 schema 版本。没有契约时，不同 Harness 可能分别用 `tool_error`、`error`、`failed=true` 表达同一件事，统一分析会非常脆弱。

### 5.2 为什么核心字段严格，attributes 保持扩展

V1 顶层公共字段严格校验，Harness 特有信息放入 `attributes`。这样公共语义稳定，新 Harness 又能记录细节。但 `attributes` 不是垃圾桶；影响通用查询、完整性或比较语义的字段应提升为正式契约。

### 5.3 为什么对象要不可变

如果 `TraceEvent` 创建后还能修改，checksum 会失效，历史事实也可能悄悄变化。V1 使用 frozen dataclass 并递归冻结 JSON；修改的正确方式是创建新对象，而不是篡改旧事实。

### 5.4 JSON 兼容与 NaN 问题

跨语言数据平面应限制在标准 JSON 类型。Python 自定义对象或 `NaN` 在其他语言和严格 JSON 中未必能一致解析。`_json.py` 负责冻结、解冻与规范化。

### 5.5 Canonical JSON 为什么重要

下面两个对象语义相同：

```json
{"a": 1, "b": 2}
{"b": 2, "a": 1}
```

直接对原文本求哈希会不同。Canonical JSON 规定稳定的字段排序、编码和分隔方式，确保语义相同的数据产生相同字节序列。

### 5.6 checksum、内容寻址与血缘

V1 使用 SHA-256 摘要判断事件是否精确相同，让 Artifact 按内容寻址，并验证 Episode 与 Metrics 是否变化。

Lineage 回答：

> 这个 Episode、指标或诊断，是由哪些原始事件、哪个规则版本和哪些配置计算出来的？

没有血缘，数字即使看起来正确，也难以复查和复现。

---

## 6. EpisodeAssembler：怎样从事件流恢复执行

Assembler 不是简单的 `group by episode_id`，而是确定性的完整性边界。

### 6.1 输入与输出

输入是原始 `TraceEvent`、每个 Episode 的 `EpisodeContext` 和 Artifact 索引。输出是成功组装的 `AgentEpisode`、被隔离的 `QuarantinedEvent` 和完整性问题。

`EpisodeContext` 提供事件流不应重复携带的稳定上下文，例如任务 ID、四类 Manifest 和 Capture Capability。

### 6.2 组装算法

#### 第一步：全局 event_id 去重

按 ID 检查精确重复和冲突重复。要在分组前做，因为相同 ID 可能错误出现在两个 Episode 中。

#### 第二步：确认 EpisodeContext

如果事件的 `episode_id` 找不到上下文，就无法可靠知道其配置与能力，事件进入 quarantine。

#### 第三步：分组并稳定排序

同一输入无论运行多少次，都应产生相同输出和 checksum。这就是确定性。

#### 第四步：检查 sequence

检查是否连续、是否有缺口、同一 sequence 是否出现不同事件。缺口表示可能丢事件；冲突表示两个事实争夺同一逻辑位置。

#### 第五步：检查 Span 关系

若事件声明 `parent_span_id`，父 Span 应能在本 Episode 中找到。找不到就是 orphan span，说明调用关系不完整。

#### 第六步：检查期望事件对

V1 检查：

- `MODEL_REQUEST` ↔ `MODEL_RESPONSE`；
- `TOOL_CALL` ↔ `TOOL_RESULT`；
- `SANDBOX_STARTED` ↔ `SANDBOX_FINISHED`；
- `VERIFICATION_STARTED` ↔ `VERIFICATION_FINISHED`。

只看到调用却没有结果，可能是崩溃、采集丢失或尚未结束，不能假装它“返回了零次错误”。

#### 第七步：解析 Artifact 引用

事件引用 Artifact，但 Store 中找不到内容时，Episode 不完整。保留引用并报告缺失，方便后续修复。

#### 第八步：推导终止和 Outcome

V1 从明确终止事件和 `EPISODE_FINISHED` 推导结果，不扫描任意日志文本猜测成功。

#### 第九步：生成 IntegrityReport 与 Lineage

检查结果汇总为 `COMPLETE`、`PARTIAL` 或 `CORRUPT`，并记录原始事件 checksum 和到达身份。

### 6.3 三种 Integrity 状态

| 状态 | 含义 | 例子 | 核心成功率 |
| --- | --- | --- | --- |
| `COMPLETE` | 未发现缺失或冲突 | 请求响应完整，只有一个终止 | 可进入，再看 validity |
| `PARTIAL` | 部分事实缺失，但不矛盾 | 少工具结果、缺 Artifact | 通常不直接进入 |
| `CORRUPT` | 事实互相冲突 | 同 ID 不同内容、多个终止 | 不可进入 |

`PARTIAL` 是“不知道全部”，`CORRUPT` 是“已有说法互相矛盾”。

### 6.4 为什么不读自然语言猜状态

日志里出现 `tests passed` 不代表 Verifier 真通过；模型也可能只是在复述。核心状态机必须依赖明确字段和受信任组件结果。自然语言可保留在 Artifact 中供人工分析，但不能偷偷决定状态。

---

## 7. 结果必须拆成三个轴

### 7.1 TaskStatus：任务做成了吗

`SUCCESS` 表示 Verifier 认为任务完成，`FAILURE` 表示执行有效但任务未完成，`UNKNOWN` 表示当前不能形成可信的任务结论。它回答业务任务结果。

注意不要和 `EventStatus` 混淆：单个事件使用 `SUCCEEDED` / `FAILED`，整个 Episode 的任务结果使用 `TaskStatus.SUCCESS` / `TaskStatus.FAILURE`。这是两个层级的状态。

### 7.2 ExecutionValidity：这次执行是否有效

它回答“这是不是一次可以评价 Agent 能力的正常尝试”。Sandbox 平台崩溃、基础设施超时或环境构建失败时，不应直接算成模型失败。

### 7.3 IntegrityState：数据是否完整可信

它不评价任务，而评价观测数据：是否缺事件、存在冲突或缺 Artifact。

### 7.4 三轴示例

| 场景 | TaskStatus | Validity | Integrity | 解释 |
| --- | --- | --- | --- | --- |
| 正常完成并通过验证 | `SUCCESS` | `VALID` | `COMPLETE` | 可作为成功样本 |
| 模型反复调用错误工具 | `FAILURE` | `VALID` | `COMPLETE` | 真实任务失败 |
| Sandbox 平台超时 | 未成功 | `INFRA_INVALID` | `COMPLETE` | 数据完整，但不是公平能力失败 |
| 工具结果事件丢失 | 未知或失败 | 视证据而定 | `PARTIAL` | 不能把没看到当成没发生 |
| 同一事件 ID 两份内容 | 不可信 | 不可信 | `CORRUPT` | 先修数据问题 |

### 7.5 failure 和 error 不是同义词

- Task failure：系统正常运行，Agent 没做成题。
- Infra error：系统没提供一次公平、有效的尝试。
- Data integrity error：我们不能确信记录是否准确。

全部记成失败会让模型为基础设施背锅；全部丢掉又看不到系统可靠性问题。

---

## 8. Capture Capability：系统看到了多少

### 8.1 为什么要显式声明能力

有的 Harness 只给最终文本和工具调用；有的能暴露模型 usage；有的还能说明为什么重试、怎样裁剪上下文。如果不声明 Capability，分析器会把“没采到”误认为“没发生”。

### 8.2 三类观测级别

1. **Black-box**：只看到外部输入输出和部分工具行为。
2. **Hook-enabled**：Harness 主动发出内部决策、重试和终止事件。
3. **Managed / deep instrumentation**：运行时、token、延迟、上下文变换均受控采集。

具体能力以 `CaptureCapability` 枚举和 Episode 声明为准，不要只凭名称推断。

### 8.3 四个容易混淆的值

| 表达 | 含义 | 工具重试例子 |
| --- | --- | --- |
| `0` | 已观测并确认没发生 | 能看到决策，确认没有重试 |
| `None` | 当前结果没有可计算值 | 没定价，所以成本无法算 |
| `UNKNOWN` | 理论可判断，但证据不足 | 看到异常，无法确定责任层 |
| `NOT_OBSERVABLE` | 当前采集能力根本看不到 | Black-box 不暴露重试决策 |

把 `NOT_OBSERVABLE` 当 0 会偏向观测较弱的 Harness：它不是问题更少，只是你看不见。

### 8.4 诊断必须被 Capability 约束

若看到工具错误后模型重复动作，且观察到 Harness 确实回传完整错误，才更有把握讨论模型未恢复。Black-box 下不能直接断言“Harness 丢失反馈”或“模型忽略反馈”，V1 会降低归因层级。

---

## 9. Manifest：让比较具有意义

### 9.1 为什么一个 harness_name 不够

结果同时受 Harness 版本与策略、模型与采样参数、环境镜像与工具、Evaluator 规则影响。只记录名字，无法知道版本是否变化，也无法解释成功率差异。

### 9.2 四类 Manifest

| Manifest | 冻结什么 | 典型问题 |
| --- | --- | --- |
| `HarnessManifest` | Harness 身份、版本、配置摘要 | 重试策略相同吗？ |
| `ModelManifest` | 模型身份、提供方、采样配置 | 是否用了同一模型？ |
| `EnvironmentManifest` | 镜像、工具和资源环境 | 环境是否变化？ |
| `EvaluatorManifest` | 评测器及规则版本 | 判分标准改变了吗？ |

Manifest digest 是内容摘要，不只是易变名称。

### 9.3 Harness A/B 为什么应控制模型

若 Harness A 用强模型、B 用弱模型，A 成功率更高也不能归因给 Harness，模型是混杂变量。因此应固定任务、模型、采样、环境、工具和 Evaluator，只改变 Harness。

模型 API 完全可以作为统一推理后端。比较 Harness 不需要训练模型，关键是控制变量与记录 Manifest。

### 9.4 `INSUFFICIENT_EVIDENCE` 也是正确答案

若模型版本、环境或评测器没有对齐，系统应明确说证据不足，而不是生成一个看似完整的因果解释。这是可靠性能力。

---

## 10. Metrics：数字怎样才不骗人

### 10.1 指标必须有操作性定义

“工具错误率”仍需定义分子、分母、超时口径、缺失事件处理方式，以及 infra-invalid 是否纳入。只有名称没有口径，就不能比较。

### 10.2 V1 的 Episode 指标

`compute_episode_metrics` 计算：

- Episode 持续时间；
- 模型轮数和验证尝试；
- 工具调用与错误数量；
- 连续重复动作和工具循环 run；
- 工具错误后的恢复率；
- 后端明确报告的 token usage；
- 按组件聚合的延迟；
- 有定价配置时的成本。

### 10.3 重复动作与循环

V1 把工具名和规范化参数形成 action key。相邻调用 action key 相同，就是连续重复动作证据：

```text
read_file({"path":"config.json"}) → error
read_file({"path":"config.json"}) → error
```

这是事实，具体原因仍需 Attribution 结合其他证据判断。

### 10.4 recovery rate 的口径

V1 定义为：失败工具结果之后，是否出现过成功工具结果。它不等于任务最终成功，也不保证成功工具解决的是同一错误；这是简单、确定的 V1 语义。

### 10.5 token 不能靠文本猜

响应文本长度不是 token usage。tokenizer、隐藏推理、缓存和供应商计费口径都会影响结果。V1 只信后端明确 usage；没有就保持未知。成本没有定价配置就返回 `None`。

### 10.6 成功率分母要排除 infra-invalid

100 次执行中 60 成功、30 有效任务失败、10 次平台事故。评价 Agent 能力时分母是 90，成功率 `60/90`；评价端到端可用性可用 `60/100`，但名称必须说明口径。

---

## 11. Failure Attribution：从事实到解释

### 11.1 归因不是重新生成一段总结

归因输出应包含 failure layer、稳定 code、summary、evidence event IDs、confidence 和 rule version。

### 11.2 事实与解释的边界

事实包括：`e17` 是工具错误、`e21` 使用相同工具参数、二者之间有重试决策、Episode 最终失败。

解释包括：模型没有调整行动、Harness 可能没组织好错误反馈、工具持续不可用。解释必须引用事实，且不能超出 Capture Capability。

### 11.3 V1 的规则路径

`AttributionEngine` 大致按顺序判断：

1. 数据不完整或损坏时优先报告 integrity；
2. infra-invalid 时从最早 ERROR/TIMEOUT 组件定位基础设施层；
3. 有效失败且有“工具错误后重复动作”时，结合 Harness 决策能力诊断；
4. 没有足够模式时返回保守 `UNKNOWN`。

先处理 integrity，是因为事实不完整时精细归因没有可靠基础。

### 11.4 confidence 不是数学概率

V1 confidence 表达规则证据强弱，不应解释成“有 82% 概率一定是 Harness 的错”。校准概率需要真实标签和专门校准。

### 11.5 Evidence Graph 是 V2 方向

```text
tool_result(error) ─┐
                    ├─> repeated_action_after_error ─> diagnosis
same_tool_call ─────┘
harness_decision ────────────────────────────────────┘
```

每个结论可展开到具体事件，支持规则升级和人工纠正。

---

## 12. 按学习顺序阅读 V1 代码

不要按文件树顺序读，要按数据生命周期读。

### 12.1 JSON 基础与 TraceEvent

先读：

- `src/contracts/_json.py`
- `src/contracts/_validation.py`
- `src/contracts/artifacts.py`
- `src/contracts/trace_event.py`

关注 `freeze_json`、`canonical_json_bytes`、`TraceEvent.__post_init__`、`to_dict/from_dict` 和 `checksum`。

读完回答：两个事件 `event_id` 相同但 checksum 不同，应该怎么办？

### 12.2 Manifest 与 AgentEpisode

再读：

- `src/contracts/manifests.py`
- `src/contracts/agent_episode.py`

关注四类 Manifest、`EpisodeOutcome`、`IntegrityReport`、`CaptureCapability` 和 Episode checksum。

读完回答：Episode 能否同时 `infra-invalid` 且 `integrity=COMPLETE`？可以，因为“执行无效”和“记录完整”是不同维度。

### 12.3 采集与持久化

阅读：

- `src/capture/event_writer.py`
- `src/capture/recorder.py`

关注 `EventWriter.append` 的重复 ID 处理、durable 写入、坏行隔离、内容寻址、sequence 分配、`HarnessHook`、`EnvironmentCapture` 和 redaction 时机。

读完回答：为何 redaction 不能只放在 dashboard 展示层？

### 12.4 EpisodeAssembler

阅读 `src/assembly/episode_assembler.py`。先读 `assemble`，再读 `_assemble_episode`，最后读 `_missing_expected_types` 和 `_derive_terminal`。

```text
events + contexts + artifacts
    → dedupe
    → quarantine unknown context
    → group / order
    → integrity checks
    → terminal / outcome
    → lineage
    → AgentEpisode
```

读完回答：为何冲突 event ID 在分组前处理？为何缺少工具结果通常是 PARTIAL 而不是 CORRUPT？

### 12.5 Metrics

阅读 `src/analysis/metrics.py`，关注 `_tool_action_key`、`_usage`、`_latency_sum` 和 `compute_episode_metrics`。

用手算演示失败 Episode 的工具调用数、错误数和重复动作数，再与产物核对。

### 12.6 Attribution

阅读 `src/analysis/attribution.py`，关注 `FailureLayer`、`Diagnosis`、`analyze` 规则顺序和 `_repeated_action_after_tool_error`。

读完回答：为何不能只根据两个重复工具调用断言“Harness 丢失反馈”？

### 12.7 端到端演示与 CLI

最后读：

- `src/demo_v1.py`
- `src/cli.py`

演示构造三条确定性轨迹：成功、有效任务失败/工具循环、infra-invalid/Sandbox 超时。它使用确定性 clock 和 ID factory，让产物可比较、测试不受当前时间和随机 UUID 影响。

---

## 13. 动手实验：不要只读文档

重复性脚本和测试可由导师补齐，你重点理解输入、不变量和输出含义。

### 13.1 运行完整测试

```bash
python -m unittest discover -s tests -v
```

不要只看“全绿”，还要从测试名识别 strict schema、JSON 往返、幂等、冲突隔离、partial/corrupt、capability-aware attribution 和 deterministic artifacts。

### 13.2 重新生成 V1 演示

```bash
python -m src.cli --help
python -m src.cli demo-v1 --help
python -m src.cli inspect --help
python -m src.cli demo-v1 --output /tmp/agentic-rl-v1-demo
```

使用临时目录避免覆盖冻结产物。预期得到三条 Episode 及 metrics、diagnoses 和摘要。

### 13.3 从 raw events 追到 diagnosis

按顺序打开原始事件 JSONL、Episode、Metrics、Attribution report 和 Summary。

针对工具循环 Episode，记录第一次 `TOOL_CALL`、错误 `TOOL_RESULT`、Harness 重试决策和第二次相同 `TOOL_CALL` 的 event ID，再检查 Diagnosis evidence IDs 是否回指这些事实。

### 13.4 理解 exact duplicate

```text
e1(checksum=A)
e2(checksum=B)
e2(checksum=B)  # 精确重复
e3(checksum=C)
```

预期 Episode 只保留一次 `e2`，工具调用数不会重复增加。

```text
e2(checksum=B)
e2(checksum=X)  # 同 ID，不同事实
```

预期不能静默选 B 或 X；它们构成冲突并保留隔离证据。

### 13.5 制造 PARTIAL Episode

从完整序列删去一个 `TOOL_RESULT`，保留 `TOOL_CALL`。预期：

- 其他事件仍被保留；
- 缺失事件对被记录；
- Integrity 变为 `PARTIAL`；
- 指标不把缺失结果当成功或零延迟；
- Attribution 优先承认数据不完整。

### 13.6 验证确定性

相同输入连续生成两次，比较 Episode checksum、Metrics checksum、Artifact digest 和 Diagnosis 证据。输入、代码和配置没变化时派生产物应稳定；变化时 lineage 应能解释来源。

### 13.7 读懂三条冻结样例

冻结产物位于 `artifacts/v1-observability/`。

成功样例：哪个事件证明任务完成？哪个组件验证？Integrity 为何 COMPLETE？

工具循环样例：为何是 valid task failure？哪两个 action 相同？Diagnosis 引用哪些事件？

Sandbox 超时样例：第一条 ERROR/TIMEOUT 来自哪个组件？为何不计入模型失败分母？数据是否仍完整？

---

## 14. V1 和 Stage 1–3 的关系

### 14.1 前面的 Stage 不是废弃工作

Stage 1–3 建立面向训练数据的下游契约：`RolloutRecord`、Capability、`RolloutBatch`、`TrainingReadyBatch` 和 `ResampleRequest`。V1 新增的是它们之前的“执行事实层”。

### 14.2 两层是不同视图

```text
执行世界                         训练世界

TraceEvent
   ↓
AgentEpisode  ──投影/筛选/转换──> RolloutRecord
   │                                  ↓
   ├─ Metrics                    RolloutBatch
   ├─ Diagnosis                       ↓
   └─ Integrity                 TrainingReadyBatch
```

`AgentEpisode` 追求可审计和因果上下文，保留组件事件、Manifest、Artifact、Integrity 和终止证据。

`RolloutRecord` 追求训练消费，关注 prompt/response、token、mask、reward、logprob 和训练能力。

### 14.3 为什么先 Episode，再训练视图

如果直接从 Harness 临时输出拼样本，将难以排除 infra-invalid、追溯失败、统一不同 Harness 语义、重放转换和解释训练异常。

Episode 层使训练投影成为版本化、可测试、可重复的派生过程。

---

## 15. 和 Harness 改进、Harness 评测的关系

### 15.1 不只是“给 Harness 打分”

评测回答谁的成功率、延迟、成本和循环率更好；改进回答失败集中在哪层、错误后为何不恢复、哪种重试浪费 token、哪些任务应进入回归集。

没有 Trace 和 Attribution，评测只能告诉你分数变化；有执行证据，才能指导下一版 Harness 修改。

### 15.2 典型闭环

```text
运行 Harness A/B
      ↓
采集统一 Trace
      ↓
计算成功率、循环、恢复、延迟
      ↓
按 failure layer 聚类失败
      ↓
修改 prompt / feedback / retry / termination
      ↓
固定其他 Manifest 后重新运行
```

---

## 16. 和 Agentic RL、off-policy 数据的关系

### 16.1 轨迹生产和训练是两个阶段

当前先生产可信轨迹是正确顺序。任何 RL 算法都需要知道样本是否有效、行为模型是谁、reward/verifier 版本、token/action 是否完整、哪些 token 可训练、能否回溯原 Episode。

### 16.2 轨迹模型必须和训练模型一致吗

不一定，但语义不同：

- 行为策略与当前策略一致或接近，是典型 on-policy / near-on-policy 数据；
- 强教师生成轨迹训练弱学生，通常是 off-policy、蒸馏或行为克隆数据，取决于训练目标。

强模型轨迹可能提供高质量示范，但不会自动带来有效 Agentic RL：

- 学生可能无法表示教师策略；
- tokenizer、工具格式和上下文分布可能不同；
- RL 若需要 behavior logprob，API 轨迹可能没有；
- 只有成功轨迹会削弱错误恢复学习；
- 分布偏移可能导致训练不稳定。

所以 Manifest 必须记录行为模型，不能把不同策略轨迹混成“同一种 rollout”。

### 16.3 V1 已经准备了什么

已有行为模型/Harness/环境/评测器身份、完整性、执行有效性、工具过程、Artifact、lineage、失败诊断和指标。

后续还要实现：

- `AgentEpisode → RolloutRecord` 版本化投影；
- token/action/observation 精确对齐；
- trainable mask；
- reward 与 credit assignment；
- behavior logprob 能力声明；
- off-policy 数据策略和训练实验。

---

## 17. V1 的边界与下一步

### 17.1 V1 有意没有解决的事情

- Writer 是单进程本地基线，不是分布式消息系统；
- JSONL 适合演示和重放，不是大规模分析最终格式；
- Attribution 是规则系统，不是校准因果模型；
- Metrics 是 Episode 级基线，尚无完整 Run 级统计；
- Artifact Store 尚无生产级权限和生命周期；
- 还没有多 Harness Adapter 的真实接入；
- 尚未完成 Episode 到训练数据的投影。

### 17.2 V2 的合理优先级

1. 定义 Harness Adapter / capture protocol，接入至少两种 Harness。
2. 增加 Run 级 Manifest、任务集版本和 A/B 对齐检查。
3. 把 raw event、Episode、Metrics、Diagnosis 的 lineage 串成可查询链路。
4. 增加 Run 聚合、有效分母、Capability coverage 和对比报告。
5. 增加失败聚类与 regression set 导出。
6. 实现版本化 `AgentEpisode → RolloutRecord` 投影。

### 17.3 每完成一阶段如何复盘

1. 解决了什么具体问题？
2. 新增了什么稳定契约或不变量？
3. 输入、输出和失败模式是什么？
4. 用哪个测试或 Artifact 证明它工作？
5. 还有哪些 `UNKNOWN` / `NOT_OBSERVABLE`？
6. 下一阶段依赖它的哪项能力？

如果只能说“加了几个类和测试”，说明复盘还停留在代码表面。

---

## 18. 面试时怎样讲

### 18.1 30 秒版本

我做的是一个面向多 Harness 的 Agent 执行数据平面。系统把模型、Harness、工具、Sandbox 和 Verifier 的执行过程统一成版本化事件，先以 append-only 方式可靠落盘，再确定性组装为带完整性、能力声明、Manifest 和血缘的 Episode。在此之上计算循环、恢复、延迟和 token 等指标，并做 capability-aware 的失败归因，从而支持控制变量的 Harness 对比和后续 Agentic RL 数据生产。

### 18.2 为什么不直接存最终结果

最终结果不能解释失败层，也无法区分任务失败、基础设施无效和采集不完整。append-only 事件和 lineage 让组装、指标与诊断可以重放和审计。

### 18.3 做了哪些可靠性设计

- strict schema 和版本；
- 不可变事件与 canonical checksum；
- at-least-once 下的 event ID 幂等；
- 冲突重复与坏数据 quarantine；
- sequence 缺口、事件对、orphan span、Artifact 完整性；
- `flush + fsync` 本地持久化基线；
- 派生数据 lineage 和确定性重放。

同时说明边界：尚未声称分布式 exactly-once。

### 18.4 怎样公平比较 Harness

固定任务、模型、采样、环境和 Evaluator，只改变 Harness；用 Manifest digest 验证控制变量；按 Capability 对齐可比较指标；单列 infra-invalid 和 integrity-invalid；证据不足时输出 `INSUFFICIENT_EVIDENCE`。

### 18.5 怎么服务 RL

AgentEpisode 是可审计事实层，RolloutRecord 是训练视图。后续通过版本化投影筛选 valid、complete Episode，生成 token/action/reward/mask 等字段，并保留行为模型、Harness、Verifier 与原事件血缘。

---

## 19. 常见误区

### 误区一：没有错误事件就是错误数为 0

错误。先确认相关事件是否可观测、事件对是否完整。看不到可能是 `NOT_OBSERVABLE` 或 `PARTIAL`。

### 误区二：模型写了“完成”就是成功

错误。成功应由受信任 Verifier 的结构化结果决定。

### 误区三：Sandbox 超时就是模型失败

错误。要判断超时来自模型预算、命令自身还是基础设施；平台事故通常属于 infra-invalid。

### 误区四：相同 event_id 保留第一条即可

错误。必须比较 checksum；同 ID 不同内容是事实冲突。

### 误区五：有 Trace 就能准确归因

错误。归因受 Capability、完整性和证据约束，系统必须允许 `UNKNOWN`。

### 误区六：Harness 比较必须本地训练模型

错误。可使用统一模型 API；关键是固定模型和其他变量。

### 误区七：强模型轨迹一定能训练好弱模型

错误。还要处理策略分布、格式兼容、logprob、reward、容量差异和训练目标。

### 误区八：测试全绿就代表生产就绪

错误。测试证明已声明的不变量；并发、容量、权限、远端存储、灾难恢复和真实接入仍要验证。

---

## 20. 自测题与参考答案

先不看答案，用自己的话回答。

### 20.1 基础题

1. Agent 和模型有什么区别？
2. Harness 对执行结果有哪些影响？
3. Event、Span 和 Episode 分别是什么？
4. 为什么大输出适合放 Artifact？
5. 为什么 sequence 不能由 timestamp 替代？

### 20.2 数据可靠性题

6. append-only 日志解决哪些问题？
7. exact duplicate 与 conflicting duplicate 有何区别？
8. PARTIAL 与 CORRUPT 有何区别？
9. checksum 为什么需要 canonical JSON？
10. `fsync` 能否让我们宣称分布式 exactly-once？

### 20.3 分析题

11. 为什么 task status、validity、integrity 必须拆开？
12. `0`、`None`、`UNKNOWN`、`NOT_OBSERVABLE` 分别是什么？
13. 错误后重复动作能直接证明 Harness 有 bug 吗？
14. 为什么 infra-invalid 不进入模型成功率分母？
15. Attribution 为什么保存 evidence IDs 和 rule version？

### 20.4 系统设计题

16. 怎样公平比较 Harness A 和 B？
17. Harness 不暴露 token usage 时，成本怎样比较？
18. 中间一条 JSONL 损坏时怎样处理？
19. AgentEpisode 和 RolloutRecord 为什么不合成一个对象？
20. V2 接真实 Harness 时最先标准化什么？

### 20.5 参考答案要点

1. 模型生成决策；Agent 还包含 Harness、上下文、工具和执行循环。
2. 它控制 prompt、反馈、工具解析、重试、终止、预算和上下文。
3. Event 是单个事实，Span 是一段操作，Episode 是一次任务尝试。
4. 避免事件膨胀、重复存储和治理困难，内容寻址也便于复用。
5. 时钟会漂移且事件会乱序；sequence 表达逻辑顺序。
6. 审计、重放、恢复、幂等消费和派生血缘。
7. 同 ID 同内容可忽略；同 ID 不同内容必须隔离并报冲突。
8. PARTIAL 是信息缺失；CORRUPT 是信息互相矛盾。
9. 让语义相同但字段顺序不同的对象得到同一摘要。
10. 不能，它只是本地持久化的一层保证。
11. 分别评价任务、执行环境和数据，避免互相污染。
12. 已知为零、不可计算值、证据不足、能力上不可观测。
13. 不能，还需决策、反馈等证据和能力声明。
14. 它不是一次公平能力尝试，否则基础设施事故会归因给模型。
15. 让结论可审计、可聚合，规则升级后可重放。
16. 固定任务、模型、环境和评测器，只改变 Harness，并验证 Manifest/Capability。
17. 标为未知或不可观测，不用 0 填充，不做虚假精确比较。
18. 隔离坏行、保留合法事件、报告 integrity，避免整批失败或静默丢弃。
19. 一个服务审计诊断，一个服务训练消费；不变量和生命周期不同。
20. Adapter 事件语义、身份传播、Capability 和 Manifest 采集。

---

## 21. 最终复盘清单

- [ ] 用两分钟讲清项目目标与 V1 数据流。
- [ ] 用贯穿示例解释为什么 `success=false` 不够。
- [ ] 画出 Event → Episode → Metrics/Diagnosis → RolloutRecord。
- [ ] 解释 append-only、幂等、quarantine 和 lineage。
- [ ] 区分 task failure、infra-invalid、partial、corrupt。
- [ ] 区分 0、None、UNKNOWN、NOT_OBSERVABLE。
- [ ] 打开冻结 Episode，找到 Manifest、Outcome、Integrity 和事件证据。
- [ ] 手算工具循环样例的关键指标。
- [ ] 说明为何 Harness A/B 要固定模型 API 和其他变量。
- [ ] 说明强模型轨迹训练弱模型时为何是不同策略分布。
- [ ] 主动说出 V1 边界和 V2 优先级。

如果某一项讲不清楚，就回到对应章节，用代码或 Artifact 找一个具体例子。真正掌握的标志不是记住术语，而是能从一条执行事实出发，解释系统为何这样建模、哪里可能出错、怎样用证据验证。
