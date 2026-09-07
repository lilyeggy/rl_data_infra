# V2 教学复盘：从真实 Pi Trace 到 Harness 上线决策（已归档）

这份文档用于你隔几天甚至隔几周回来时，能回答四个问题：这个项目究竟在做什么、V1 已经解决了什么、V2 新增了什么、看到 `REJECT` 后下一步该改哪里。

## 1. 先用一句话理解整个项目

我们在做一套 Harness-neutral 的 Agent Execution Data Plane：把 Pi、Claude Code、Kimi Code 等不同 Harness 的真实执行变成统一且可审计的 Episode，再基于同一份事实做故障诊断、Harness 对比、上线 Gate，并为之后的数据筛选和 Agentic RL 留出规范接口。

这里的主角不是模型训练，而是 Agent Infra。训练系统只能消费已经被正确记录、验证、去污染的数据；如果基础 trace 连“任务失败”和“模型服务挂了”都分不开，后面的 reward、SFT 或 RL 都会学到错误信号。

## 2. V1 和 V2 的关系

V1 回答的是纵向问题：

```text
一次执行发生了什么？
  → 哪些 event 是原始事实？
  → 能否确定性组装成 Episode？
  → Episode 是否完整？
  → 为什么失败？证据在哪里？
```

V2 回答的是横向问题：

```text
Harness candidate 是否比 control 更值得上线？
  → 两边是否真的只差 Harness policy？
  → 是否能按相同任务配对？
  → 正确率、失败类型、成本、延迟怎么变化？
  → 证据足够吗？最终 ACCEPT、REJECT 还是 INSUFFICIENT_EVIDENCE？
```

所以 V2 没有另造数据模型。它继续消费 V1 的 `TraceEvent → AgentEpisode`，在上面增加 Experiment、Comparison、Gate 和 Observatory。这一点很重要：你可以从 Gate 的失败规则一路下钻到 Episode、ToolResult，最终回到 Pi 原始协议，而不是看一张无法审计的统计表。

## 3. 这次真实跑了什么

我们给 Pi 准备了三个本地、非敏感的 JSON 文件任务。每个任务都故意要求先读取一个不存在的文件：

```text
missing-1.json  → FILE_NOT_FOUND
真实文件 task-1.json 中有 2 条 FAILURE 记录
```

两组唯一有意改变的变量是错误恢复策略：

```text
Control
  read missing-1.json → error
  read missing-1.json → error（原样重复）
  输出失败

Candidate
  read missing-1.json → error
  find task-*.json
  read task-1.json
  输出正确 JSON
```

三组都固定：

- Harness runtime：Pi 0.84.2；
- provider/model：`opencode-go/gpt-5.6-luna`；
- thinking：minimal；
- tools：read、grep、find、ls；
- task snapshot 和环境；
- exact JSON Verifier。

没有 silent fallback。假如 Luna 后端报错，我们不会偷偷换另一个模型后把结果算进 Luna 实验；该次执行会标为 `UNKNOWN + INFRA_INVALID`。

## 4. 为什么生成轨迹的模型可以和以后训练的小模型不同

可以不同，但必须把语义说对。

这次行为模型是 `gpt-5.6-luna`，未来训练目标可能是本地小模型，所以它们是 off-policy 数据。这里至少有三种用途：

| 用途 | 是否可用 | 原因 |
|---|---|---|
| Harness 评测 | 可以 | 比较两种 Harness 时，只要两臂固定同一行为模型即可 |
| SFT/蒸馏候选 | 可以 | 通过 Verifier 的高质量动作和结果可供筛选 |
| 严格 on-policy RL | 不可以 | 没有目标模型自身采样的 token ids、logprobs 和策略版本 |

“off-policy”不是“坏数据”的同义词。它只是说明行为来自另一个策略。错误发生在把 teacher trace 假装成目标策略的 on-policy rollout，然后直接套 PPO/GRPO 的重要性假设。

## 5. Pi Adapter 为什么是 Agent Infra 核心

Pi 输出自己的 NDJSON 协议，其他 Harness 也会有不同字段。上层 comparison 不应该为每个 Harness 重写一次。因此 Adapter 做的是 anti-corruption layer：

```text
Pi session/message/tool protocol
              ↓ PiJsonAdapter
统一 TraceEvent(event_type, component, status, span, sequence, timestamp, attrs)
```

它完成这些关键工作：

1. 把 assistant message 转成成对的 `MODEL_REQUEST / MODEL_RESPONSE`；请求正文不可见时写 `NOT_OBSERVABLE`，不猜 prompt。
2. 用 Pi 的 `toolCallId` 将 `TOOL_CALL` 与 `TOOL_RESULT` join 起来。
3. 从重复的 toolResult protocol record 找到真实时间戳，使 Episode duration 接近真实墙钟时间，而不是事件数量。
4. 保留 input/output/reasoning/cache token 和 provider cost 事实。
5. 区分进程退出码和语义成功：exit code 0 但连续 backend 500 仍然是 infra-invalid。
6. 捕获后立即脱敏 credential、signature 和 plaintext thinking；公开 fixture 不携带 chain-of-thought。

注意：Pi 黑盒协议没有暴露 Harness 内部 policy decision。我们虽然知道实验中给了哪条 system policy，但 Trace 本身只能证明动作序列，不能证明内部究竟为何做出这个动作。

## 6. 独立 Verifier 为什么不能省

Agent 最后输出：

```json
{"task_id":"task-1","status":"SUCCESS","failure_count":2}
```

不能因为它写了 `SUCCESS` 就把 Episode 标成成功。模型既可能算错，也可能输出不合法 JSON。因此 `verify_reference_answer` 会独立检查：

- 能否解析成 JSON；
- key 是否恰好是规定的三个；
- task_id 是否对应当前任务；
- failure_count 是否等于 ground truth；
- status 是否为预期值。

Control 即使输出结构化的 `FAILURE`，也只是一次 `VALID task failure`；Candidate 只有全部字段匹配才是 `SUCCESS`。Verifier 的判定 event id 会进入 Episode outcome evidence。

## 7. ExperimentManifest 解决什么问题

如果今天 control 用 Luna、candidate 用另一个更强模型，即使 candidate 更好，也不能归因于 Harness。`ExperimentManifest` 冻结：

- experiment id/revision；
- task dataset revision；
- control/candidate run id；
- 最小 pair 数；
- 目标 failure slice；
- 模型固定且无 fallback 的策略。

Comparison 还会逐 pair 核对 model、environment、evaluator 和 experiment ref。Harness manifest 被允许不同，因为它正是实验变量。其他字段不一致会进入 `compatibility_mismatches`，并使 Gate 不能给出可信 ACCEPT/REJECT。

配对键是：

```text
task_id + seed + attempt
```

本次 API 没有暴露 seed，所以显式写 `seed=NOT_OBSERVABLE`。这比假造 seed=7 更诚实；当前 pairing 依靠同一 task 和 attempt，结论边界也必须承认解码随机性没有被完全控制。

## 8. 从 Trace 如何归因到 failure slice

Control 的三条 Episode 都出现：

```text
TOOL_CALL(read, missing-X.json)
TOOL_RESULT(ERROR, FILE_NOT_FOUND)
TOOL_CALL(read, missing-X.json)  ← 参数完全相同
```

AttributionEngine 能用 canonical JSON 比较两个动作，从而检测重复。但因为没有 `HARNESS_DECISION` capability，归因只能是：

```text
layer       = UNKNOWN
reason_code = OBSERVED_TOOL_ERROR_LOOP
confidence  = 0.5
```

如果未来我们给 Pi 做显式 Hook，捕获了 decision=`RETRY_UNCHANGED` 与 reason code，才可以升级为：

```text
layer       = HARNESS
reason_code = TOOL_ERROR_FEEDBACK_LOSS
confidence  = 0.8
```

这体现了 observability 的基本纪律：没有观测能力时降低结论强度，不把合理推断包装成事实。

## 9. Comparison 到底比较了什么

逐 pair 输出：

- outcome transition，例如 `FAILURE→SUCCESS`；
- duration、turn、tool call、duplicate action、input/output token delta；
- control/candidate 的 reason codes；
- pair key 和两个 Episode id。

聚合输出：

- success rate 和 delta；
- infra-invalid rate；
- mean token 与增长比例；
- mean duration 与增长比例；
- 目标 failure slice 数量；
- paired coverage；
- compatibility mismatch。

infra-invalid 不进入 task success 分母。否则供应商服务故障会被误计成模型任务失败，并污染 Harness 的质量判断。

## 10. 为什么成功率 0→100%，Gate 仍然 REJECT

这正是 V2 最有价值的地方。Comparison 给事实，Gate 执行事先定义的上线政策。默认阈值要求：

- pair coverage = 100%；
- 至少 3 对；
- success rate 不下降；
- target failure slice 必须改善；
- token 增长不超过 20%；
- latency 增长不超过 25%；
- infra-invalid 不增加；
- 不产生新的 `SUCCESS→FAILURE`。

真实结果是：

```text
success       0/3 → 3/3       PASS
tool loop     3   → 0         PASS
tokens        +43.54%         FAIL（阈值 +20%）
latency       +51.15%         FAIL（阈值 +25%）
infra invalid 0   → 0         PASS
```

因此 `REJECT` 是正确实现，不是项目失败。它准确表达：“恢复策略有效，但当前实现太贵、太慢，不能按这份发布政策直接上线。”如果事后把阈值改成 60% 只为得到 ACCEPT，就是典型的 experiment p-hacking。

三态 Gate 的含义：

- `ACCEPT`：证据充分且每条上线规则通过；
- `REJECT`：证据充分，但至少一条质量/成本规则失败；
- `INSUFFICIENT_EVIDENCE`：配对不足、coverage 不够、存在 confounder，或关键 metric 不可观测。

## 11. Observatory 为什么必须是只读的

Observatory 展示 Compare、Gate 和 Episode timeline，但不在浏览器中重新计算真相，也不会修改 raw event。HTML 内嵌的是已经 checksum 化的 canonical/derived artifact。

面试演示时可以按这个顺序：

```text
Gate REJECT
  ↓ 展开失败规则：token / latency
Comparison：0/3→3/3，同时开销增加
  ↓ 打开 control pair
Episode：第二次 read 参数与第一次相同
  ↓ 查看证据
TOOL_RESULT(FILE_NOT_FOUND) + repeated TOOL_CALL
```

这比只展示“成功率提高”更像真正的 Agent Infra：结论可下钻、可追溯、可复算。

## 12. TrainingCandidateView 应该怎样理解

通过 Verifier 的 3 条 candidate Episode 被标为 SFT candidate，但没有被直接导出成训练样本。View 记录：

- Episode checksum 和行为模型；
- outcome、Verifier、integrity；
- teacher 与 target policy 的关系；
- SFT eligibility；
- on-policy RL eligibility；
- 缺少哪些 capability。

当前结果是 3 条 SFT candidate、0 条 on-policy RL candidate。原因不是轨迹质量差，而是行为模型与目标学生不同，并且 API 没提供学生策略自己的 token ids/logprobs。

下一阶段如果做训练数据，需要另写 projection：从 Episode 选择 observation/action/tool result，删除 evaluator 泄漏和不可训练字段，定义 truncation、loss mask、数据版本及去重策略。不能把整份 raw trace 直接喂进训练器。

## 13. 你现在如何复盘和运行

先运行测试和冻结重放：

```bash
python3 -m unittest discover -s tests -v
python3 -m src.cli demo-v2 --output artifacts/v2-harness-decision
```

然后按顺序看：

1. `artifacts/v2-harness-decision/summary.json`：先知道结论和边界；
2. `gate-result.json`：找到两个失败的效率规则；
3. `comparison.json`：看每对任务和聚合 delta；
4. `diagnoses-control.jsonl`：看 tool loop 的 evidence ids；
5. `episodes-control.jsonl`：沿 id 找到具体 ToolResult/ToolCall；
6. `capture-evidence.json`：确认独立 Verifier 和源 checksum；
7. `observatory.html`：用 UI 完成同一条下钻路径。

## 14. 下一步最应该由你写的核心逻辑

下一迭代的核心不是再堆 Adapter，而是设计一个效率更高的 recovery policy。你可以选择并实现其中一个核心策略：

```text
FILE_NOT_FOUND
  → 从错误路径提取 basename/pattern
  → 最多一次有界目录搜索
  → 候选唯一时直接 read
  → 候选不唯一时按稳定规则选择或终止
  → 缓存 discovery，后续相同目录不再搜索
```

你负责决定状态机、边界条件和为什么这样做；重复 fixture、comparison assertions、Gate regression tests 和批量重放代码可以由我补齐。目标不是让阈值迁就实现，而是在相同 Gate 下把 +43.54% token、+51.15% latency 压下来，同时保持 3/3 success 和 0 tool loop。

## 15. 面试时的三分钟讲法

> V1 我先做了 Harness-neutral event log 和 deterministic Episode assembler，把 task failure、infra-invalid、trace integrity 和 capability boundary 分开。V2 接入真实 Pi NDJSON，用固定的 GPT-5.6 Luna 对相同任务做 control/candidate 成对实验，外部 Verifier 不信任模型自报成功，并通过 compatibility report 排除非 Harness confounder。Candidate 消除了三次重复工具错误并把 success 从 0/3 提到 3/3，但 token 增长 43.5%、延迟增长 51.1%，超过冻结 Gate 阈值，因此系统诚实输出 REJECT；所有结论都能从 Gate 下钻到 Episode 和原始 ToolResult。通过的 teacher trajectories 只标为 off-policy SFT candidates，没有冒充 on-policy RL rollouts。

这段讲法的亮点不是“我调了一个 prompt”，而是你建立了从 runtime facts、数据契约、实验控制、可解释诊断到发布决策的完整 Agent Infra 闭环。
