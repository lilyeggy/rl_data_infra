# Agent Infra Architecture：Trace / Data / Observability

## 1. 系统定位

Harness 是 Agent 的控制平面：它决定 context、tool、retry、compaction、verification 和 termination。当前项目是执行数据平面：它保存 Harness 运行时发生的可观测事实，把事实转成可以回放、查询、诊断和比较的数据产品。

```text
                    Control Plane
       Harness: context / loop / retry / terminate
                            │
                            ▼
Model API ─── Tool ─── Sandbox ─── Verifier
    │           │          │            │
    └────────── Capture / Hook ──────────┘
                            │
                    TraceEvent log
                            │
                      AgentEpisode
                            │
              Metrics / Attribution / Compare
                            │
                    Regression Gate
```

系统不执行模型训练，也不自动生成 Harness patch。它负责让修改前后的行为变化有证据可查。

## 2. 为什么先写 Event Log，而不是直接保存 Episode

Agent 执行具有长事务特征：一次任务可能持续数分钟，期间会发生模型重试、工具超时、进程崩溃和并发回调。如果只在结束时写最终对象，中途崩溃会丢掉全部证据。

V1 使用 append-only JSONL，相当于本地 Write-Ahead Log：

- 每个事实完成后立即追加；
- 已写事实不原地更新；
- `event_id` 是幂等键；
- 相同 ID、相同内容的 at-least-once 重放被去重；
- 相同 ID、不同内容意味着 producer 违反不可变约束，进入 quarantine；
- 单行损坏不会让相邻正确事件消失。

生产系统可把文件替换为 Kafka、对象存储或数据库事务日志，但 canonical contract 不需要改变。

## 3. Identity、Order 与 Causality

事件包含四类不同身份：

| 字段 | 回答的问题 |
|---|---|
| `run_id` | 属于哪次实验运行？ |
| `episode_id` | 属于哪次 task attempt？ |
| `trace_id` | 属于哪棵分布式调用链？ |
| `span_id / parent_span_id` | 哪个动作导致了哪个子动作？ |

`sequence` 用于 producer 的逻辑顺序，timestamp 用于时间观测。Assembler 的稳定排序键是 `(sequence, timestamp, event_id)`，但它不会把 sequence collision 或 gap 悄悄修成“完整数据”。排序解决展示确定性，integrity 负责暴露数据质量；这是两个不同问题。

`span_id` 表达因果嵌套，不只是 UI 缩进。工具调用可以是模型 turn 的 child span，Verifier 可以是独立 root span。V1 保留多 trace/span，但尚未冻结多 Agent 的 reward ownership 和 credit assignment，所以不能声称完整多 Agent 训练语义。

## 4. Outcome 与 Integrity 必须分开

Outcome 回答任务结果，Integrity 回答数据是否完整：

```text
TaskStatus:        SUCCESS | FAILURE | UNKNOWN
ExecutionValidity: VALID | INFRA_INVALID | UNKNOWN
IntegrityState:    COMPLETE | PARTIAL | CORRUPT
```

典型组合：

| 组合 | 语义 |
|---|---|
| `SUCCESS + VALID + COMPLETE` | 可信成功 |
| `FAILURE + VALID + COMPLETE` | Agent 真实做错，Verifier 正常工作 |
| `UNKNOWN + INFRA_INVALID + COMPLETE` | Trace 完整记录了一次基础设施无效执行 |
| `UNKNOWN + UNKNOWN + PARTIAL` | capture 中断，不能判断任务结果 |
| 任意 outcome + `CORRUPT` | canonical identity/order 冲突，强结论应被阻断 |

因此“Episode 完整”不等于“任务成功”，“任务失败”也不等于“系统坏了”。这是 Agent evaluation 和 RL 数据清洗中最容易混淆的边界之一。

## 5. Capability 是观测权限，不是装饰字段

Black-box proxy 能看到 model/tool I/O，却看不到 Harness 为什么 retry。Hook-enabled capture 才能可靠记录 context selection、compaction 和 termination decision。

如果缺少 `TOOL_IO`，工具调用次数必须是 `None + NOT_OBSERVABLE`，不能写成 0；写 0 会把“没有采集”误报成“Agent 没有使用工具”。同理，没有 pricing manifest 时 cost 不可观测，没有 token usage 时不能用文本重新估算成 canonical token 数。

Capability 的意义是让 consumer 在运行前或计算时检查证据是否足够：

```text
Tool recovery metric       requires TOOL_IO
Compaction diagnosis       requires CONTEXT_COMPACTION + HARNESS_DECISIONS
Verifier-aware termination requires VERIFIER_EVIDENCE + TERMINATION_DECISIONS
GRPO Training View         requires target-policy token/logprob semantics
```

## 6. Manifest 是因果归因的前置条件

Harness A/B 只能让目标 Harness 变量变化。模型 provider/revision、sampling、task revision、Sandbox、tool schema、Verifier、seed 或 timeout 不一致，都会形成 confounder。

V1 已把 Harness、Model、Environment 和 Evaluator manifest 固定进 Episode，但尚未执行跨 run compatibility check。V2 会把检查放在 comparison 之前：不兼容时返回 `INSUFFICIENT_EVIDENCE`，而不是生成一个看似精确的涨跌百分比。

## 7. Raw Fact、Metric 与 Diagnosis 的边界

下面是事实：

```text
TOOL_RESULT status=FAILED exit_code=1
后续 TOOL_CALL 的 tool_name + arguments 与上次相同
VERIFICATION_FINISHED status=FAILED
```

下面是 V1 规则的派生判断：

```text
layer=HARNESS
reason_code=TOOL_ERROR_FEEDBACK_LOSS
confidence=0.8
evidence_event_ids=[failed_result, retry_decision, repeated_call]
rule_version=failure-attribution/v1
```

派生记录必须保存 Episode checksum、规则版本和 evidence IDs。规则升级时可以对同一份事实重跑，不需要篡改历史 Episode。

当只有黑盒轨迹，或没有实际捕获到对应 `HARNESS_DECISION` 时，同样的重复动作只能诊断为 `UNKNOWN / OBSERVED_TOOL_ERROR_LOOP`。系统可以报告行为模式，但不能把 Harness 内部原因伪装成已观测事实。

## 8. Artifact 为什么内容寻址

stdout、stderr、patch 和 verifier report 可能很大，不适合复制到每个 event。V1 使用 SHA-256 文件名：

- 相同内容自然去重；
- `ArtifactRef.sha256` 能发现内容漂移；
- event 只保存稳定 artifact ID；
- diagnosis 可以从 evidence event 继续追到外部原始证据。

Artifact 缺失不会被静默忽略；Assembler 会把 Episode 标成 `PARTIAL` 并列出缺失引用。

## 9. V1 的可靠性边界

当前 writer 保证单进程内幂等和落盘 `fsync`，但没有实现多进程 file lock 或分布式事务。多 writer 部署需要 Kafka partition、数据库 unique constraint 或显式锁。

当前 metric 是 per-Episode deterministic derivation，没有统计置信区间。当前 attribution 是小型规则引擎，没有使用 LLM Judge，也不声称单因果真相。当前演示是机制证据，不是 benchmark 泛化证据。

这些限制会直接进入 V2 的 Gate 和 release artifact，而不是只写在口头说明中。
