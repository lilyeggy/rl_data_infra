# Agent Improvement Data Plane：面试级吃透实操手册

> 目标：不是“读过项目”，而是能独立解释、排查、修改并在面试中守住技术追问。
>
> 推荐节奏：每次 60–90 分钟；每完成一关再进入下一关。全程优先操作真实 artifact 和测试，不按目录从头阅读。

## 开始前：项目的正确心智模型

这不是一个训练框架，也不是一个普通的 Agent 应用。它是位于 Harness 与训练/分析消费者之间的**可信执行数据平面**：把一次真实 Agent 执行记录成不可变证据，组装、验证、认证后，再允许不同消费者使用它。

```text
Task + Environment + Harness + Policy
                  │
          Producer / Capture
                  │
       immutable TraceEvent evidence
                  │
    AgentEpisode + ExecutionBundle
                  │
           certification
                  │
    ┌─────────────┴─────────────┐
    ▼                           ▼
Harness compare / gate     SFT / Preference / RL dataset
```

先记住三条底线：

1. **事实与解释分离**：事件是事实；指标、诊断、训练资格是可版本化的派生结论。
2. **任务失败不等于基础设施失败**：真实失败轨迹可能是有效训练/分析证据；基础设施失效不能被伪装成 reward 0。
3. **不能证明就拒绝**：能力、identity、policy lineage、数据切分或 token 语义不足时，系统 fail closed。

## 使用方式

每一关都按固定循环执行：

1. 先不看答案，完成“预测”。
2. 执行命令或阅读限定文件。
3. 用自己的话写下“结论”。不要复制文档。
4. 完成“验收问题”；答不出才回到代码。
5. 将答案发给 AI，让 AI 只做追问、纠错和补洞，不让它直接替你总结。

建议在本文件同级新建自己的 `interview-notes.md`，每关最多写半页。笔记要记录结论与证据位置，而不是大段伪代码。

---

## 第 0 关：建立项目地图（45 分钟）

### 目标

能在不打开源码的情况下讲清项目问题、边界、主链路和当前真实状态。

### 只读这些材料

1. [README](../README.md)
2. [项目范围](../PROJECT_SCOPE.md)
3. [架构文档](architecture.md) 的第 1、4、5、6 节

### 产出

写一段 90 秒项目介绍，必须覆盖：

- 项目解决的具体可信性问题；
- Harness、Producer、Polar、Slime 各自的边界；
- 为什么不能直接把普通 Agent trace 当作 RL 数据；
- 已经验证的能力，以及尚未完成、不能夸大的能力。

### 验收问题

1. 这个系统“拥有”的能力和“不拥有”的能力分别是什么？
2. 为什么项目把 Local Docker Launcher 设为默认入口，而把 Polar 视为插件？
3. 当前 SFT candidate 的真实结论是什么？

### 面试表达

> 我们没有重写 trainer，而是在异构 Harness 和训练/分析消费者之间建立可信数据边界。核心价值是让每个训练或晋升结论都能回溯到具体、不可变、可认证的执行证据。

---

## 第 1 关：先看一条真实执行，而不是先读源码（60 分钟）

### 目标

从真实 artifact 还原一次执行生命周期，并区分事实、结果和完整性。

### 操作

在仓库根目录运行：

```bash
python3 -m src.cli inspect \
  --episodes artifacts/v1-observability/episodes.jsonl \
  --episode-id episode-tool-loop
```

然后依次打开：

- `artifacts/local-execution-smoke-001/raw-events.jsonl`
- `artifacts/local-execution-smoke-001/finalized/episode.json`
- `artifacts/local-execution-smoke-001/finalized/execution-bundle.json`
- `artifacts/local-execution-smoke-001/finalized/finalization.json`

### 预测后验证

在读取 `episode.json` 前，先预测它会比 raw events 多出哪些内容；在读取 bundle 前，先预测它会保存大 payload 还是 checksum 引用。

### 产出：一张执行证据表

| 阶段 | 输入 | 主要输出 | 可信性作用 |
|---|---|---|---|
| Capture | Harness/Proxy 事实 | `TraceEvent` | 及时保存不可变事实 |
| Assemble | event + artifact refs | `AgentEpisode` | 重建有序、可判断完整性的执行 |
| Bundle | Episode + trace + verifier refs | `ExecutionBundle` | 将同一次 attempt 的证据严格关联 |
| Certification | bundle + consumer profile | `EligibilityDecision` | 决定哪些消费者可以使用 |

### 验收问题

1. 为什么 episode 是派生物，而不是采集时直接写出的最终对象？
2. `ExecutionBundle` 为什么保存 checksum，而不是复制所有 artifact？
3. `COMPLETE`、`VALID`、`SUCCESS` 分别回答什么问题？请给出一个三者不一致的合法组合。

### 限定源码

只在答完问题后阅读：

- [TraceEvent](../src/contracts/trace_event.py)
- [AgentEpisode](../src/contracts/agent_episode.py)
- [ExecutionBundle](../src/contracts/execution_bundle.py)

不要展开其他 contracts。

---

## 第 2 关：掌握“脊柱”——契约与不变量（90 分钟）

### 目标

理解项目最重要的设计：不是对象字段，而是跨模块不变量。

### 阅读顺序

```text
ExecutionIdentity
→ TraceEvent
→ AgentEpisode
→ ExecutionBundle
→ EpisodeCertification
→ EligibilityDecision
→ DatasetManifest
```

对应文件：

- `src/contracts/execution_identity.py`
- `src/contracts/trace_event.py`
- `src/contracts/agent_episode.py`
- `src/contracts/execution_bundle.py`
- `src/contracts/episode_certification.py`
- `src/certification/engine.py`
- `src/contracts/dataset.py`

### 对每个对象填写契约卡

```text
对象：
它绑定/描述的边界：
创建者：
消费者：
关键不变量（至少 2 个）：
违反时如何失败：
它避免的真实错误：
```

### 必做推理题

不要搜答案，先自己回答：

1. 为什么 `ExecutionIdentity` 不能只使用 `task_id`？
2. 同一 `event_id` 的重复写入，什么时候可接受，什么时候必须 quarantine？
3. 为什么 certification 不放进 `ExecutionBundle`？
4. 为什么未知 canonical 顶层字段默认拒绝？
5. 为什么 DatasetManifest 按 logical task 切分，而不是按 trajectory 切分？

### 验收标准

能把任意一个“不变量”翻译成以下完整句式：

> 系统要求 ___，因为否则 ___ 会把 ___ 错误地当作 ___；所以在 ___ 阶段用 ___ 拒绝它。

---

## 第 3 关：沿一次调用链读代码（90 分钟）

### 目标

能从命令入口追到核心执行链路，而不是孤立地解释模块。

### 任务

以 `prepare-local` 和 `execute-local` 为入口，画出调用图。每一个节点只写四件事：输入、输出、不变量、失败路径。

阅读顺序：

1. `src/cli.py` 中 `prepare-local` / `execute-local` 的参数注册和处理函数；
2. `src/orchestration/local_execution.py`；
3. `src/launchers/local_docker.py`；
4. `src/capture/model_proxy.py` 与 `src/capture/harness_http.py`；
5. `src/assembly/local_run_finalizer.py`；
6. `src/assembly/execution_bundle_assembler.py`；
7. `src/certification/engine.py`。

### 产出

画出如下结构的“你的版本”，并给每条箭头标注传递的对象：

```text
CLI → prepare → run manifest / launch plan
    → proxy + isolated harness
    → events + model evidence + artifacts
    → finalizer → episode + bundle
    → certification → eligible / rejected / insufficient evidence
```

### 验收问题

1. 为什么 local execution 把 proxy、Harness、verifier、finalize、cleanup 放到一个 orchestrator transaction 中？
2. execution bearer token 解决的是什么问题？
3. 如果 Harness 被 Ctrl-C 中断，哪些证据仍应留下，哪些结论绝不能给出？
4. cleanup 为什么要 identity-bound，而不是按“最近启动的容器”清理？

---

## 第 4 关：用测试学习失败语义（90 分钟）

### 目标

通过失败案例掌握 fail-closed 的真实原因；这是本项目最容易被面试深挖的部分。

### 先运行精确测试

```bash
python3 -m unittest \
  tests.contracts.test_execution_identity \
  tests.contracts.test_trace_event \
  tests.contracts.test_execution_bundle \
  tests.assembly.test_execution_bundle_assembler \
  tests.validation.test_episode_certification \
  tests.certification.test_engine -v
```

如果本地测试模块路径有差异，使用：

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

### 挑六个 test，逐个写“它阻止什么事故”

至少覆盖：

- duplicate event / immutable conflict；
- sequence gap 或 orphan span；
- artifact 缺失；
- verifier 不可信；
- policy fingerprint 不匹配；
- training split leakage 或 preference pair 不完整。

### 验收问题

1. “失败轨迹可用于分析”与“失败轨迹可作为 SFT 正样本”为什么不冲突？
2. 外部 API 只返回文本时，系统能声明哪些能力，不能声明哪些能力？
3. 为什么不能把没有采集到 tool I/O 写作“工具调用数为 0”？

---

## 第 5 关：理解两个闭环，而非只理解数据管道（75 分钟）

### 目标

能区分 Harness 改进与模型改进实验，也能指出混杂变量。

### 阅读材料

- [数据契约](data-contract.md) 的 Experiment、capability、training 相关章节；
- [SFT/RL 数据流](sft-rl-data-flow.md)；
- `src/analysis/compare.py`、`src/analysis/regression_gate.py`；
- `src/training/eligibility.py`、`src/training/policy_fingerprint.py`；
- `src/learning/dataset_compiler.py`。

### 产出：双闭环对照表

| 问题 | Harness 闭环 | 模型闭环 |
|---|---|---|
| 唯一允许变化的变量 |  |  |
| 必须冻结的变量 |  |  |
| 核心比较单位 |  |  |
| 通过条件 |  |  |
| 常见混杂因素 |  |  |
| 错误时系统行为 |  |  |

### 面试题

> 如果 candidate 的 reward 更高，但 model revision、tool schema 或 task revision 也变了，你能否宣布 Harness 改进？为什么？

标准不是“不能”，而是要说清：系统会在哪个兼容性/manifest 边界给出 `INSUFFICIENT_EVIDENCE`，并拒绝伪精确的比较结论。

---

## 第 6 关：动手做一次小而完整的维护改动（2–4 小时）

### 目标

从阅读者变成能安全维护此仓库的人。

### 推荐改动（由易到难，选一个）

1. 为一个现有 certification 拒绝路径补充更有定位价值的错误上下文，并补单测；
2. 在 data-quality report 中新增一个来源于已有事实、不会推断缺失数据的统计项，并补单测；
3. 新增一个 consumer eligibility 条件，使其贯通 contract、certification、CLI 输出和测试。

### 改动前必须写下的设计说明

```text
需求：
影响的 canonical schema：有 / 无
影响 checksum：有 / 无，原因：
影响哪些消费者：
失败策略：reject / insufficient evidence / quarantine，原因：
测试层级：unit / integration / CLI：
兼容性策略：
```

### 验收标准

- 原有相关测试通过；
- 新测试能在移除改动后失败；
- 能解释为什么没有把逻辑塞进 CLI 或 dataclass 中；
- 能描述该改动对历史 artifact 的影响。

---

## 第 7 关：面试演练与漏洞清单（持续进行）

### 30 秒版本

> 这是一个 Agent 改进数据平面。它将真实 Harness 执行采集为不可变事件和内容寻址 artifact，组装成可验证 Episode，再通过 capability、lineage、verifier 和数据集规则进行 fail-closed certification。这样 Harness 对比和 SFT/Preference/RL 数据都能追溯到可信证据，而不是把普通日志直接当训练数据。

### 高频追问清单

1. 为什么 append-only event log 优于只保存最终 Episode？
2. Outcome、execution validity、integrity 为什么要拆开？
3. 你如何处理重复、乱序、缺事件和损坏 artifact？
4. `ExecutionIdentity`、checksum、manifest 各自解决什么不同问题？
5. 为什么普通 OpenAI response 不能成为 token-faithful RL trajectory？
6. capability 为什么是“观测权限”而非功能标签？
7. 如何避免 A/B 比较受到混杂变量影响？
8. 如何阻断 TRAIN/DEV/TEST 数据泄漏？
9. 单 GPU rollout/train 如何避免资源争用与错误恢复？
10. 本项目目前最重要的未完成项及风险是什么？

### 每次演练的评分标准

| 维度 | 0 分 | 1 分 | 2 分 |
|---|---|---|---|
| 系统边界 | 模糊 | 能说组件 | 能说 ownership 与非目标 |
| 数据流 | 背名词 | 能排序 | 能解释对象与转换原因 |
| 设计取舍 | 只说“更安全” | 有一个理由 | 有失败模式和替代方案比较 |
| 真实性 | 夸大 | 知道局限 | 主动给出已验证证据和未完成边界 |
| 代码落地 | 不能定位 | 能说目录 | 能定位函数、测试和影响面 |

目标是连续两轮每题至少 1 分，并对核心题达到 2 分。

---

## 你现在该做什么

从第 0 关开始，不要跳到源码。完成后把你的“90 秒项目介绍”和三个验收题答案发给 AI；AI 应只扮演严格面试官：指出错误、追问证据、让你补全，而不是把答案直接重写给你。

完成第 1 关时，保留你画出的执行证据表。它会成为后续所有源码阅读的导航图。

## 完成定义

你完成本手册并不代表“记住所有文件”；而代表你能：

- 独立画出一条真实执行到训练/分析结论的链路；
- 明确任何结论的证据、前提与失败策略；
- 解释六个以上关键设计取舍；
- 在限定范围内完成一次安全改动和验证；
- 不夸大项目当前结果，并能把未完成项说成清楚的工程风险与下一步验证计划。
