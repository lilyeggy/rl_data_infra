# 结合本项目理解 Agent SFT

这份文档的目标不是让你背 SFT 名词，而是让你能够回答：

> 我们的 Agent 轨迹是如何变成训练样本的？模型到底在学习什么？为什么要这样筛选数据？训练后如何证明有效？

阅读时建议按照顺序完成每一章。每章最后都有一个“你应该能回答”的问题。

---

## 0. 先记住项目主线

我们的项目不是单纯收集代码答案，而是让 Agent 在工具环境中完成任务：

```text
代码任务
  → Agent + Harness 执行
  → 读取/修改 workspace
  → 运行测试
  → verifier 验证
  → 保存 trace 和 evidence
  → 筛选可信轨迹
  → 转换成 SFT 数据
  → LoRA 训练
  → 固定 holdout 评测
```

最重要的理解是：

> SFT 本质上是模仿 teacher 的监督学习。在本项目中，teacher 的“答案”不是只有最终代码，而是完整的 Agent 行为序列，包括工具调用、代码修改和测试操作。

例如模型要学会：

- 先查看工作区
- 读取已有文件
- 使用 `write` 或 `edit` 修改代码
- 运行测试
- 根据错误继续修改
- 最终完成任务

---

## 1. 第一阶段：看懂一条原始轨迹

本项目的主要轨迹文件包含以下信息：

```json
{
  "task_id": "mbpp-602",
  "messages": [...],
  "steps": [...],
  "final_answer": "[verifier] benchmark tests passed",
  "verifier_status": "PASSED",
  "certification_verdict": "ELIGIBLE",
  "execution_bundle_checksum": "..."
}
```

其中：

- `messages`：模型看到的消息和模型发出的 tool call
- `steps`：tool call 与 tool result 的结构化对应关系
- `verifier_status`：代码测试是否通过
- `certification_verdict`：轨迹是否允许进入 SFT
- `checksum`：保证数据资产可追溯、未被静默修改

一条轨迹可能是：

```text
user: 请实现一个 Python 函数
assistant: 调用 ls
tool: 返回 solution.py
assistant: 调用 read
tool: 返回模板内容
assistant: 调用 write
tool: 写入实现
assistant: 调用 bash
tool: 测试通过
```

### 练习

从 A6000 上的 SFT 数据中查看一条样本：

```bash
python3 -c 'import json; p="mbpp-sft-export-260827-v5/examples-nosystem.jsonl"; print(json.dumps(json.loads(open(p).readline()), ensure_ascii=False, indent=2))'
```

如果文件在远端，需要先通过 SSH 查看，或者只查看脱敏后的样本。重点观察：

1. 哪些内容是 user 输入？
2. 哪些内容是 assistant 的训练目标？
3. tool result 是不是模型生成的？
4. verifier 结果在 messages 里，还是在元数据里？

### 你应该能回答

为什么不能只保存最终的 `solution.py`？

因为那会丢失 Agent 如何观察环境、如何选择工具、如何根据测试反馈修正代码的行为信息。

---

## 2. 第二阶段：理解什么叫“可信轨迹”

不是模型有输出，就可以拿来训练。当前项目会区分：

```text
ELIGIBLE
REJECTED
INSUFFICIENT_EVIDENCE
```

一条 SFT 轨迹至少需要满足：

```text
任务身份明确
+ trace 完整
+ tool call 可解析
+ tool result 有对应关系
+ workspace 有实际修改
+ verifier 真实执行
+ verifier 通过
+ 没有 timeout / serving error
+ 数据格式和长度合法
```

代码中可以重点阅读：

- `src/certification/engine.py`
- `src/validation/episode_semantics.py`
- `scripts/export_teacher_sft.py`
- `scripts/build_sft_training_package.py`

### 为什么 verifier 通过还不够？

因为下面这些情况可能出现：

- 测试通过，但 Agent 的 tool call 没被完整记录
- 模型响应被 serving 截断
- verifier 没有真正执行，只是模型自己说“通过了”
- tool call 和 tool result 无法对应
- 轨迹来自不同任务或不同 workspace
- 数据超过训练长度限制

因此可信数据同时要求：

```text
结果正确 + 执行过程可证明 + 数据格式可训练
```

### 练习

分别解释下面三种结果：

```text
VALID + PASSED
VALID + FAILED
INFRA_INVALID + ERROR
```

正确理解是：

- `VALID + PASSED`：模型正常执行并完成任务
- `VALID + FAILED`：模型正常执行，但代码没通过测试
- `INFRA_INVALID + ERROR`：执行链路有问题，不能直接归因于模型

---

## 3. 第三阶段：理解原始轨迹如何变成 SFT 样本

入口脚本是：

```text
scripts/build_sft_training_package.py
```

它不会把整条轨迹作为一个巨大样本，而是把轨迹转换成多个 turn-level 样本。

一条轨迹：

```text
user
→ assistant: ls
→ tool: ls result
→ assistant: read
→ tool: read result
→ assistant: write
→ tool: write result
→ assistant: bash
→ tool: test result
```

会产生多个训练目标：

```text
训练样本 1：学习输出 ls tool call
训练样本 2：学习输出 read tool call
训练样本 3：学习输出 write tool call
训练样本 4：学习输出 bash tool call
```

本次实验是：

```text
289 条完整轨迹
→ 1163 条 training turns
```

每个 turn 的结构大致是：

```json
{
  "turn_id": "episode-mbpp-602-1:3",
  "task_id": "mbpp-602",
  "messages": [
    "system",
    "user",
    "历史 assistant/tool 消息",
    "当前 assistant 目标"
  ],
  "tools": [...],
  "target_message_index": 4
}
```

`target_message_index` 表示这一条样本中，哪一个 assistant 消息是训练目标。

### 为什么不直接训练整条轨迹？

拆成 turn-level 样本有几个好处：

- 每个 assistant 行为都能成为监督信号
- 不同长度轨迹可以共享 prefix/context
- 更适合 causal language model 的 next-token prediction
- 可以对重复 tool call、错误步骤进行质量控制

### 你应该能回答

为什么 289 条轨迹最后会变成 1163 条训练样本？

因为每条 Agent 轨迹包含多个 assistant action，每一个可训练 action 都被转换成一个 target turn。

---

## 4. 第四阶段：理解 tokenization 和 loss mask

入口脚本是：

```text
scripts/train_qwen_lora_sft.py
```

核心函数是 `_render()`。

它会构造：

```text
prompt = system + user + 历史消息 + generation prompt
full   = prompt + 当前 assistant 目标
```

然后生成 labels：

```python
labels = [-100] * len(prompt) + full[len(prompt):]
```

在 Transformers 中，label 为 `-100` 的位置不参与 loss。

因此训练目标是：

```text
不训练 system/user/tool result
只训练当前 assistant 输出（可能是 tool call，也可能是代码/文本）
```

例如：

```text
输入上下文：
请修复 solution.py
read 返回当前文件内容

训练目标：
assistant 调用 write，并给出正确参数
```

模型学习的是 teacher 如何行动；tool result 只是模型下一步决策的上下文，不是需要模仿的目标。

### causal language modeling 的直觉

模型依次预测目标序列中的每个 token：

```text
P(token_1 | context)
P(token_2 | context, token_1)
P(token_3 | context, token_1, token_2)
```

训练 loss 通常是目标 token 的平均负对数似然：

```text
loss = - mean(log P(target_token | previous_tokens))
```

这也是为什么 tool call 的 JSON 格式必须稳定：模型实际上会学习工具名、字段名、括号和参数结构。

### 长度处理

当前 package 使用 `max_length=4096`。如果一条 turn 太长，代码会从历史 middle context 中删除最早的消息，直到序列能放进长度上限。

这意味着：

- 当前目标 assistant 消息必须保留
- 最近的工具结果优先保留
- 太早的历史上下文可能被压缩

### 练习

阅读 `_render()`，回答：

1. 为什么 `prompt` 必须是 `full` 的前缀？
2. 为什么 tool result 不应该作为当前 target？
3. 如果当前 assistant target 被截断，会发生什么？

---

## 5. 第五阶段：理解 LoRA 训练

当前不是全量微调 14B 模型，而是：

```text
Qwen2.5-Coder-14B base model
+ LoRA adapter
```

LoRA 的基本思想是冻结原模型参数，只学习低秩更新：

```text
W' = W + ΔW
ΔW = B × A
```

其中 `A` 和 `B` 是较小的可训练矩阵。

好处：

- 显存和存储成本低
- 训练速度较快
- 保留原模型通用能力
- 可以为不同任务保存不同 adapter

当前训练大致是：

```text
base model：Qwen2.5-Coder-14B
training examples：1163 turns
epochs：2
precision：bf16
gradient checkpointing：开启
device：A6000 48GB
```

训练输出不是一个完整的新模型，而是 adapter 文件和训练元数据：

```text
sft-runs/qwen14b-mbpp-train-260827/
```

部署时：

```text
base model + adapter = candidate model
```

### 为什么把 `lm_head` 也纳入 LoRA？

本项目要学习 Qwen 原生 tool-call 输出格式。输出层对工具 token、JSON token 和终止格式很重要，因此当前 package 对 output head 有额外策略约束。

### 你应该能回答

为什么不用全量训练？

因为当前目标是验证 Agent 数据闭环和行为适配，LoRA 可以在较低资源下快速迭代，同时保留 base model 的代码能力。

---

## 6. 第六阶段：训练前必须做什么检查？

训练前不是直接启动 GPU，而是先跑 preflight。

当前检查包括：

```text
example_count
min/max/mean token length
trainable token count
tokenizer size
model vocab size
maximum token id
是否需要 resize embedding
```

典型命令：

```bash
python3 scripts/train_qwen_lora_sft.py \
  --package <training-package> \
  --preflight-only
```

训练前还需要确认：

- package checksum 与 dataset checksum 一致
- tool contract checksum 一致
- train/development/test split 没有泄漏
- 每个 target 都有可训练 token
- batch size 和 gradient accumulation 能完整覆盖数据
- 数据不会静默丢弃 epoch 尾部

这是 Agent Infra 项目里很重要的工程点：

> 训练失败不一定是模型问题，也可能是数据 package、tokenizer、batch collator 或配置不一致。

---

## 7. 第七阶段：如何评估训练是否有效

训练后不能只看 train loss。

我们使用没有进入训练的数据集做 holdout：

```text
同一批任务
→ base model 执行
→ candidate model 执行
→ 相同 harness / verifier
→ 比较有效通过率
```

关键指标要分开：

### 模型能力指标

```text
VALID + verifier PASSED
```

表示模型正常执行并生成了通过测试的代码。

### 数据管线指标

```text
trace 完整率
verifier 可执行率
evidence 完整率
ELIGIBLE 比例
```

### serving 指标

```text
invalid JSON
timeout
请求失败率
延迟
显存
吞吐
```

如果出现 `INFRA_INVALID`，不能直接当作模型失败。应该重放或单独统计。

本项目已经遇到过两个典型问题：

1. 200-token 上限导致 tool-call 和代码被截断
2. 非有限 logprob 导致 serving 返回非法 JSON

这就是为什么要把模型评测和管线健康度分开。

---

## 8. 第八阶段：理解本项目当前实验结论

修复 serving 问题并重放异常任务后，当前结果是：

```text
MBPP：
base 69/90
candidate 77/90

HumanEval：
修正后的 base 约 140/162
candidate 133/161
```

当前可以得出：

- 289 条 MBPP 轨迹对 MBPP 有明显收益
- 对 HumanEval 没有稳定泛化收益
- 训练数据可能偏窄，存在 MBPP 分布适应或过拟合
- serving 和 evidence 问题会严重污染 benchmark 结论

因此下一轮不应只是盲目增加数据，而应：

```text
扩大数据量
+ 增加任务类型多样性
+ 保持 verifier 筛选
+ 比较 1 epoch / 2 epochs
+ 保持固定 holdout
+ 单独报告模型失败和 infra 失败
```

---

## 9. 面试时如何用 90 秒讲清楚

可以这样回答：

> 我们做的是一个面向 coding agent 的 trajectory-to-training data pipeline。首先让 Agent 在隔离 workspace 中通过真实 harness 调用文件和 shell 工具完成 MBPP 任务，并记录每轮 model response、tool call、tool result、workspace diff 和 verifier 结果。只有 trace 完整、代码真实通过 verifier、没有 timeout 或 infrastructure error 的轨迹，才会进入 SFT 数据集。之后我们把一条多轮轨迹拆成多个 assistant turn，使用 chat template 做 tokenization，只对当前 assistant action 计算 causal LM loss，system、user 和 tool result 通过 `-100` mask 排除。训练侧采用 Qwen2.5-Coder-14B 的 LoRA，训练后在固定 holdout 上用相同 harness 比较 base 和 candidate。这样我们不仅能测模型是否完成任务，还能区分模型失败和 serving、harness、verifier 造成的基础设施失败。

这段话覆盖了：

```text
数据从哪里来
如何验证可信
如何变成训练样本
loss 如何计算
训练用什么方法
如何评估结果
```

---

## 10. 面试高频追问

### Q1：为什么 verifier 通过的数据还需要 trace？

因为最终结果正确不代表过程可信。trace 用来确认 tool call 真实执行、workspace 真实变化、verifier 真实运行，并支持回溯。

### Q2：我们的 SFT 为什么不只训练最终代码？

SFT 仍然会训练最终代码，但当前数据还把 tool call、代码修改和测试操作作为 assistant 目标。因此它模仿的是完整 Agent 行为，而不只是最终代码补全。

### Q3：SFT 数据能直接用于 RL 吗？

不一定。SFT 需要 messages 和正确 target；RL 还可能需要完整 action/observation、reward、mask、token ids 和 behavior logprob。应该为 SFT 和 RL 做不同的 admission policy。

### Q4：模型通过率下降一定是模型问题吗？

不一定。需要先排除 serving JSON 错误、timeout、tool protocol 错误、verifier 未执行和 evidence 不完整。

### Q5：为什么要固定 holdout？

如果每轮都换测试集，就无法判断模型提升来自训练，还是来自测试集变化。固定 holdout 才能做公平对照。

### Q6：如何判断数据量是否真的有用？

做受控实验：固定模型、prompt、harness、verifier 和 holdout，只改变训练数据量、数据组成或 epoch，然后比较有效通过率和失败类型。

---

## 11. 建议的学习顺序

按下面顺序读源码：

```text
1. scripts/run_mbpp.py
2. src/orchestration/pi_host_execution.py
3. src/capture/pi_adapter.py
4. scripts/verify_mbpp.py
5. scripts/export_teacher_sft.py
6. scripts/build_sft_training_package.py
7. scripts/train_qwen_lora_sft.py
8. docs/data-contract.md
9. docs/sft-rl-data-flow.md
10. docs/polar-integration.md
```

每读一个文件，都问自己三个问题：

```text
输入是什么？
输出是什么？
失败时如何处理？
```

最后你应该能够手动画出：

```text
一条 MBPP 任务
→ 一条 Agent episode
→ 一个 certified example
→ 多个 turn examples
→ token ids + labels
→ LoRA gradient update
→ holdout verifier result
```

这张图能画清楚，基本就真正理解了我们当前的 SFT 系统。
