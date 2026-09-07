# v3 — 15-task 套件 · Track A + Track C（已归档）

> 版本：v3 · 状态：已执行 · 日期：2026-08-19 · 服务器：新疆 RTX 4090
> 目标：在受控的 15-task 套件上同时拉动**真实 harness 迭代**（Track A）和
> **本地模型训练闭环**（Track C，Qwen-7B SFT + GRPO）。

---

## 1. 结果总览（关键数字）

```text
[Track A · 真实 harness 迭代]  真实 Pi × 15-task × candidate/v3
  candidate : 15/15 ✅（candidate 策略在 toy suite 已满分）
  v3        : 14/15 ⚠️（加"立即作答+忽略干扰状态"约束后 task-3 翻车）
  Gate      : INSUFFICIENT_EVIDENCE（v3 成功率回落，Gate 拒绝）— 迭代机制正确工作

[Track C · 7B 模型]             Qwen2.5-7B-Instruct（LoRA，GPU）
  零样本   : 0/15（基线）
  SFT      : 11/15（teacher 蒸馏主线）  ← 训练 21.8s，train_loss 0.050
  GRPO-lite: 11/15（与 SFT 持平，无崩坏）← 曾疑崩坏，查实为评估装配 bug，已修
```

15-task 拆解（SFT 7B）：
```text
train 1-8 : 全 8 ✅（学会 teacher 工具协议）
dev  10   : ✅,  11: 0.5
eval 13   : ✅（6/15 计数成功，泛化强）
eval 12/14/15 : 0.5（恢复+读文件正确，但干扰状态计数错）
reward=0.5 = 模型已正确执行工具协议，卡在细节计数 → GRPO 应在此发力的任务
```

---

## 2. 数据集与复现材料

```text
15-task 套件定义 : src/task_suite.py（train 1-8 / dev 9-11 / eval 12-15）
真实 Pi 捕获     : tests/fixtures/pi/v3-15task/{candidate,v3}-{task}.ndjson（30 条）
SFT 训练数据     : experiments/local_model/sft-dataset-v3.jsonl（8 条 verified 候选轨迹）
runner          : src/real_pi_v3.py（--replay 支持离线重放，不再调 API）
训练脚本        : experiments/local_model/{sft_train,grpo_lite}.py
```

---

## 3. Track A：真实 harness 迭代（数据 → 策略 → Gate）

```text
candidate 策略：读失败 → find task-*.json → 读唯一文件（V2.1 的赢家，作新基线）
v3 策略       ：candidate + "读完立即作答，只数 FAILURE，忽略其他状态"
结果对比：
  candidate  15/15
  v3         14/15（task-3 翻车：v3 的"立即作答"指令副作用）
  Gate       INSUFFICIENT_EVIDENCE（成功率未提升反而回落，Gate 保留基线）

意义：策略迭代循环闭环 —— 提出改进 → 真实采样 → Gate 判定。
      Gate 诚实地拒绝了"加了约束反而更差"的候选，这正是评估把关的价值。
```

**与 SWE-bench 的关系**：candidate 策略正是 SWE-bench 上 7/10 的同一套 agent，
toy 套件上深seak 已满分（15/15），所以 v3 的"额外约束"在这里是**过度设计**——
这本身是有价值的负面证据（harness 复杂度不是越高越好）。

---

## 4. Track C：Qwen-7B SFT 成功；GRPO-lite 不稳定（诚实记录）

### 4.1 SFT（教师蒸馏）— 干净成功
```text
数据  : 8 条 candidate 验证通过轨迹（teacher = deepseek）
训练  : LoRA SFT，21.8s，train_loss 0.050（收敛良好）
结果  : 0/15 → 11/15
结论  : teacher 轨迹蒸馏在 7B 上工作良好，协议学习 + 泛化（eval 13 ✅）都达成
```

### 4.2 GRPO-lite（on-policy）— 诚实结论：不崩、与 SFT 持平
```text
真 bug：早期手动评估把 GRPO adapter 加载到裸 base（丢了 SFT-merge）→ 伪崩坏 2/15。
       实际是评估装配错误，不是训练崩了。K=0（零训练）adapter 正确评估=11/15 证实。
修正  ：SFT-merged base + GRPO LoRA 评估。
真实结果（多个配置：full / answer-only / 短 / 长训 10 步）：
   全部 = 11/15，与 SFT 持平；无崩坏。
   长训 answer-only 把 dev task-9 推到 1.0，但 task-11/12/14 仍 0.5
   （干扰状态计数是模型真实能力瓶颈，短 GRPO 教不会）
诚实结论：GRPO-lite 在此任务上未稳健超越 SFT（持平）；
          “GRPO > SFT”需一个“SFT 必卡死、GRPO 能过”的更硬任务 + 更充分训练，
          当前未达成，如实记录。

数据面/RL 严谨性教训（加分项）：
   评估底座（merge_and_unload 的 SFT base）必须与训练一致，否则 RL 结果
   被评估装配 bug 污染成假崩坏 —— “评估自身不可信”正是我们数据面要防的东西。
```

---

## 5. 诚实边界（面试表述）

```text
- SFT 是干净的主线：teacher 数据 → 本地模型 → 0→11/15，可复现
- GRPO-lite 不稳定 = 真实 RL 工程难点，我们保存了可复现的崩坏证据与分析
  而不是挑一个"碰巧更好"的种子掩盖它
- toy 套件上 harness 迭代的 Gate 拒绝（v3 回落）是诚实负面结果
- on-policy 的真实价值（超越 SFT）需要稳定实现，当前未达成，明确标注
```

---

## 6. 与 SWE-bench / 全局故事的连接

```text
真实 benchmark（SWE-bench 7/10）
        + 受控套件（15-task，candidate 15/15）
                ↓ 同一数据面
有效数据生产：SWE-bench 7 条 + v3 15-task 8 条 verified teacher 轨迹
模型改进：   Qwen-7B SFT 0/15 → 11/15（蒸馏主线成立）
RL 稳定性：   GRPO-lite 崩坏 = 已诊断为 coarse-reward + 逐 token credit +
             共享 LoRA + greedy 脆弱性 —— 开放的工程挑战，如实记录
```
