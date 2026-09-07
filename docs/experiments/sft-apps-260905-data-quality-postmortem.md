# APPS SFT 数据质量事故复盘 — 2026-09-05/06

## Scope

`apps-qwen14b-eligible-v1-260905`（5242 样本 / 799 tasks）LoRA SFT（3 epochs，loss 0.65→0.51）
训练完成后，在 HumanEval / MBPP / BigCodeBench-Complete 上按 Qwen2.5-Coder report 协议
（greedy temp=0，EvalPlus + official bigcodebench harness）做了 base/candidate 对照。
评估过程中暴露出 **三个训练数据质量问题** 和 **两个历史评估基线缺陷**。
本文记录问题、证据、影响与修复方向，供后续实验避免重蹈覆辙。

## Results（先说结论，评估本身有效但引出数据问题）

| Benchmark | 口径 | base | candidate | Δ |
|---|---|---:|---:|---:|
| HumanEval | pass@1 (base tests) | 0.598 | 0.646 | +4.8 |
| HumanEval+ | pass@1 (extra tests) | 0.567 | 0.628 | +6.1 |
| MBPP | pass@1 (base tests) | 0.836 | 0.828 | −0.8 |
| MBPP+ | pass@1 (extra tests) | 0.722 | 0.720 | −0.2 |
| BigCodeBench Complete | pass@1 (1140, sanitized calibrated) | 0.564 (643/1140) | 0.543 (619/1140) | −2.1 |

解读：简单/中等纯代码补全有正迁移，最难、需要读长指令的任务回退。
与"蒸馏 SFT 收益有限、真正提升要靠 on-policy RL"的社区经验一致。
但抽查训练数据后发现 candidate 本身带着三个严重数据缺陷（见下），
−2.1 的回退与缺陷 2 直接相关，评估结论只对"这份数据"负责，不代表 agent SFT 本身无效。

## 问题一：completion 中的 thinking 字段是字面量 `[REDACTED]`

### 证据

`examples.jsonl` 抽样（样本 1）：

```json
completion_text: [{"thinking":"[REDACTED]","thinkingSignature":"[REDACTED]",
  "type":"thinking"},{"arguments":{...},"id":"call_83b948d40ddd48eca8bc3251",
  "name":"ls","type":"toolCall"},...]
```

### 根因

teacher 轨迹入库时（隐私/日志 redaction 层）把 thinking 内容替换为占位符，
`build_apps_sft_package.py` 从 episode 的 `MODEL_RESPONSE.attributes["content"]`
直接取全文做 token 化，没有过滤或重建 thinking 块。
于是每个 completion 的 thinking 位置都在教模型输出字面量 `[REDACTED]`。

### 影响

- 模型被系统性训练去生成红色占位符文本；
- 推理链（teacher 的解题思路）完全没有教到，而这恰恰是 agent 轨迹里最值钱的部分；
- thinking token 的 loss 还稀释了 tool-call/action 部分的有效梯度。

### 修复方向

- 重建 package 时二选一：**剔除** thinking 块（只训 action），或用 teacher 的
  原始未脱敏 thinking 重导（需要回到 teacher 源数据，确认是否保存了原文）；
- `build_apps_sft_package.py` 增加 fail-closed 校验：completion 里出现
  `[REDACTED]` 字面量即拒绝该样本并计数上报。

## 问题二：tool 参数固化了宿主机绝对路径

### 证据

```json
{"arguments":{"path":"/home/f630/homePLUS/agent-data-plane/apps-workspaces/sft-teacher-v4/apps-train-000001"},"name":"ls",...}
{"arguments":{"command":"cd /home/f630/homePLUS/agent-data-plane/apps-workspaces/sft-teacher-v4/apps-train-000001 && ls -la"},"name":"bash"}
```

### 根因

teacher（DeepSeek）在真实 Pi harness 工作区里跑，工具调用参数里天然带绝对路径。
构建 package 时未做路径归一化，学生的训练目标里出现了与任务本身无关的
宿主机目录结构（`sft-teacher-v4/apps-train-000001` 等每题不同、每轮不同）。

### 影响

- 模型学到"先 ls 一个固定模式的绝对路径"这类与环境绑定而非与任务绑定的动作；
- 评估环境（BigCodeBench 容器 `/workspace`、HumanEval 工作区）路径分布完全不同，
  学到的动作序列失效——这是 BigCodeBench −2.1 最可疑的来源；
- 更隐蔽的危害：模型把"记住的路径"当成了任务记忆的一部分，奖励了死记而非推理。

### 修复方向

- 构建时把 tool 参数里的绝对路径归一化为相对 workspace 路径
  （`cd <workspace> && ls -la` → `ls -la`；`/abs/.../apps-train-000001` → `.`）；
- 校验器：completion 中出现工作区绝对路径前缀即拒绝样本。

## 问题三：逐 turn 独立样本，学生从未学过走完整条 episode

### 证据

每个样本是"某一步的上下文 → 那一步的响应"；前 200 样本里 prompt 含 0–12 个
assistant 历史 turn。历史全部来自 teacher 自己走出来的轨迹。

### 根因

`build_apps_sft_package.py` 按 MODEL_REQUEST/RESPONSE 对逐条取样本，
这是标准的 teacher-forcing 行为克隆，但学生训练时永远站在 teacher 的状态分布上。

### 影响

- 经典的 distribution shift：学生推理时一旦偏离 teacher 轨迹（走错一步），
  后续状态是训练时没见过的，错误会累积（error compounding）；
- 这解释了为什么 SFT 后仍然需要 on-policy RL（GRPO）来校正学生自己的状态分布。

### 修复方向（按投入排序）

- 短期不改数据：接受 SFT 只是初始化，把期望放到下一阶段 RL；
- 中期：对同一 episode 的相邻 turn 构造"学生前缀 + teacher 动作"的混合样本
  （DAgger 思想），训练时逐步掺入学生自己的前缀状态。

## 附带发现的两个历史评估基线缺陷（同样是宝贵教训）

### 缺陷 A：BigCodeBench 标准基线被 token 截断毁掉（260901）

- 旧 base 成绩 24/1140（2.1%），大量 `NameError: name 'task_func' is not defined`；
- 抽查发现所有 completion 截断在 ~1000 字符（1140 中仅 153 个含 `return`），
  函数体没写完 → serving/评测层 token 上限过小；
- **教训**：低得离谱的分数（官方 ~51% vs 本地 2%）不是"模型差"，几乎一定是
  执行链路缺陷。看到异常分数先查截断/解析/环境，再谈能力。
- 修复后 base 实测 **643/1140 = 56.4%**，与官方量级一致，基线才有效。

### 缺陷 B：旧 canonical-v2 的 BigCodeBench candidate 轮是坏 adapter

- 37 题平均 18 秒/题完成，模型输出退化（答案字段为 None），
  `verifier_status: FAILED` 几乎全部；
- 根因指向当时 adapter 服务异常（输出退化），该轮 candidate 结果不可用；
- **教训**：candidate 轮跑得"特别快"本身就是红灯——agent 式评测的正常耗时有下限，
  远低于下限 = 模型没有真正干活。

## 加速经验（顺手记录）

- BigCodeBench 生成：官方 openai backend 串行 3h → vLLM in-process backend
  （`bigcodebench.generate --backend vllm`）**7 分 52 秒**，约 23×；
- LoRA candidate 用 `merge_and_unload` 合并进 base 权重后与 base 用同一引擎评测，
  消除引擎差异且数学上精确等价；
- 数据集下载失败用 `BIGCODEBENCH_OVERRIDE_PATH=/home/f630/.cache/bigcodebench/BigCodeBench-v0.1.1.jsonl`
  直接指向本地缓存；`--split` 参数指 prompt 类型（complete/instruct），不是数据集版本；
- EvalPlus base 成绩可跨实验复用（frozen baseline，协议一致即可）。

## Follow-up：数据修复后重训结果（2026-09-07）

重建 package（thinking 恢复 3244 turns、路径归一化 3605 处、fail-closed 拒收 255）后重训
（clean-v2，0.878→0.764→0.690，16.9h），对照 base（与脏数据对比不计入结论）：

| Benchmark | base | clean-v2 candidate | Δ vs base |
|---|---:|---:|---:|
| HumanEval | 0.598 | 0.646 | +4.8 |
| HumanEval+ | 0.567 | 0.616 | +4.9 |
| MBPP | 0.836 | 0.847 | +1.1 |
| MBPP+ | 0.722 | 0.720 | −0.2 |
| BigCodeBench Complete | 0.564 | 0.553 | −1.3 |

结论：数据修复后，SFT 在 HumanEval 保持 +5 左右、MBPP 转正、BigCodeBench 回退收敛到 −1.3。
蒸馏 SFT 的天花板依旧存在；剩余缺口首推训练序列化格式与 serving chat-template 不对齐
（复盘标记的第 4 项），其余预期由 on-policy RL 收尾（分布漂移问题三的设计归期）。

## 下一步（SFT 后进入 RL 的起点选择）

决策：base（instruct 模型）vs clean-v2 SFT candidate 谁作为 RL 起点，
依据是两组 benchmark 对照与 RL 冷启动稳定性，选择记录见后续文档。

## Artifacts

- 训练 package（含缺陷）: `sft-packages/apps-qwen14b-eligible-v1-260905`
- 本次 candidate adapter: `sft-runs/qwen14b-apps-v1-260905`（+ `-merged/`）
- EvalPlus: `standard-evalplus-260901/`（base 复用）、`standard-evalplus-apps-v1-260905/`
- BigCodeBench: `standard-bigcodebench-apps-v1-260906/`（bcb_results、verifier-base、verifier-candidate）
- 训练日志: `sft-runs/apps-qwen14b-train-v2-260905.log`
- 脚本: `scripts/train_apps_lora_sft_v2.py`（分桶 batching + SDPA + 分块 CE）、
  `scripts/run_std_bcb_260906.sh`、`scripts/eval_std_bcb_260906.sh`
