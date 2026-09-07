# v4 — 真实 Harness + 本地模型原型（已归档，不代表当前 RL 路线）

> 版本：v4 · 状态：已执行 · 日期：2026-08-19 · 服务器：新疆 RTX 4090（24GB）
> 目标：纠正 v3 的架构味道——**不再有 micro-harness**。让真实 Pi 通过自定义
> provider 直接驱动本地 Qwen-7B，训练/评估/教师走同一个真实 harness，并在这条
> 真实链路上做 **on-policy GRPO**，证出 **GRPO > SFT**。

---

## 1. 结果总览（关键数字）

```text
学生模型：Qwen2.5-7B-Instruct + LoRA，全部由真实 Pi 驱动（local-qwen provider）

P2  SFT 学生 · 真实 Pi · 贪心        : 7/15   {1,2,3,4,5,8,9}
P3  GRPO on-policy · 真实 Pi          : 10/15  {1,2,3,4,5,6,7,8,9,12}   ✅ GRPO > SFT

净增任务 6,7,12 —— 全部是探针标记的"前沿任务"（模型有方差、偶尔能数对）
死区任务 10,11,14 原地不动（从没数对 → 无正信号 → 诚实不涨）
mean_reward 轨迹（收集期）: 0.639 → 0.731 → … 单调上升，on-policy 学习成立
```

**对比 v3（micro-harness）**：v3 里 GRPO 与 SFT 持平（11/15）无法超越；v4 换到真实
harness 后，SFT 基线降为 7/15（暴露真实能力缺口），GRPO 有了干净的提升空间并兑现
（7→10/15）。**真实 harness 让 RL 的价值显现出来。**

---

## 2. 架构纠正：为什么删掉 micro-harness

v3 的味道：教师在真实 Pi 产轨迹，学生却在另一个 Python micro-harness 里训练/评估
→ 训练分布 ≠ 评估分布 ≠ 教师分布，且 micro-harness 的工具语义与真实 Pi 不一致。

**实锤**：micro-harness 的 `find` 是 Python glob（能用）；真实 Pi 的 `find` 底层是
`fd` 二进制（本服务器不可用 → find 永远报错）。学生在 micro-harness 学到"find 能用"，
到真实环境就卡死。这正是"不该有 micro-harness"的直接证据。

**v4 正确架构**：

```text
教师(deepseek) ──真实 Pi──► 黄金轨迹 ──┐
                                      ├─► 数据面 (canonical episode + verifier)
学生(Qwen-7B) ──本地服务──真实 Pi ────┘        │
   ▲                                            │
   └── GRPO: 重算 logprob + 更新 LoRA ──────────┘
       └── /reload_adapter / 重启 权重热同步 ──► 服务
```

接入方式（Pi 原生支持，见 pi 文档 `docs/models.md`）：在 `~/.pi/agent/models.json`
注册自定义 provider 指向本地 OpenAI 兼容服务：

```json
{ "providers": { "local-qwen": {
    "baseUrl": "http://127.0.0.1:8000/v1", "api": "openai-completions", "apiKey": "local",
    "compat": { "supportsDeveloperRole": false, "supportsReasoningEffort": false },
    "models": [ { "id": "qwen2.5-7b-instruct", "reasoning": false,
                  "contextWindow": 32768, "maxTokens": 4096 } ] } } }
```

---

## 3. 新增组件（都在 `experiments/local_model/`）

| 文件 | 作用 |
|---|---|
| `openai_server.py` | transformers 版 OpenAI 兼容服务：加载 7B+LoRA，把 Qwen 的 `<tool_call>` 解析成 OpenAI `tool_calls` 供 Pi 执行真实工具；记录每次生成的 `prompt_ids`/`completion_ids` 到 rollout 日志；`/reload_adapter`、`/set_sampling` 支持运行时切换 |
| `pi_local.py` | 让本地模型经真实 Pi 跑 15 任务：搭 workspace、生成环境准确的 prompt、调 `run_pi_process(provider=local-qwen)`、严格 verifier 打分 + 0/0.5/1 reward 整形 |
| `grpo_pi.py` | on-policy GRPO：服务采样收集 rollout（按 token 偏移读日志）→ 停服务 → 训练端重算 old/new logprob 做 PPO-clip + k3-KL 更新 LoRA → 存 adapter → 服务带新 adapter 重启。外加 `--probe` 采样探针 |

**为什么推理/训练交替**：单张 24GB 卡放不下"推理服务(~15GB) + 训练进程(~20GB)"
两份 7B。所以循环是：服务起(采样收集) → 服务停(等显存真正释放) → 训练 → 存 adapter
→ 服务带新 adapter 重启。这是 agentic-RL 栈标准的"推理/训练解耦 + 权重同步"。

---

## 4. P2：SFT 学生在真实 harness 的基线 = 7/15

学生 prompt 是**环境准确**的（"列目录发现 task 文件"，而非写死用已损坏的 find）。
学生的真实行为轨迹（task-1）：

```text
read missing-1.json → ENOENT（预期，文件本不存在）
find task-*.json    → fd 不可用（失败）
ls                  → task-1.json  ✅ 恢复（学会了 find 失败改用 ls，和教师一样）
read task-1.json    → 成功
最终                → {"task_id":"task-1","status":"SUCCESS","failure_count":2} 纯 JSON 正确
```

- **协议层全对**：ls 恢复 + 纯 JSON 输出过严格精确匹配 verifier。
- **瓶颈 = 带干扰项的计数**（6,7,10-15 失败全是数错，不是协议错）。

---

## 5. 关键探针：任务的"可学习前沿"（采样 6 次/任务）

```text
task  full/6  性质
2     6/6     已解决（无方差 → 无信号）
1,5   5/6     基本解决
3,4,9 2-3/6   ▶ 前沿（有方差，GRPO 可放大）
6,7,8,12,13,15 1-2/6 ▶ 前沿（偶尔数对 → 有正信号）
10,11,14 0/6  ✗ 死区（从没数对 → 无正信号，GRPO 学不了）
```

**这直接解释了 v3 GRPO 为何无法超越 SFT**：当时把死区任务也丢进去 + 组太小 → 信号
稀疏 → 训练在近死区 batch 上反而把协议学崩。正确做法 = 只对前沿任务、用足够大的组、
加 KL 防回归。

---

## 6. P3：真实 harness 下的 on-policy GRPO = 10/15

配置：前沿 9 任务(3,4,6,7,8,9,12,13,15) × 组6 × 3 轮，lr=1e-5, clip=0.2, beta=0.02(KL)。

```text
OUTER 0 (SFT adapter)   collect mean_reward=0.639 → train
OUTER 1 (grpo-o0)       collect mean_reward=0.731 → train     ← 收集奖励单调上升
OUTER 2 (grpo-o1)       collect → train
FINAL EVAL (grpo-o2, 贪心) = 10/15
```

**GRPO > SFT（7→10/15）**，净增 6,7,12（前沿任务），死区任务诚实不动。最终模型
`qwen-grpo-pi-7b-o2`。

**机制**：优势 = 组内 (reward − mean)/std；rollout 里"数对(1.0)"的轨迹得正优势、
"读对但数错(0.5)"得负优势 → 把最终的计数 token 往正确方向推。这是真实 harness 里的
on-policy 强化，不再是 micro-harness 的玩具信号。

---

## 7. 过程中修掉的真实 bug（工程沉淀）

| bug | 根因 | 修法 |
|---|---|---|
| Pi 请求 500 | Pi 发的 `content` 是结构化列表 `[{type,text}]`，Qwen 模板要字符串 | `_flatten_content` 拍平 |
| 后台进程被 SSH 断开带走 | nohup 在同会话进程组 | `setsid` 完全脱离 |
| 服务端收集中途 OOM 死 | 为算 logprob 做全序列 float32 logits forward（长 context 1.2GB） | 服务端只记 token ids，logprob 训练端重算 |
| 训练 OOM | 整条长序列 `logits.float()` + 显存碎片 + 一次性累积 270 turn | 只取 completion 位置再 softmax（1.2GB→18MB）+ `expandable_segments` + 按任务分批 step |
| 采样啰嗦撑爆显存 | temp>0 时模型 gen 到 max_tokens=1024 | 服务端钳 max_tokens=200 + 每次生成后 empty_cache |
| 同步脚本"假 network down" | `SSH()` 漏传 HOST；`SSHPASS` 没 export | 修正；网络对小文件其实一直正常 |

---

## 8. 迁移包（换大显存服务器用，成本低）

OOM 修复后 **24GB 已够用**（推理/训练交替，仅慢）。若要并行/更快/更大模型再换 48GB+。

- `scripts/server-setup.sh`：新机器装环境 + modelscope 下模型 + 注册 provider（镜像/绕封锁内置）
- `scripts/migrate-to-new-server.sh`：本地一键 推代码 → 远端 setup → 22s 重训 SFT → 冒烟
- **无需传输任何训练产物**：SFT adapter 由本地 `sft-dataset-v3.jsonl` 22 秒重训复现。

---

## 9. 诚实边界

- 15 任务套件是受控合成套件，不是公开 benchmark；SWE-bench 证据见 v3-swebench（7/10 子集，集成证据非 benchmark 声明）。
- GRPO 的提升限于"前沿任务"（模型本就有方差的地方）；死区任务（10/11/14）的计数能力
  超出当前 7B + 短 GRPO 的范围，需要过程奖励 / 更强基座 / 更多算力，这是如实记录的局限。
- 单卡 24GB 的推理/训练交替是工程妥协；生产级会用推理集群 + 训练集群 + 权重同步。
