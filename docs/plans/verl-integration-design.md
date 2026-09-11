# 我们是怎么接进 verl 的，以及为什么这么做

日期：2026-09-11。这份文件只解释设计与理由，不含新代码。
补充：`docs/architecture.md`（数据平面内核）与 `verl-native-takeover.md`（本轮实施计划）。

---

## 0. 一句话定位

项目分两个平面，我们**只替换训练平面**：

| 平面 | 归属 | 内容 |
|---|---|---|
| 训练平面 | **verl** | Ray 调度、FSDP2 分片、LoRA 优化器、权重同步、vLLM 推理引擎 |
| 证据/认证平面 | **我们自己** | 采证、verifier、episode/bundle 认证、admission 门 |

所以「接入 verl」不是把我们的东西塞进 verl，而是**让 verl 的 optimizer 和 vLLM 去跑我们的数据平面**。项目核心不在 trainer，所以 trainer 越属于框架越好。

---

## 1. 我们用的是 verl 的四个官方扩展点（不 patch 安装包）

| # | 扩展点 | 我们的实现 | 位置 |
|---|---|---|---|
| 1 | `rollout.agent.agent_loop_manager_class` | `CertifiedVerlAgentLoopManager` | `ray_trainer.py:824` 读取，`:838` 用它替换默认 manager |
| 2 | `AgentLoopManager.agent_loop_workers_class` | `PiAgentLoopWorker`（`ray.remote`） | 决定每个 worker 进程跑谁 |
| 3 | `rollout.agent.agent_loop_config_path` 指向的 YAML `_target_` | `PiAgentLoop` | `agent_loop.py:434-438` 注册进 `_agent_loop_registry` |
| 4 | `AgentLoopWorker.__init__` 的 `if not hasattr(self, "server_manager")` | `PiServerManager` | 基类只在**没有** `server_manager` 时才自建；我们先赋值，等于注入自己的 server manager |

**为什么坚持这四条**：不 patch venv 里的 verl 源码，升级/换 pin 不会把我们的改动冲掉，也让「我们用了哪些扩展点」这句话可核查。第 4 条是基类注释里写明 `# for recipe to change` 的正式钩子，不是 hack。

---

## 2. 完整调用链：从 `main_ppo` 到我们的入口

```
python -m verl.trainer.main_ppo            # Hydra 解析配置，起 Ray
 └─ RayPPOTrainer.fit()
     ├─ checkpoint_manager.update_weights(global_steps=0)     ray_trainer.py:1252
     │    └─ naive backend → actor_rollout_worker.update_weights
     │         └─ 收集 LoRA → 推给 vLLM → set_global_steps(0)   vllm_rollout.py:178-181
     └─ 训练步循环
         ├─ 1274  self.global_steps += 1
         ├─ 1310  gen_batch.meta_info["global_steps"] = self.global_steps
         └─ 1321  async_rollout_manager.generate_sequences(gen_batch)
              └─ AgentLoopManager.generate_sequences            agent_loop.py:1030
              │    （把 batch 切成 N 份，分给 N 个 worker，再 gather）
              └─ PiAgentLoopWorker.generate_sequences
                   ├─ 491  batch 若没有 agent_name，就用默认 loop 名
                   └─ 逐行：
                        agent_loop.py:552-566
                        hydra.instantiate(_target_=PiAgentLoop, server_manager=…)
                        await agent_loop.run(sampling_params, **kwargs)   ← 我们的入口
                        └─ PiAgentLoop.run                              pi_loop.py:208
                             → 返回 AgentLoopOutput
                        agent_loop.py:569  _agent_loop_postprocess
                        （prompt 左补零、response 右补零；response_ids 逐字保留，不重新 tokenize）
              → DataProto(prompts, responses, response_mask,
                          rollout_log_probs, rm_scores, …)
         ├─ GRPO 优势估计
         ├─ actor 更新（FSDP2 + LoRA）
         └─ checkpoint_manager.update_weights()   ← 原地把新 LoRA 同步进活着的 vLLM
```

**这条链上最关键的一个事实**：`_agent_loop_postprocess`（`agent_loop.py:569-633`）**从不重新 tokenize**。它把 `output.response_ids` 原样右补零，并信任：
- `response_mask`：1 = LLM 生成的 token，0 = 工具观察/padding（`:618-633` 逐字实现）
- `response_logprobs`：直接放进 `rollout_log_probs`（`:820` 附近的 `_postprocess`）

这正是我们数据平面的语义。所以我们的认证序列可以**无损**映射到 verl 的类型上，不需要迁就任何框架约定。

---

## 3. `PiAgentLoop.run` 内部：数据平面真正做的事

`pi_loop.py:208` 起，每个样本一次：

1. **建 workspace**，写入题面（`pi_loop.py:352-356`）。workspace 刻意放在 episode 目录**之外**——认证门会枚举 episode 目录，放进去会被当成多余产物。
2. **起真实 Pi 0.84.2 子进程**，通过我们已有的 `PiHostExecutionOrchestrator`。
3. **Pi 只认识我们的本地代理**：`PiHostExecutionSpec.upstream_url` 指向本地 `ModelProxyHttpServer`，不是 vLLM。Pi 永远不直连引擎。
4. **代理是采证层**：向上游索要 `return_token_ids` / `return_tokens_as_token_ids` / `logprobs`，把每一次调用的原生 `prompt_token_ids` / `response_token_ids` / 逐 token `response_logprobs` 落成 `model-evidence.jsonl`，并绑定 `ExecutionIdentity`（run/task/episode/attempt/policy fingerprint/sampling fingerprint）。
5. **verifier 在 workspace 里跑**，产出 PASSED/FAILED。PASSED→reward 1.0，FAILED→0.0，**其它状态直接抛错**，绝不编造 reward（`pi_loop.py:269-280`）。
6. **组装**：证据 → `bridge.py` 的严格连续性检查 → `assemble_episode_sequence` → `AdmittedVerlSequence`，落盘 `admitted-sequence.json`。
7. **返回** `AgentLoopOutput(prompt_ids, response_ids, response_mask, response_logprobs, reward_score, num_turns, metrics, extra_fields)`。

**为什么必须有自己的代理**（这是护城河，不能省）：如果让 Pi 直连 vLLM，就没有任何组件在**逐调用**地记录原生 token 与 logprob、并把它们和 episode 身份绑成可审计证据。代理是数据平面的落点。

---

## 4. 认证门为什么放在 manager，而不是 loop 里

我们的认证需要**整个 group**才能判定：任务绑定、组最小规模、组内奖励方差。
但 `AgentLoopBase.run` 是**逐样本**的，看不到组。

所以门放在 `CertifiedVerlAgentLoopManager.generate_sequences`（`verl_manager.py:153`）：

```
generate_sequences
 ├─ 157  engine_step = meta_info["global_steps"] - 1     ← 定位“上一份已同步的权重”
 ├─ 160  非 0 时用 checkpoint_root/global_step_N/actor 的 sha256 作为 policy 身份
 ├─ 187  await super().generate_sequences(prompts)        ← 真正的 rollout
 ├─ 189  _certify_from_evidence(...)                      ← 读盘上证据，批级认证
 ├─ 190  _verify_returned_batch(...)                      ← 校验返回张量与证据一致
 └─ 失败即抛，不给 trainer
```

**为什么放这里**：`generate_sequences` 是 trainer 拿到数据前的最后一道关口，在这里拒绝等于「认证在训练前执行，失败不能旁路」。而 `engine_step` 绑定 checkpoint 摘要是为了满足 §5.3 第 9 项——policy 身份必须绑定到**真实同步过的权重**，不能用服务名或计数器冒充。

---

## 5. 新的 native transport：为什么要把 HTTP 那一段拆掉

**旧路径的问题**（这是上一轮卡住的根因）：

```
Pi ──HTTP──> 我们的代理 ──HTTP──> vLLM
```

上下文由 **Pi 决定**。而 Pi 会把模型上一轮的生成**丢掉**，用自己的方式重渲染助手轮。实测（smoke16 `a0s0`）：

```
call 0 生成 1024 token：一段 prose + 两个裸 JSON 工具调用
Pi 追加进下一轮的只有 40 token：<tool_call>{"name":"read",…}</tool_call> + 真实工具返回
→ P_1 不包含 G_0 的任何一个 token
```

于是 `prompt_{i+1} = prompt_i + generation_i + observation` 不成立，严格契约在多轮上必然失败。

**新设计不放松契约，而是不再问 Pi「上下文是什么」。** 代理维护自己的**只追加原生 token 账本**：

```
账本 = 冻结模板编码的首轮 prompt
     + 原生生成的 token（逐字保留，永不重新 tokenize）
     + 新增的工具观察与模板边界的编码

每次调用： server_manager.generate(prompt_ids=账本)     native_transport.py:55
```

Pi 发来的 `messages` **降级为需要校验的协议证据**，不再是 prompt 的来源（`token_context.py:73-81` 校验它有没有改写历史、工具结果有没有错配）。

**为什么这样更好**：

1. `prompt_{i+1} = prompt_i + gen_i + obs_i` **由构造保证成立**，严格契约一个字不用改。
2. 少一跳 HTTP，直接拿到 `TokenOutput`：`token_ids` / `log_probs` / `extra_fields["global_steps"]`。`global_steps` 正是**把策略身份绑到真实权重同步**的凭据。
3. `accept()` 要求原生 EOS（`token_context.py:106-109`），任何截断/中止的生成当场暴露，不会被当成正常回合。

---

## 6. 已验证 vs 假设（重要）

**已验证**（对着真实产物测过）：

- `AsyncLLMServerManager.generate` 签名与调用一致；返回 `TokenOutput`，字段齐全。
- pre-rollout 同步把 `global_steps=0` 推进引擎（`ray_trainer.py:1249-1252` → `set_global_steps`），
  而首步 `global_steps=1`（`ray_trainer.py:1274`），故 `engine_step = global_steps - 1 = 0` 对得上，无 off-by-one。
- 账本能逐 token 保住原生生成（真实 tokenizer 实测：保留为真，仅追加 20 个后缀 token）。
- 服务器有 `/usr/bin/bwrap`，隔离方案可落地。

**实测为假的假设，以及各自的处置**：

| # | 假设 | 实测 | 处置 |
|---|---|---|---|
| A | 策略会吐 `<tool_call>` 标签，上游 Hermes parser 能解析 | 10 条真实生成**解析出 0 个调用**（12 条里 0 个标签） | 已修：`_tool_calls` 在有标签时用上游 parser、无标签时走仓库既有的严格 JSON 抽取（非宽松修复） |
| B | 模型会在预算内正常收尾并发 EOS | 4/4 回合耗尽整个预算、从不发 `<|im_end|>` | 见 §8：按 verl 语义把截断回合当合法动作 |
| C | Pi 会原样回传 assistant 消息 | Pi 回 `content: null`，且只保留部分 tool_calls | 已修：只比对承重字段（role + tool_calls），content 非空时才要求一致 |

C 的定位实验（真实消息驱动 `AppendOnlyTokenContext`）：

| pending 构造 | 第二轮 |
|---|---|
| content = 正文（当时的代码） | 抛 `rewrote prior messages` |
| content = `""` + Pi 真实 tool_calls | **通过**，账本原生生成保留为真 |

即：**B 与 C 都不是设计错，而是配置与一行取值错**；A 是真正的设计缺口——数据平面既已认定「Pi 的文本重渲染不可信」，就不能把动作抽取完全交给只认标签的上游 parser。

---

## 8. 截断的回合算不算一个回合（2026-09-11 决定：算）

**问题。** 实测（smoke22，4/4 回合）：`tokens == budget == 3584`、`stop_reason == "completed"`、
从不出现 `<|im_end|>`。旧实现要求回合必须以原生 EOS 收尾，于是**一个可训练回合都产不出来**，
#1（多轮组装）在 GPU 上从来没被真正跑过。

**为什么按 verl 的语义处理，而不是加严。** 这不是"放宽标准"，是对齐我们要接进去的框架：

- `tool_agent_loop.py:246`：`agent_data.response_mask += [1] * len(response_ids)` —— **无条件下 mask=1**；
- `:254`：终止条件只有 `len(agent_data.response_mask) >= self.response_length`；
- 在整个 `agent_loop` 目录里 grep `im_end|eos|EOS` → **0 命中**。

也就是说 verl 自己的多轮循环从不检查 EOS，长度耗尽就是正常终止。我们比框架更严，而且这种严是
**自相矛盾**的：`stop_token_ids=[<|im_end|>]` 让引擎在 EOS 处停下，而 vLLM 的契约是
**特殊 stop token 不返回**——所以"干净收尾"的回合永远拿不到那个 token。

**改法（三层，缺一层都跑不通）。**

1. `accept()`：缺少 terminator 的回合**照样计入账本**。区别（finished vs truncated）挪到证据里
   （逐回合 `terminated` 字段、`truncated_turns` 计数、`extra_fields`）——台账不再承担这个语义。
2. 预算重排：原来 `per-generation = response_length = 3584`，**一个截断回合就吃光整个 episode**，
   第二轮在数学上不可能存在。现在 `5 × 1024 + 4 × observation ≤ 8192`，
   `max_model_len = 2048 + 8192`。为什么每回合 1024 够：smoke16 那条被 1024 截断的生成里
   **含一个完整的 `read` 调用**（fixture 实测，位置在正文第 85 字符）。
3. 预算耗尽必须**干净收尾，而不是失败一次模型调用**：orchestrator 要求**每一条**记录的调用都带
   token ids / logprobs / status<400，一次失败就让整个 episode 的证据作废（`all_model_calls_usable`），
   而工具输出过长不是策略的错。所以 proxy 在**调用引擎之前**问 transport 的 `can_serve`：
   付不起就回一个**不含任何模型输出的** assistant 回合（空 content、无 tool_calls、不记录证据），
   并把原因写进 `engine-closeout.jsonl`。顺序上必须在引擎之前决定，否则会留下一条失败的 evidence。

**代价（诚实记下）：** 被截断的尾部会以 mask=1 进 loss（verl 亦然）。可核对的口径是
`truncated_turns` 与 `engine_closeout`，它们会随每个 episode 落盘。

---

## 9. 没有工具调用的回合算不算样本（2026-09-11 决定：算，奖励照实记）

**问题。** smoke23（8 个 episode，n=8）里 8/8 的首回合都被接受了（1024 token、截断、
`terminated: false`）——§8 的改动生效了；其中 **2 个回合带真实工具调用，并且已经发出第二次模型请求**
（`MODEL_REQUEST → MODEL_RESPONSE → MODEL_REQUEST`，在 batch 被拆掉时正在跑）。整轮死在我们**自己**的
门上：`pi_loop` 的 `num_tool_rounds < 1` 抛错，而 `generate_sequences` 的 gather 一旦有异常，
**同一批里另外两个正在跑第二回合的 episode 也一起被丢掉**。

**为什么该改的是这道门，而不是数据。** 三条实测：

1. 那个"没有动作"的 episode 本身被 ON_POLICY_RL 认证为 `VALID` + `ELIGIBLE`、`score: 0.0`
   （`termination_reason: PI_PROTOCOL_DECLARATION`）。也就是说这道门**比认证契约更严**——
   它不是在保护契约，而是在契约之上又加了一条。
2. **没有调用是被预算截断的**：8/8 都没有停在未闭合的 JSON 里（`cut_off=0`）。
3. 严格抽取器**没有拒掉任何合法调用**；它拒的是 `{}`、`{...}`、`{j!=i}`、JS 片段这类
   叙述占位符。真正自发调用的那两条，调用出现在文本偏移 0（即以调用开头）。

结论：**6/8 的首回合是策略"只说不做"**——它把 1024 token 全用来叙述。这不是集成缺陷，
是**奖励现象**：动作就是策略自己生成的 token，奖励是任务结果，这正是 PPO 类算法要处理的东西；
把它变成"整批作废"既没有依据，也丢掉了同批其他样本。至于"整批都没有调用"的极端情形，
**batch gate 的方差不变量**本来就该拦（`manager.py:122`），那才是"没有学习信号"该被拒的地方。

**保留可见性**：`num_tool_rounds` 照旧写进 `admitted-sequence.json` 与 `extra_fields`，
所以"这条样本没有工具轮"在证据里是可查的，不是被藏起来。

**一处已知的不一致（留给用户决定，未单方面改）**：CPU 认证路径 `admit_on_policy_manifest`
→ `_admit_episode_sequence` 仍然要求 `num_tool_rounds >= 1`（那是"合格轨迹"的项目定义），
而 live 的训练路径不再要求。两条路径允许不同，但如果要把"合格轨迹"的定义统一，
那是方法论决定，不该由我悄悄改掉。

---

## 10. batch gate 的 `tuple != list`：一个不可能满足的比较（smoke24 实测）

**现象。** smoke24（n=16）第一次真正跑到 batch gate，16 个 episode **全部**被拒：
`policy artifact does not bind actual inference context`。这条消息既没给 episode id，也没给字段，
而它比较的两侧都是落盘产物——**失败无法从 evidence 复现**，因为"现在再比一次"是通过的
（当时的排查因此空转很久）。

**修法第一步不是改逻辑，而是让消息可诊断**：加上 episode 名、字段、两侧长度、首个不同下标。结果一次命中：

```
prompt_ids: stored_len=1703 rebuilt_len=1703 first_diff_index=None   ← 四个数组全都如此
```

长度相同、**没有任何一个元素不同**，但整体 `!=`。原因是
`ProducerArtifact.from_dict` 会把 payload **深度冻结**：存盘读回来是 **tuple**，
而重建出来是 **list**，Python 里 `tuple == list` 恒为 `False`。直接量到的事实：

```
raw_type=list  from_dict_type=tuple  raw == typed -> False
list(raw) == list(typed) -> True
```

也就是说这道门**在构造上不可能通过**；它此前从没被跑到，所以一直没暴露，并且挡死了每一轮框架训练。

**修法（`training_sequence_matches`）**：按元素比较，而不是按容器比较——这不放松任何要求，
比的是同一批 token。`describe_sequence_difference` 对"token 相同、只是容器类型不同"会明确
写成 `same tokens, containers differ (tuple vs list)`，而不是伪装成 drift。

**顺带修好的可观测性**：gate 现在对每个 episode 打一行
`[pi-manager] certify <episode> calls=… tool_rounds=… seq_len=… verifier=…`。一个会一次拒掉
16 个样本的门，必须在运维真正会读的那份日志里说清是哪一个、为什么。

**这一轮同时证明了 §8/§9 的改动在真实训练里成立**：16 个 episode 里 6 个跑了多轮（最多 5 轮），
每个回合都以截断动作被接受，其中 4 个完成了真实工具轮，**2 个真正解出了 Mbpp/118**
（reward 1.0 vs 0.0）——即组内**确实有奖励方差**；把同一份 evidence 离线重放 batch gate，
结果是 `BATCH CERTIFIED`。

---

## 11. 结果：verl 自己的 trainer 跑完了一整步（smoke25，2026-09-11）

`verl.trainer.main_ppo` → `RayPPOTrainer.fit()` 完成了一个完整的 GRPO step，证据在
`docs/plans/verl-closeout-evidence/phase-g-native-trainer/smoke25/`：

| 事项 | 实测 |
|---|---|
| 16 条真实 Pi episode（bwrap 隔离、真实工具轮） | `certify-lines.txt`：`calls` 1–5、`tool_rounds` 0–4、`seq_len` 736–5524 |
| 组内奖励方差（不能凭空造） | `critic/score/mean: 0.0625`（= 1/16 通过），`min 0.0` / `max 1.0` |
| GRPO advantage 真的算了 | `critic/advantages/mean: 0.1779`，`max 3.75`，`min -0.25`（没有方差这些不可能出现） |
| 框架执行了参数更新 | `timing_s/update_actor: 117.57`、`actor/grad_norm: 0.1114` |
| 框架**原地**同步权重 | `timing_s/update_weights: 2.98` |
| checkpoint 由框架自己的 FSDP2 worker 写出 | `global_step_1/actor/{model,optim,extra_state}_world_size_2_rank_*.pt`、`data.pt` |
| **真实参数变化（不是空转）** | P1 adapter sha256 `f0ddcac8…` ≠ P0 `6db6a40c…` |
| 我们的 token 契约在框架内自洽 | `training/rollout_actor_probs_pearson_corr: 0.99951` |

**§5.3 第 2 项（实际训练由 verl 执行）由此升级为满足。**

**§5.3 第 9 项当时仍为部分满足，而且原因很清楚**：这一轮**所有 rollout 都发生在同步之前**，
所以"同步后的新权重被 Pi 使用"没有证据。要拿到它需要一个**跑到第二步**的 run——
smoke26（`total_epochs=2`）本来正是为此，但那一组 16 条**全部 FAILED、组内零方差**，
被 gate 正确拒绝（`smoke26/` 留档：这本身就是 §5.3 第 6 项的正面证据）。

当时的结论是"剩下的约束是数据，不是集成"。**这个结论只对了一半**：数据确实是约束，但在它前面
还站着一个真的代码缺陷——`max_attempts` 从来没生效过。见 §12。

---

## 12. `max_attempts` 是假的；以及第 9 项的证据（smoke27，2026-09-11）

### 12.1 一个看起来像重试循环的单次尝试

`verl_manager.py` 的 `generate_sequences` 里原本写着：

```python
for attempt in range(self.max_attempts):
    self._stamp_attempt(prompts, attempt, call_root)
    batch = await super().generate_sequences(prompts)
    try:
        certified = self._certify_from_evidence(call_root, attempt)
        self._verify_returned_batch(batch, prompts, call_root, attempt)
    except ContractValidationError as exc:
        raise ContractValidationError(
            f"{self.run_id}: batch rejected without resampling: {exc}"
        ) from exc
    ...
```

`except` 分支自己 `raise`，所以**循环体永远只执行一次**：`attempt` 之后的分支不可达，
`last_error` 是死代码，`max_attempts` 只被读出来校验、从没被使用。报错文案
"batch rejected **without resampling**"因此是字面属实的——它确实不重采，尽管类注释、
配置项和 `_stamp_attempt` 的 attempt 参数都在承诺会重采（`pi_loop.py:228` 也确实按
`attempt-{n}` 分目录，即整套设计都预期会有第二次）。

为什么这个缺陷能活到今天：它**只在一种输入上可见**，而这种输入前面几轮都没走到——smoke18–22
分别死在 step guard、int64 序列化、EOS 检查、截断语义上；smoke24 死在 `tuple != list`
（§10）；smoke25 一次就认证通过（`n=16` 且恰好有 1 条解出）；smoke26 第一次真正触发零方差，
于是整个 run 结束。**一个只在失败路径上存在的缺陷，只有在失败真的发生时才会暴露。**

### 12.2 修法：只重采"退化批"

放行所有失败去重试是错的：像 §10 那种 `tuple != list` 的缺陷每次都会以完全相同的方式复现，
重采三次只会把同一轮 GPU 花三次，并把一个确定的 bug 报成"偶发"。所以按**能否被新的抽样改变**
把它分成两类：

- `src/errors.py` 新增 `DegenerateBatchError(ContractValidationError)`：组内奖励方差为零，
  advantage 恒为 0，GRPO 拿不到任何信号。这是**抽样运气**，新的 rollout 可以改变它。
- 两处方差检查（`manager.py:certify_batch`、`admission.py:_require_reward_variance`）改抛它。
- 重试策略单独成模块 `src/integrations/verl/resampling.py`（不含 ray/verl，因此可在无 GPU
  环境下测）：只重采 `DegenerateBatchError`；其余 `ContractValidationError` **首次即停**；
  每次被拒的尝试写 `rejected-attempt{n}.json`，让"重采"在事后看得出来，而不是长得像第一次。
- 被否决：把重试放在 `max_attempts` 之外的地方（例如 trainer 层），因为重试必须重新生成，
  而生成只发生在这个函数里。

### 12.3 组大小来自实测解题率，不是来自口径

收尾规格把每轮定为 4 条 episode。但 GRPO 需要组内有奖励差，也就是**这一组里必须至少有一条解出**，
而冻结的 P0 在 Mbpp/118 上的实测解出率约 1/16（smoke24 2/16、smoke25 1/16、smoke26 0/16，
合计 3/48）：

| n | 一次尝试至少含一条解的概率 | 备注 |
|---|---|---|
| 4 | ~22% | 原收尾口径；E 阶段靠它连过两轮，是低概率事件 |
| 16 | ~60% | smoke25/26 的取值 |
| 32 | ~87% | smoke27 的取值；配 `max_attempts=3` 后 step-1 过闸 >99% |

### 12.4 结果：两个 step 都跑完，且第二步的 rollout 用的是第一步同步出去的权重

`verlpi-smoke27`，2026-09-11，22m34s，`n=32` / `total_epochs=2` / `max_attempts=3`。

| 事项 | 实测 |
|---|---|
| 第一次抽样就被拒（而不是终止整个 run） | `resample.txt`：`attempt 0 carries no learning signal, redrawing: group '53ca36e3…' has no intra-group reward variance` + `rejected-attempt0.json`（`reason: DegenerateBatchError`）。**同样的输入在 smoke26 会直接结束 run** |
| 重采后过闸 | gen-000 有两个 attempt 目录，`certified-batch-attempt1.json` 存在；该次 32 条中 2 条解出 |
| 框架跑完两个 step | `step:1` 与 `step:2`、`local_global_step_folder: …/global_step_1` 与 `…/global_step_2`、`Training Progress: 100%\|██\| 2/2 [22:34]` |
| 两次真实的参数更新 | P1 `15a741e2…`、P2 `30e3f3ad…`，均 ≠ P0 `6db6a40c…`；`actor/grad_norm` 0.1194 / 0.1009 |
| 框架原地同步 | `timing_s/update_weights` 2.90s（step 1）/ 2.83s（step 2） |
| **同步之后仍有 rollout，且用的是新权重** | gen-001 的 `policy_generation: P1`、`adapter_revision: 13cc8199…`，而该值 = 对 `global_step_1/actor` 重算的摘要（`smoke27/checkpoints.txt` 标 **MATCH**）；gen-000 则是 P0 / `6db6a40c…` |
| 两次采样的行为策略确实不同 | episode 级指纹 `e3eca6fe…`（gen-000）vs `aed2a953…`（gen-001），各自等于其轮策略 checksum；批内混指纹会被 gate 拒绝 |
| 第二步的梯度只能来自第一步的权重 | `global_step_2` = `30e3f3ad…` ≠ `global_step_1` = `15a741e2…`，而 step 2 的 rollout 只有 gen-001 |
| 我们的 token 契约在框架内仍自洽 | `training/rollout_probs_diff_valid: 1`、`rollout_actor_probs_pearson_corr` 0.99957 / 0.99959 |
| 真实工具循环 | `timing_s/agent_loop/tool_calls/mean` 0.8125 / 0.875；`num_turns/mean` 1.8125 / 1.875，max 5 |

**§5.3 第 9 项由此升级为满足，十三项全通过，§5.4 的项目宣称启用。**

第 9 项是"框架能允许的最强形式"：引擎仍不能自报它服务的是哪一步
（`checkpoint_engine.backend=naive` 丢弃 `global_steps`，缺口逐次写进
`engine-notes/<episode>.step.jsonl`）。把身份钉住的是**框架自己写出的 checkpoint 的摘要**，
且这一身份在三个层级上一致：轮策略、每条 episode 的指纹、以及会拒绝任何其它指纹的批认证。

**不宣称效果**：2/32、1/32 是噪声，留出评测是 F 的事，这一轮是链路结果。

### 12.5 仍未解决

- 重采只在 smoke27 的第一次抽样上真实触发过一次（attempt 0 被拒 → attempt 1 通过）。
  "所有尝试都退化"这条耗尽路径目前只有单测覆盖。
- `checkpoint_engine.backend=naive` 不告知引擎 step（见上），已记录而非假设。
- 两个序列构造器（`pi_loop.py` 注入的 `training_sequence_builder` 与 `_assemble`）仍并存。

---

## 7. 工作约定（用户 2026-09-11 明确要求）

1. **先说明白为什么，再改代码。** 每次修改前先给出：改什么、为什么这么改、哪条实测支持它、被否决的方案是什么。
2. **设计主张必须由真实产物支撑**：真实 Pi、真实 token、真实引擎。合成输入的单测不算证据——上一轮 20 个 CPU 测试全绿，却漏掉了一行 `content` 取值错误，正是因为它们用的是编造的输入。
3. **明确区分「已验证」与「假设」**，假设必须标注出来，不允许混在结论里。
