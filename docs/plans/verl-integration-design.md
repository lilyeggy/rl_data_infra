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

**仍是假设，且实测为假——这三条必须在改代码前解决**：

| # | 假设 | 实测 | 后果 |
|---|---|---|---|
| A | 策略会吐 `<tool_call>` 标签，上游 Hermes parser 能解析 | 10 条真实生成**解析出 0 个调用**（12 条里 0 个标签） | `message` 无 tool_calls → Pi 不执行 → `num_tool_rounds<1` 抛错 |
| B | 模型会在预算内正常收尾并发 EOS | 12/12 被截断在 1024（launcher 仍写 `max_tokens_per_generation: 1024`） | `accept()` 抛 `generation lacks native EOS` |
| C | Pi 会原样回传 assistant 消息 | Pi 回 `content: null`，且只保留部分 tool_calls | 回传校验抛 `Pi rewrote prior messages` |

C 的定位实验（真实消息驱动 `AppendOnlyTokenContext`）：

| pending 构造 | 第二轮 |
|---|---|
| content = 正文（代码现状） | 抛 `rewrote prior messages` |
| content = `""` + Pi 真实 tool_calls | **通过**，账本原生生成保留为真 |

即：**B 与 C 都不是设计错，而是配置与一行取值错**；A 是真正的设计缺口——数据平面既已认定「Pi 的文本重渲染不可信」，就不能把动作抽取完全交给只认标签的上游 parser。

---

## 7. 工作约定（用户 2026-09-11 明确要求）

1. **先说明白为什么，再改代码。** 每次修改前先给出：改什么、为什么这么改、哪条实测支持它、被否决的方案是什么。
2. **设计主张必须由真实产物支撑**：真实 Pi、真实 token、真实引擎。合成输入的单测不算证据——上一轮 20 个 CPU 测试全绿，却漏掉了一行 `content` 取值错误，正是因为它们用的是编造的输入。
3. **明确区分「已验证」与「假设」**，假设必须标注出来，不允许混在结论里。
