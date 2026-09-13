# 面试题库：面试官视角全真模拟（Agentic RL 数据平面 + verl 接入）

> 配套 [interview-mastery-workbook.md](interview-mastery-workbook.md) 使用：workbook 练"吃透"，本文件练"被拷打"。
> 题源两层：① 本项目真实代码与事故（file:line / run artifact 可回溯）；② 市面真题池（GRPO/PPO/DPO/DAPO 八股、verl·AgentLoop 架构题、vLLM 推理题、Agentic RL 工程题），链接见文末附录 C。
> 用法：每题先主问、再追问；**答不出追问的第二层 = 该题未通过**。建议录音自测，答案写进自己的 `interview-notes.md`（半页以内，结论 + 证据位置）。
> verl 源码行号基于 pinned v0.7.1（bec9ef74）安装副本，上游升级后行号会漂。

## 如何评分（先读这个）

- **3 分**：主问 + 全部追问链都能答，且能给出 file:line 或 run artifact 证据。
- **2 分**：主问 + 第一层追问能答。
- **1 分**：只有背诵内容，一追问就崩。
- **0 分（Fail）**：说出措辞红线任一——"RL 提升了模型能力" / "已接入 Slime" / "r3 已完成 300 步"；或把必背数字答错。

## 高危题 TOP 10（面试官最可能用来一击致命的题）

1. GRPO 组内零方差为什么没信号？你的重采样门失败语义怎么分层的？
2. 为什么 RL 必须训练在模型原生采样的 token 上？re-tokenize 到底错在哪一步？
3. 组装后的训练序列里，哪些 token mask=0？它们的 logprob 填什么？为什么？
4. `rollout_actor_probs_pearson_corr 0.9996` 是"什么和什么"的相关？为什么它是你最有力的单一证据？
5. RL 没有显著提升，这个实验的价值是什么？（Fisher p=0.826 你怎么解读？）
6. `save_freq` 为什么必须是 1？磁盘安全的唯一杠杆是什么？
7. 18 小时训练进程消失、没有任何 Traceback，你的诊断顺序是什么？
8. 你改了四个官方扩展点——框架在启动时是**怎么找到**你的类的？
9. verl 原生 multi-turn loop 从不检查 EOS，你的 turn 终止语义怎么对齐的？
10. DAPO 的 dynamic sampling 和你的重采样门是什么关系？

---

## 第 0 轮：开场题（30 秒定调，每轮面试必问）

**Q0.1 用 90 秒讲清这个项目：问题、方案、做到哪了。**
- 追问①：一句话说清"数据平面"和训练框架的边界画在哪。
- 追问②：如果只保留一句话到简历上，是哪句？
- 评分点：问题（普通 Agent 日志缺工具 I/O / policy 错位 / 基础设施故障当 reward 0 / re-tokenize 失真）→ 方案（capture→组装→fail-closed 认证→编译 + 4 官方扩展点接 verl）→ 状态（链路已验证 / 效果未复现 / 负面留档）。边界句："清洗是 rollout 侧的出门质检，不是 trainer 侧过滤器；verl 侧我们一行代码都没在训练循环里。"

**Q0.2 你在项目里的角色？AI 参与了多少？**
- 追问①：举一个你自己拍板、AI 不可能替你拍的技术决策。
- 评分点：拍板例——truncated turn 算不算合法样本（与框架对齐，mask=1 保留真实 token）；它改变"我们训练在什么数据上"，是契约级决策。

**Q0.3 现在什么做完了、什么没做完？**
- 评分点：三层——机制（已验证：两个框架 step、同步后被消费、Pearson 0.9996）、规模（r2 24/24 步 done；r3 300 步进行中 276/300 中断）、效果（未显著，如实报告）。绝不含糊百分比。

**Q0.4 为什么叫"数据平面"？**
- 追问：你项目里有几行代码跑在 verl 训练侧？
- 评分点：边界 = `generate_sequences` 的返回值；vLLM 和认证门都在 rollout 侧；训练侧（advantage/PPO-clip/FSDP/同步/checkpoint）全部框架拥有。

**Q0.5 项目最难的三个技术问题？**（筛信号题，答案质量决定面试官往哪深挖）
- 参考好答案：① token 对齐（append-only 账本不变量）；② 零方差门的失败语义分层；③ on-policy 身份绑定（引擎不报 step 时的三级 digest）。

---

## 第 1 轮：数据平面（capture → 认证 → 编译）

**Q1.1 ExecutionIdentity / TraceEvent / AgentEpisode / ExecutionBundle 各解决什么问题？**
- 追问①：为什么 `task_id` 不能当执行身份？
- 追问②：如果让你砍掉一层，砍哪层？砍了会坏什么？
- 评分点：身份≠任务（同 task 多 attempt）；TraceEvent 是不可变事实单元；Episode 确定性组装；Bundle 做 join 且**认证不写回 bundle**——追问②答"认证不写回"的理由：避免循环 checksum 图。

**Q1.2 "任务失败 ≠ 基础设施失败"具体怎么实现？**
- 追问①：三轴是哪三轴？（task_status / execution_validity / integrity）
- 追问②：一个 episode 首个模型调用就 502，系统怎么标？进不进训练？（INFRA_INVALID，不进——verifier_status 是 ERROR 语义）
- 追问③：如果把它当 reward 0 喂给 GRPO，组均值被怎样污染？（组相对优势的 baseline 里混入非策略信号；策略为"假动作"接梯度）
- 评分点：`all_model_calls_usable` = 每个调用 status<400 且有 TOKEN_IDS/BEHAVIOR_LOGPROBS/POLICY_VERSION——一个调用失败作废整个 episode，所以长工具输出不能是策略的锅（引出 Q2.7 的 200 closeout）。

**Q1.3 fail-closed 认证真实拒绝过什么？讲一个 bug 故事。**
- 标准故事：`ProducerArtifact.from_dict` 深冻结 payload → 存的是 tuple，重建的是 list，`tuple == list` 是 False → 检查**构造上不可满足**，阻塞所有框架轮。
- 追问①：为什么 legacy 路径从没暴露？（内存对象容器类型恒一致；只有序列化 round-trip 后才出现）
- 追问②：错误信息没说哪个 episode、哪个字段，你第一步做了什么？（先修**可诊断性**：`describe_sequence_difference` + per-episode certify 日志；下一轮就打出 `stored_len=rebuilt_len=1703, first_diff=None`——"元素全同、容器不同"直指容器。原则：gate 能拒整批时，先让它报出 item 和 field 再谈理论。）
- 追问③：修复为什么不是"放宽成 =="？（比较元素而非容器——`training_sequence_matches`，语义不变、实现健壮）

**Q1.4 幂等写入 / checksum / quarantine / backpressure 各防什么？**
- 追问：backpressure 为什么是有界队列 + HTTP 429，不是无限缓冲？（下游读速恒定，无限队列只是把 OOM 换个位置；429 让上游显式重试，事件不丢）
- 评分点：每项给一个会发生的真实事故（重复事件、乱序、历史改写、磁盘满）。

**Q1.5 【代码题】设计"确定性组装"：事件流乱序到达、可能重复，如何从原始事件重建出 byte-identical 的 episode？**
- 评分点：排序键（全局序）、去重键（event id）、append-only、重建可重放；能引到"episode 可由原始事件重建"这条 README 底线。

**Q1.6 数据清洗的代表性故事（postmortem）。**
- 评分点：`[REDACTED]` thinking token、宿主绝对路径泄漏、逐轮分布偏移三个真实缺陷 + fail-closed 修复（`docs/experiments/sft-apps-260905-data-quality-postmortem.md`）。

---

## 第 2 轮：verl 接入机制

**Q2.1 四个官方扩展点是什么？框架怎么"找到"你的类？**
- 追问①：config 里字符串 vs Python attribute 两种注入方式的边界？（四个类都是**框架 new 的** → creation rights：名字/属性是唯二入口）
- 追问②：`agent_loop_manager_class` 前面为什么要加 `+`？（不在 YAML struct 里，Hydra 需要显式新增键）
- 追问③：`agent_name` 写在数据行的哪里？per-row 有什么用？（**顶层**字段，不是 extra_info 里；行级 loop 选择器，当前只有 `pi_agent` 一个入口）
- 评分点：`rollout.agent.agent_loop_config_path` → YAML `_target_`；`agent_loop_manager_class` FQN；`agent_loop_workers_class` Python 属性；AgentLoopWorker 的 `server_manager` recipe hook（`hasattr` 约定）。

**Q2.2 `pi_certification` 参数块为什么放 config 根部？**
- 评分点：`rollout.agent` 是 typed dataclass（AgentLoopConfig），未知键直接拒绝——这是框架类型系统的 fail-closed；所以自有参数开顶层命名空间。

**Q2.3 你的 Manager 覆写了 `generate_sequences`，写出它相对父类的职责序列，并说一个你真实引入的 bug。**
- 评分点：盖章（4 stamps）→ 派发（chunk+gather+concat, agent_loop.py:1030-1044）→ 等待 → per-episode 重验（`certify_for(ON_POLICY_RL)` 从盘重跑, verl_manager.py:304）→ 批 gate → 重抽。
- 真实 bug：覆写时丢了父类的 `@auto_await`（ray_utils.py:97）——trainer 是同步调用（ray_trainer.py:546/:1321），拿到裸 coroutine，全批死亡。
- 追问：padding 单行 batch 到 worker 数会发生什么？（**复制该行**；所以 episode 以 session id + generation 为键）

**Q2.4 为什么 gate 放在 Manager 而不是 Worker？**
- 评分点：worker 看不见全批 / 不可信（自己验自己）/ 来不及（批级决策在更新前）；manager 是"验收"，worker 是"车间"。stamps 的目录就是 stamps 的产物（episode/attempt-*/）。

**Q2.5 零方差重采样门：失败语义怎么分层？为什么这样分？**
- 追问①：`max_attempts` 曾经是假的——讲这个故事。（retry 循环在自己 `except` 里 re-raise → 单发；message "batch rejected without resampling" 字面为真；修复 = `DegenerateBatchError` + 策略抽到 `resampling.py`）
- 追问②：为什么确定性违规不重试？（重抽改变不了它——确定性缺陷会复现，重试只会把它伪装成间歇性问题、烧 GPU）
- 追问③：`resampling.py` 为什么不许 import ray/verl？（无 GPU 可测——9 个单测）
- 评分点：只重抽"flat reward group"（新抽样可能改变它）；其余首现即停；每次拒绝写 `rejected-attempt{n}.json` 留痕（smoke27 的 attempt-0 → attempt-1 就是它救的）。

**Q2.6 turn 终止语义：verl 原生 multi-turn loop 检查 EOS 吗？**
- 追问①：证据？（`tool_agent_loop.py:246` 无条件 `response_mask += [1]*len(response_ids)`；`:254` 只按长度终止；agent_loop 包 grep EOS 零命中）
- 追问②：`stop_token_ids=[151645]` 和"检查返回 ids 末尾是 EOS"为什么不兼容？（vLLM 对 special token 停止但不回吐——互斥；注意 Qwen2.5 `eos_token_id` 是 151643，不是 151645）
- 追问③：三态 terminator 是哪三态？（present / stripped / absent——截断改由 `len(ids)==budget` 判定）
- 追问④：无终止符的截断 turn 为什么最终判合法？你拍板的依据？（与框架对齐：mask=1 保留真实 token，终止信息进 evidence 的 `terminated` 字段；被 strip 的 `<|im_end|>` 恢复为上下文 mask=0）

**Q2.7 预算算术与 episode 耗尽：为什么 per-generation == response_length 是致命配置？**
- 评分点：一代截断吃光整 episode → turn 2 算术上不可能（`remaining<=0` 直接 raise）；修复后 `5×1024 + 4×observation ≤ 8192`。
- 追问①：耗尽为什么返回 200 closeout 而不是 429 或失败？（一个失败调用作废全 episode——见 Q1.2；closeout 保留前几轮合法证据）
- 追问②：为什么 `can_serve` 检查必须在请求引擎**之前**？（之后 = 已写失败 evidence 行，顺序不可逆）
- 评分点：refusal 返回空 content、无 tool_calls、**无 evidence row**，原因写 `engine-closeout.jsonl`。

**Q2.8 数据行 schema：框架看到什么，我们看到什么？**
- 追问①：`agent_name` 在哪一层？（顶层——纠正常见错误答案）
- 追问②：`extra_info` 为什么必须**不**在 manager 注入？（manager 盖的是运行时身份章，extra_info 是冻结的实验材料；manager 不该依赖隐藏外部数据）
- 评分点：verl 标准骨架 `prompt / data_source / reward_model.ground_truth / extra_info`（官方透传区承载我们的契约：task_id/task_prompt/task_slug；uid 故意不写——由框架 n-扩展开出）。

---

## 第 3 轮：token 对齐与 on-policy（最深水区）

**Q3.1 为什么 RL 必须训练在模型原生采样的 token 上？**
- 追问①：管道里共几处 token 转换？（prompt tokenize / 模型输出**原生** ids+logprobs / observation re-encode）
- 追问②：re-tokenize 错在哪一步、什么后果？（decode→re-encode 可能得到不同 id 序列/边界；behavior logprob 与训练 token 错位 → importance ratio 失义 → on-policy 语义破坏）
- 评分点：per-token 梯度只对"模型当时真的采出的那个 token"有意义。

**Q3.2 AppendOnlyTokenContext 的不变量？为什么它让 bridge 严格检查"by construction"通过？**
- 评分点：`prompt_{i+1} = prompt_i + generation_i + observation`——账本只追加原生生成 verbatim + 新观察/模板边界 re-encode，**从不重 token 化历史**；bridge（`src/integrations/verl/bridge.py`）的连续性检查因此无需放宽。
- 追问①：Pi 会"重写" assistant message（content 丢成 null、只回 tool_calls 子集）——ledger 怎么容？（`_same_assistant_turn` 只比承重字段 role+tool_calls，content 非空才比——因为 prompt 是账本本身，content 进不了 prompt）
- 追问②：`_message()` 会 in-place 解析 arguments，由此引出什么双保险设计？（`_next_suffix`（已归一）vs `next_suffix`（公开、会归一）；二次归一必炸 `invalid tool arguments`）

**Q3.3 训练序列里每个 token 的 (mask, logprob) 组合有哪几种？**
- 评分点：① 原生生成 mask=1 + 真实 logprob；② 观察与模板边界 mask=0 + 0.0 占位；③ 被 strip 的终止符恢复为上下文 mask=0 无 logprob。
- 追问：为什么观察必须 mask=0？（非策略动作、无 logprob；进损失即污染梯度）

**Q3.4 Pearson 0.9996 是什么的相关？**
- 评分点：rollout 侧上报的 per-token 行为概率 vs trainer 重算概率——**框架自己的校验指标**（`rollout_probs_diff_valid: 1`），smoke25 0.99951、smoke27 0.99957/0.99959。
- 追问①：剩下 0.0004 从哪来？（引擎 KV/精度/实现差异，不要求背，要求知道"不是恒等而是工程一致"）
- 追问②：为什么这是你最有力的单一证据？（它同时证明：原生 token 进了训练、logprob 真实、on-policy 成立——三个论断一个数）

**Q3.5 引擎的 `global_steps` 为什么是 None？你怎么绑定 policy 身份？**
- 评分点：`checkpoint_engine.backend=naive`（默认）→ `fsdp_workers.py:1735` 的 `update_weights` 丢弃 global_steps；`set_global_steps`（vllm_rollout.py:181）只在 CheckpointEngineWorker 路径（base.py:286）被走。colocated hybrid + naive 下**框架自身就无法自证**。
- 追问①：三级身份绑定是什么？（`engine_step = global_steps-1`；manager 对 `global_step_N/actor` 全文件 hash 进 `adapter_revision`；episode 指纹随之变化；gate 拒绝任何其他）
- 追问②：为什么不让 manager 先 push step 再验证 echo？（循环论证）
- 评分点：smoke27 的 gen-001 以重算 `global_step_1/actor` digest = `13cc8199…` 证实消费了 P1。

**Q3.6 legacy 路径为什么"能跑通"而框架路径早期跑不通？三个不对称。**
- 评分点：① 失败是 report vs fatal（E 轮 r2 被拒后换 r2b 重抽，框架路径直接炸 gather）；② 绑定检查只在序列化 round-trip 后存在（tuple/list 类 bug 的来源）；③ 终止规则只在框架路径。诚实句：E 的 1/4 成功有运气成分——样本从 4 到 16 才暴露真实 1/16。

---

## 第 4 轮：GRPO 与 RL 原理（市面八股 + 白板）

**Q4.1 【白板】推导 GRPO 目标函数。**
- 评分点：组相对优势 `A_i = (r_i − mean(r)) / std(r)`；baseline = 组均值替代 critic；clip 目标 + KL 到 ref。
- 追问①：全同分组（std=0）怎么办？（advantage 置 0——否则除零；等价于该组零梯度）
- 追问②：n=1 的 GRPO 能用吗？（不能：组内无对照 → 无方差 → 无信号；这正是 gate 的数学根源）
- 追问③：为什么"全 0 奖励组"对更新毫无贡献？（`r_i − mean ≡ 0`，梯度为零——所以 gate 重抽而不是硬训）

**Q4.2 GRPO 是 on-policy 还是 off-policy？**（市面真题）
- 评分点：训练目标 on-policy（数据来自当前策略、importance ratio ≈1、你的 clip_ratio 0.0），但工程上存在"步内多 attempt / 多 rank 聚合"的轻微 staleness——能分层回答即 3 分。

**Q4.3 为什么 GRPO 不需要 Critic？（腾讯混元真题）**
- 追问：没有 critic 的代价是什么？（无法做步级/状态级 credit assignment——组内所有 token 同一优势；引 DAPO/GSPO/LightningRL 的改进方向）

**Q4.4 DAPO 对 GRPO 的改进，逐条对照你的系统。**（高分题：市面八股 × 自己项目的交叉）
- 评分点：① **dynamic sampling**（组内全对/全错重采直到有方差）↔ 你的 DegenerateBatchError 重采样门，机制同源；② clip-higher ↔ 鼓励探索、防熵塌缩；③ token-level loss ↔ 你全程 mask 粒度；④ overlong reward shaping ↔ 你的三态 terminator / 截断标注。
- 追问：Dr. GRPO 修正了什么偏差？（1/std 与 1/len 归一化引入的长度/难度偏置）

**Q4.5 KL 为什么打在 action-mask token 上？打到观察 token 上会怎样？**
- 评分点：观察 token 非策略动作，对它算 KL/损失 = 对环境行为施加梯度。

**Q4.6 你的 hand-written trainer（`scripts/train_grpo_lora.py`，~300 行）与 verl 差在哪？**
- 评分点：数学同构（组相对 + clip + KL@mask）；三个可背 delta（无 critic / per-token KL to policy-v0 / 仅 action-mask）；verl 多出 FSDP2、vLLM colocate、权重同步、checkpoint、重抽编排——**"项目核心不在 trainer，在数据可信"**。

**Q4.7 `ppo_mini_batch_size` 数的是什么？（真实踩坑）**
- 评分点：数 **prompts** 不是扩后样本——配错即 batch 语义错位。

**Q4.8 verl 的 reward step 在你的 run 里耗时 4.77e-05 秒，为什么？**
- 评分点：verifier 在 loop 内出分（episode 结束即评分），框架侧 reward 退化为 no-op——这本身就是"分数来自真实执行证据"的证明。

---

## 第 5 轮：训练基建（FSDP / vLLM / Ray / 运维事故）

**Q5.1 FSDP1 → FSDP2 你踩了什么坑？**
- 评分点：FSDP1 单 rank 持 81.5 GiB OOM → `fully_shard` + `wrap_policy.transformer_layer_cls_to_wrap`（`PeftModel._no_split_modules` 是 set，`apply_fsdp2` 无法下标）。
- 追问：LoRA-only 34.4M 可训、14.7B 冻结，显存大头在哪？（激活 + 优化器 + vLLM KV，不是可训参数）

**Q5.2 一个默认值炸掉 host 内存：`model_dtype` fp32。**
- 评分点：`fsdp_workers.py:387` 直接把 dtype 喂 `from_pretrained` → 每 rank 物化 55.8 GiB fp32（bf16 0.8 GiB，权重保持可回收 mmap 视图）；Ray 读的是 `smaps_rollup` 不是 VmRSS。
- 追问：为什么同一问题在 CPU probe 里复现而不烧 GPU？（先测量后改动——probe 文化）

**Q5.3 权重同步链路：actor → vLLM 发生了什么？**
- 评分点：colocated hybrid；`load_format=dummy`（默认）导致 pre-rollout 同步把整个 base gather 到 CPU（fsdp_workers.py:742），resume 时再 gather 一次（:786）→ 改 `safetensors` + `layered_summon=true`（fsdp_utils.py:591，逐层搬 + 清 cache；并钉住 sleep_level=1 offload 而非丢弃）。
- 追问：`update_weights` 实测 2.90s，为什么这么快？（LoRA 增量 + 层级流水）

**Q5.4 DTensor 直接 `save_pretrained` 为什么崩？**
- 评分点：safetensors 走 data_ptr 对分片 DTensor 无效 → `full_tensor()` gather 后再存；P1 hash 与单卡冒烟一致是跨 run 佐证。

**Q5.5 vLLM colocate 的显存划分与一个平台坑。**
- 评分点：`gpu_memory_utilization=0.30`、TP=2、CUDA graph；SM 12.0 上 vendored `custom_all_reduce` kernel fault → `VLLM_BATCH_INVARIANT=1` 是官方开关。
- 追问：为什么两卡对称轮转而不是"一张专职采样一张专职训练"？（对称部署 + 同步语义；单跑跑过双卡未验过的设计要诚实标注）

**Q5.6 Ray 角色与并发模型。**
- 追问①：async 指什么、不指什么？（步内 run 级 asyncio+vLLM continuous batching；**step 严格串行**——并行 step = 采样自旧权重，违反 on-policy）
- 追问②：AgentLoopWorker 数量是什么旋钮？group 跟谁走？（吞吐旋钮；group 跟 uid 不跟 placement）
- 追问③：一步的上限并发是多少？（batch×n 个 Pi 子进程，不叠加）

**Q5.7 【事故题】训练 18 小时后进程消失：无 Traceback、无 EXIT=、tmux 会话没了。诊断顺序？**
- 评分点：签名识别——raylet "over 95% full" 警告 + save 前 progress 停滞 + `df /` 100% + `dmesg` 净 + RAM 正常 → **磁盘满**而非代码 bug；echo 自身都写不进去所以连 EXIT= 都没有。
- 追问①：恢复动作的边界？（只动自己的 run 目录：剪自己旧 ckpt 释放 464GB → **byte-identical 命令重跑**，`resume_mode=auto` 恢复 dataloader 状态，7 步 ~24 分钟补完）
- 追问②：**为什么 `save_freq` 不能调大省磁盘？**（批 gate 每次 rollout 后读 `global_step_{engine_step}/actor` 并全文件 hash；稀疏保存 → step 2 直接 `synchronous checkpoint missing`。r3 用一次 eval 崩溃换来的教训）
- 追问③：磁盘安全的唯一杠杆？（`trainer.max_actor_ckpt_to_keep`：训练=2 ~58GB 稳态 / 评测=1 ~29GB；gate 只读最新 ckpt 所以兼容）
- 追问④：哪个 checkpoint 永远不能删？（eval adapter **symlink 进**的那个 `global_step_300/actor/lora_adapter`）

**Q5.8 【事故题】共享 GPU 被邻居任务杀了三次训练，怎么办？**
- 评分点：排他性技术上不可强制 → gate 在争用窗口内**正确地**拒批（70% episode 死在 call 0 → 20 次重抽耗尽 → fail-closed）→ 恢复 = 等卡空 + byte-identical resume；绝不全 GPU pkill（按 owner/cmdline 选自己的 PID）。
- 追问：为什么"blind retry"在邻居占卡时无用？（episode 系统性超时 → INFRA_INVALID → gate 必拒——重试烧的是时间不是概率）

**Q5.9 vLLM 八股（市面真题，阿里面试官风格）。**
- PagedAttention 为什么快？类比 OS 什么机制？（分页管理 KV cache，消碎片提利用率）
- continuous batching vs static batching？（请求级动态进出，消 GPU 空转）
- chunked prefill / prefix caching / 投机解码 / KV cache 量化各解决什么？
- 追问（结合项目）：你的多轮 agent 为什么吃 prefix cache 的红利？（组内同题 n 条 rollout 共享前缀）

**Q5.10 LoRA 八股。**
- 追问①：你的配置？（rank 8 / alpha 16 / dropout 0 / q,k,v,o,gate,up,down 七投影 / 34.4M 可训）
- 追问②：alpha/rank 的关系；为什么 dropout=0 对 RL 合理？

---

## 第 6 轮：实验设计与统计（负面结果防御）

**Q6.1 RL 没有显著提升，实验价值是什么？**
- 评分点：机制证明与效果证明**分离**；power 分析（n=400 → CI ±4.6pp，对 +8–10pp 效应 power≈80%）；双轨归因（同一清洗数据 instruct 持平 / base loss 0.895→0.787、BCB-Hard +2.7pp）→ "起点饱和而非数据无效"。
- 追问①：Fisher 精确检验为什么选它？（2×2 小样本计数：16/48 vs 14/48，p=0.826/1.000——**不显著 ≠ 证明无效应**）
- 追问②：样本从 48 到 400 改变了什么？（检测能力，不是效应存在性——campaign 的目的就是给检验以 power）
- 追问③：holdout 为什么 first-attempt-only + lr=0？（评的是策略真实一次通过能力，不是采样预算；lr=0 保证 eval 不改权重）

**Q6.2 任务筛选规则：为什么只选 base=pass 的任务？**
- 评分点：flat-batch death rule——p≈0/1 的任务耗尽 `max_attempts` → 该步永久卡死且 resume 重放同一任务；evalplus 难度分层（pass/pass 247、pass/fail 54、fail/fail 76、fail/pass 1）+ 均匀间隔确定性填充；r3 实测 2 个 0/8 任务在第 20 次重抽内恢复——规则有效。
- 追问：这和 DAPO dynamic sampling 的关系？（同源问题：GRPO 只能从组内对比学习；一个在数据侧筛，一个在采样侧重抽）

**Q6.3 r3 的训练侧学习曲线说明什么？（0.504→0.648）**
- 追问：为什么训练侧曲线上升不能宣称泛化？（同任务 epoch 重访 + 重抽偏差；唯一独立证据是 holdout）
- 追问：r3 中断在 276/300，科学上损失多大？（24 步是 epoch-5 重访，方差小于信息——如实说，别夸大损失也别假装没损失）

**Q6.4 instruct 时代的负面结果你怎么讲？**
- 评分点：数字全对（APPS holdout 48/50/50；HumanEval 87.20/89.02/87.20；BCB-Hard 25.00/22.97/20.95）+ 机制归因（分布饱和 / 低信息修复性更新 / KL 拉回 / 稀疏奖励）——**解释机制，不甩锅资源**。

**Q6.5 如果 r3 跑完仍不显著，下一步？**
- 好答案方向：加步数不是杠杆（r2 已证 24 步无信号）；真正的杠杆是任务多样性/更强 base/奖励设计与课程；以及承认"以当前 base 在 MBPP 上接近 ceiling，效应空间本身可能只有几 pp"。

---

## 第 7 轮：压力测试与行为题

**Q7.1 GPU 预算超了 4 小时，你怎么处理？**（真实发生）
- 评分点：如实记录进 `acceptance.json:budget_hours_used`，不悄悄清零；超支由用户明确指令授权——"报告 vs 擅自"是这题的考点。

**Q7.2 项目里最大的失误是什么？**
- 好答案：fake retry loop（假重试存活到第一次真正到达失败路径才暴露）——教训："只在失败路径执行的代码，需要专门把它推到失败路径来测"（replay gate 工具）。

**Q7.3 重做一遍，你会先改什么？**
- 好答案：**先扩任务集再做框架接入**——方差/信号是一阶约束（1 行数据集 × 1/16 解题率让每一轮 GPU 都在赌 22%），管道是二阶。

**Q7.4 和 Polar/Slime 的对比？为什么不算重复造轮子？**
- 评分点：Polar=在线 rollout 服务，本项目=数据信任平面（互补）；Slime 在 backlog；"token-faithful envelope 已有，差的只是一层映射"这类话只说做过验证的部分。

**Q7.5 反问环节（准备 2 个真问题）。**
- 建议：团队的多轮 agent RL 管道里，数据认证/质量这道工序由谁拥有？rollout 与训练的权重同步延迟在你们的规模下怎么处理？

---

## 第 8 轮：项目之外的必答题（市面真题池，答不出锅外）

> 这些题在项目之外，但同一场面试大概率出现。每题给"你项目的挂点"——用项目经验答出差异化。

**8.1 RL 算法组**
- PPO 四模型角色（Actor/Critic/RM/Ref）各干什么？verl 一个 step 的完整流程？（gen_batch → old_log_prob → ref_policy → rewards → advantage → actor/critic 更新 → 权重同步）【挂点：你的 run 里 reward 是 no-op、critic 无、多轮 rollout 换成 Pi】
- DPO 损失推导；DPO 与 RLHF/PPO 的关系与失效场景。【挂点：你的 preference 视图一 CHOSEN 一 REJECTED、按 logical task 切分防泄漏】
- 奖励模型输出尺度不稳定会引发什么？奖励 hacking 举三例。【挂点：你的 verifier 是可执行断言，不给模型打分的 RM——天然免疫一类 hacking】
- GRPO 为什么被 DeepSeek-R1 选择？GSPO 序列级优势解决什么？

**8.2 verl / AgentLoop 架构**
- AgentLoop 在 verl 里解决什么问题？（连接 Agent/环境框架与 RL 训练框架的桥梁：多轮、工具调用、loss_mask 轨迹 → Sample）【挂点：你就是沿这条接口接入的，四个扩展点如数家珍】
- hybrid engine 怎么做训练↔推理权重互转？FSDP/Megatron × vLLM/SGLang 组合边界。
- RFC 黑盒 agent 训练（OpenAI 兼容接口零改造）与你的 capture proxy 思路对比。【挂点：你的 fake-OpenAI 代理（隔离 HOME + models.json + 一次性 token → 转发框架 vLLM 并记录 evidence）就是一个更严格的同型方案】

**8.3 推理 infra**
- vLLM 十连（PagedAttention/continuous batching/chunked prefill/抢占 recompute vs swap/prefix caching/投机解码/KV 量化/并行策略/CUDA graph/为什么 GPU 利用率高）。
- TP=2 时 custom all-reduce 的作用域（你真实踩过 SM12.0 的 kernel fault）。

**8.4 Agentic RL 工程**
- 为什么做 agentic RL 而不是成功轨迹 SFT？（只强化最终成功会把无效中间路径一起强化；RL 需要探索与对比信号）【挂点：你的 1/16 → 组内对比 → 方差门，就是这题的第一手证据】
- 多轮轨迹 loss mask 的完整规则与边界处理（trailing token、tool 输出、模板边界）【挂点：三种 (mask, logprob) 组合】
- credit assignment：trajectory-level vs turn-level vs token-level；LightningRL 的 transition 分解思路。【挂点：你现在 episode 级奖励 + token 级 mask——劣势与改进方向都能讲】
- rollout 终止的预算与原因标注（max turns / token budget / EOS / 工具循环）【挂点：三态 terminator + engine-closeout 是市面少见的完整实现】
- staleness 控制：async rollout 的版本一致性。【挂点：你 steps 严格串行的 on-policy 论证】

---

## 附录 A：必背数字（答错任何一个 = 该轮减分）

| 主题 | 数字 |
|---|---|
| on-policy 一致性 | `rollout_actor_probs_pearson_corr` 0.99951 / 0.99957 / 0.99959；`rollout_probs_diff_valid: 1` |
| 框架 step | smoke27：2 步 22m34s；`update_weights` 2.90s/2.83s；`update_actor` ~187s；`clip_ratio 0.0` |
| 正式 r2 | 24/24 步；holdout 16/48=33.33% vs P0 29.17% / P12 31.25%；Fisher p=0.826/1.000（n=48） |
| r3（进行中） | 60 任务×5 epoch=300 步；holdout 400 样本 P24 基线 178/400=44.50%；~204s/步；276/300 中断 |
| 信号率 | Mbpp/118 实测 ~1/16 → n=4/16/32 认证概率 ≈22%/60%/87% |
| 训练量级 | 14B base + LoRA rank8/alpha16，可训 34.4M / 冻结 14.7B；FSDP2 bf16 每 rank ~22GB 峰值 |
| 基建 | 302 passed / 9 skipped；每步全模型 ckpt ~29GB；`max_actor_ckpt_to_keep` 训练 2 / 评测 1 |
| checkpoint 指纹 | P0 `6db6a40c…` / P1 `15a741e2…` / P2 `30e3f3ad…` / P24 `40f20ea1…` |
| 效果（诚实区） | instruct：APPS holdout 48/50/50，HumanEval 87.20/89.02/87.20；base SFT：BCB-Hard 19.59%→22.3% |

## 附录 B：一条压轴叙事（2 分钟，全部数字内嵌）

"我把一个真实的多轮代码 Agent 接进了 verl 的 GRPO 训练循环——只走四个官方扩展点，训练循环零改动。链路是否真的 on-policy，我不自己说了算：框架重算的 logprob 和 rollout 上报的相关是 0.9996，第二步 rollout 的 adapter 指纹等于对框架自己写出的 checkpoint 重算的 digest。信号问题上我踩过真坑：GRPO 组内零方差就没有梯度，我的 gate 按'能否被重抽样改变'分层失败语义，把 n 从 4 提到 32 让认证概率过 99%。效果我如实报告：24 步正式训练后固定 holdout 33.3% 对基线 29.2%，Fisher p=0.826 不显著——所以 r3 把规模扩到 300 步、holdout 400 样本，让检验有 power 检出 8–10pp 的真实效应。这个项目证明的是机制，追求的是效果，两者我从没混在一起说。"

## 附录 C：题源链接

**RL 算法八股**
- [大模型 RL 方向面试题 90 道（CSDN）](https://blog.csdn.net/u012374012/article/details/148223023)
- [大模型工程面试经典：对比 PPO 和 GRPO 核心原理（知乎）](https://zhuanlan.zhihu.com/p/1954158663255195996)
- [LLM 算法岗八股问答：强化学习与 RLHF（博客园）](https://www.cnblogs.com/moonout/p/19749191)
- [大模型面试题剖析：PPO 与 GRPO 核心差异（掘金）](https://juejin.cn/post/7544008774816595995)
- [RL 面试相关问题：GRPO 是 on-policy 还是 off-policy（知乎）](https://zhuanlan.zhihu.com/p/1948681769332240910)
- [腾讯混元秋招 1 面：GRPO 比 PPO 好在哪（知乎）](https://zhuanlan.zhihu.com/p/1947406577486263261)
- [大厂推荐算法面经 10 问：DPO/GRPO/DAPO（牛客）](https://www.nowcoder.com/feed/main/detail/65407d13624942d0ace275771b918826)
- [大模型面试必备：PPO、DPO、GRPO、DAPO 与 GSPO（CSDN）](https://agent.csdn.net/6a699c3310ee7a33f293e627.html)

**verl / AgentLoop 架构**
- [VeRL 框架入门 & 代码带读（知乎）](https://zhuanlan.zhihu.com/p/27676081245)
- [verl PPO 分布式训练框架实战指南（CSDN）](https://adg.csdn.net/696f42d7437a6b403369cc6f.html)
- [verl Agent Loop 官方文档](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)
- [verl 的 AgentLoop 解决的问题（知乎）](https://zhuanlan.zhihu.com/p/1977385816742461997)
- [AgentLoop 源码浅析（Awesome-ML-SYS-Tutorial）](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme-6.md)
- [verl RFC #5790：黑盒 Agent 训练框架（知乎）](https://zhuanlan.zhihu.com/p/2033193771538583961)
- [蚂蚁金服一二面面经：verl 训练流程真题（牛客）](https://www.nowcoder.com/feed/main/detail/c1f9b7cec4eb4d1ea27f86416daa89f0)

**vLLM / 推理 infra**
- [AI Infra 面试常考：vLLM 推理框架（知乎）](https://zhuanlan.zhihu.com/p/2011083570035319972)
- [阿里面试官问：为什么 vLLM 能加快推理速度（CSDN）](https://blog.csdn.net/m0_59163425/article/details/144444514)
- [vLLM 原理详解：面试官常问的 10 个问题（博客园）](https://www.cnblogs.com/ljbguanli/p/18933064)
- [大厂大模型算法岗推理类面试题总结（牛客）](https://www.nowcoder.com/feed/main/detail/68262da0086c49dfad1848931306b17d)

**Agentic RL 工程**
- [Agentic RL 八股（GitHub 面试题库）](https://github.com/MarsChange/Learning-For-LLM-Intern/blob/main/%E5%90%8E%E8%AE%AD%E7%BB%83/RL%20%E5%9F%BA%E7%A1%80/Agentic%20RL%20%E5%85%AB%E8%82%A1.md)
- [Agentic RL 系列：环境、轨迹、Reward 与训练闭环（知乎）](https://zhuanlan.zhihu.com/p/2070106530058392520)
- [From Reasoning to Agentic: Credit Assignment in RL for LLMs（arXiv, LightningRL）](https://arxiv.org/html/2604.09459v2)
- [面试问答式：为什么做 Agentic RL 而不是成功轨迹 SFT（掘金）](https://juejin.cn/post/7683433296757293110)
- [Agent-R1 v2：多轮 Agentic RL 与 Loss Mask（智猩猩）](https://aiorang.com/article/1nmOhMc.html)
- [多轮对话转训练样本的 loss mask 实现（博客园）](https://www.cnblogs.com/rossiXYZ/p/20524400)
