# Agentic-RL 接入 verl 实施计划与验收约束

本文是交给编码 agent 的最终执行规格。规划日期：2026-09-09。本文保存时仅完成只读调查与方案设计，尚未为本计划安装环境或执行 GPU 验证。

## 1. 最终决策与交付目标

### 1.1 只接入 verl

选择：

**verl v0.7.1 + FSDP2 + PEFT LoRA + vLLM + 真实 Pi Harness。**

采用同步训练闭环，使用框架原生双卡共置分时。异步 Agent Loop 仅表示并发执行 rollout，不表示异步更新策略。

最终验收必须使用本次 **Qwen2.5-Coder-14B Base 的 SFT checkpoint**。小模型仅能用于环境诊断，不能替代最终验收。

| 比较项 | verl | Slime | 本项目判断 |
|---|---|---|---|
| 接入真实 Harness | 自定义 Agent Loop | 自定义 rollout/generate 接口 | 两者均具备扩展边界 |
| 当前 PEFT adapter 续训 | 指定版本有 FSDP/FSDP2 + vLLM LoRA 接口 | 服务器现有 v0.3.0 只有 Megatron 训练后端 | verl 更贴合当前 14B SFT 产物 |
| 环境复用 | 需要新建隔离环境 | 已有旧环境，但不等于支持所需训练路径 | 不以“已经安装”代替兼容性证明 |
| 现有项目模块 | 需新增 adapter | 已有 admission 数据契约 | admission JSON 不等于已接入 Slime trainer |
| 双卡有限窗口 | 使用原生共置和权重同步 | 支持分离及共置部署 | 不为固定分卡另造调度 |
| 本次新增工作 | Harness bridge、轨迹适配、认证入口 | 还需解决旧栈与当前 adapter 的训练路径 | 选择 verl |

以上选择针对本项目的模型、产物和收尾目标，不是对两个框架的一般排名。verl v0.7.1 明确支持加载已有 PEFT adapter，并通过 vLLM rollout；Polar 官方示例桥接 Slime，并不能证明我们的 admission 已经接通 Slime。[verl LoRA 接口](https://github.com/verl-project/verl/blob/v0.7.1/docs/advance/ppo_lora.rst)、[Polar Slime bridge](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server/blob/stable/src/slime_bridge/README.md)

### 1.2 完成标准

必须留下以下连续证据：

```text
当前 SFT adapter P0
  → verl 管理的 vLLM 使用 P0
  → Pi 执行真实工具调用及后续决策
  → 本项目捕获、组装、认证轨迹
  → verl 完成有效参数更新，保存 P1
  → verl 同步 P1
  → Pi 使用 P1 完成新的 rollout
  → verl 再次更新并保存 P2
  → 固定小评测集比较 P0 与 P2
```

“接入成功”和“效果提升”分别报告。效果没有提升不自动否定接入，但参数没有有效更新、策略未同步、轨迹认证被绕过，均不得宣布完成。

## 2. 已确认事实、资源与禁止事项

### 2.1 当前事实

2026-09-09 只读检查确认：

- 服务器：`cxr@172.17.43.193`。
- 两张 RTX PRO 6000 Blackwell Server Edition，每张约 96 GB。
- 驱动版本：`580.159.03`。
- 主机内存约 251 GiB。
- `/data` 可用空间约 2.5 TB；系统盘仅约 200 GB。
- 当前 Base SFT 仍在执行，训练目录尚不能视为冻结输入。
- 已观察到的 adapter 配置：rank 8、alpha 16、dropout 0，覆盖 attention 与 MLP 的七类 projection。
- 两张 GPU 均存在其他任务；即使其 Unix 用户也是 `cxr`，也不代表属于本项目。
- 旧 Slime 环境仍在，但不是本次执行环境。
- 当前工作区有大量未提交修改和删除，不能恢复到历史版本后覆盖开发。

这些是规划时快照。执行每个 GPU 阶段前必须刷新状态。

### 2.2 资源安排

- GPU 累计预约占用时间上限：**4 小时**，双卡同时运行 1 小时计为 1 小时窗口、2 GPU-hours。
- 预算从开始占用 GPU 起计算，包含模型加载、失败重试和清理。
- CPU 开发、单元测试、依赖准备提前完成。
- GPU 必须在 SFT 结束、checkpoint 冻结且使用窗口明确后启动。
- 新环境、缓存和运行产物放在 `/data` 下本任务独立目录。
- 不升级或修改现有 SFT 环境、旧 Slime/Polar 环境、系统驱动、CUDA 或 Docker daemon。
- 每个阶段使用独立运行 ID，保留命令、配置、日志和退出码。

### 2.3 编码 agent 的硬限制

1. 只实现 verl 接入；禁止同时接 Slime、Polar 或第三个框架。
2. 禁止复制或修改 verl 的优化器、GRPO、advantage、FSDP 和权重同步实现。
3. 禁止调用项目自有 `train_grpo_lora.py` 代替 verl 验收。
4. 禁止用直接 `model.generate()` 的 APPS 路径代替 Pi 验收。
5. 禁止用 Teacher 历史轨迹冒充当前策略 rollout。
6. 禁止修改 Pi 核心 Agent Loop；只允许配置、启动封装和模型协议适配。
7. 禁止关闭认证、吞掉对齐错误、填造 logprob、reward 或策略指纹。
8. 禁止把 mock 成功、文件存在、loss 非零当作真实闭环成功。
9. 禁止为了获得正向结果修改固定评测集、奖励函数或训练后重新挑选任务。
10. 禁止执行 `git reset --hard`、全局清理、广泛 `pkill`、停止共享 Ray 或清空 GPU。
11. 只清理本次运行登记的 PID、进程组和容器。
12. 遇到框架或内核不兼容，不得自行换框架、换模型、修改上游内核或引入量化补救。

不满足阶段验收时，提交阻塞报告，不得把后续阶段标成完成。

## 3. 接入架构与接口边界

### 3.1 数据与控制流

```text
verl RayPPOTrainer
  │ 原生训练循环、参数更新、checkpoint、权重同步
  ▼
CertifiedAgentLoopManager
  │ 注入本轮策略身份，校验完整 batch
  ▼
PiAgentLoop
  │ 启动真实 Pi，等待执行和 verifier
  ▼
Pi → 本项目模型 bridge → verl AsyncLLMServerManager → vLLM
  │                           │
  │ 工具事件                  │ 实际输入/输出 token、logprob
  └──────────────┬────────────┘
                 ▼
      Episode / ExecutionBundle / Certification
                 ▼
             AgentLoopOutput
                 ▼
           verl 原生 GRPO 更新
```

使用 verl v0.7.1 已有的自定义 Agent Loop 配置和自定义 Manager 扩展点，不复制其训练循环。[Agent Loop 类型与接口](https://github.com/verl-project/verl/blob/v0.7.1/verl/experimental/agent_loop/agent_loop.py)、[自定义 Manager 接入位置](https://github.com/verl-project/verl/blob/v0.7.1/verl/trainer/ppo/ray_trainer.py)

### 3.2 模块职责

新增实现集中在 `src/integrations/verl/`，分清以下职责：

| 模块 | 负责 | 禁止负责 |
|---|---|---|
| `PiAgentLoop` | 一条 rollout 的生命周期、Pi 启动、verifier、结果返回 | 工具决策、训练更新 |
| 模型 bridge | Pi 协议适配、调用 verl token 接口、保存模型证据 | 自己启动另一套模型服务、改变奖励 |
| 轨迹转换器 | 组装 token/mask/logprob，验证上下文一致性 | 根据最终文本重建生成 token |
| admission | 复用本项目认证，拒绝不合格轨迹 | 为框架凑 batch 而放行 |
| Manager | 策略身份、分组、batch 认证和审计产物 | 自己计算 advantage 或同步权重 |

复用现有 contracts、capture、assembly、certification、policy fingerprint、APPS verifier 和 Pi 事件解析能力。

对 `PiHostExecutionOrchestrator` 只做必要的依赖注入和异常清理改动：

- 允许配置每次执行的模型 bridge 地址。
- 允许注入已验证的策略身份和最终训练轨迹。
- 修复接入必需的 verifier timeout、进程取消和证据落盘问题。
- 保持现有调用入口和默认行为兼容。

当前 Pi policy artifact 缺少完整训练上下文，并存在 `action_mask` 与 admission 所读 `loss_mask` 的命名差异。新路径必须显式转换和验证，不能因 capability 标志存在就判定可训练。

不迁移整个旧 Slime 模块，不顺带重构全项目。

### 3.3 对外输入和输出

配置必须明确提供：

- 冻结的 base model、SFT adapter、tokenizer/chat template。
- APPS manifest 与固定任务 ID。
- GPU UUID、运行根目录、使用窗口截止时间。
- Pi 版本、工具集合、采样参数、资源限制。
- upstream lock 与代码版本。

`AgentLoopOutput` 至少返回：

- `prompt_ids`
- `response_ids`
- `response_mask`
- `response_logprobs`
- `reward_score`
- `num_turns`
- `metrics`
- `extra_fields` 中的 episode、bundle、认证、group、策略身份引用

输出直接来自本轮不可变产物，不能同时维护一套未经校验的“训练专用结果”。

### 3.4 最关键的 token 契约

采用 **单 episode 对应一条训练序列**，本次只支持可保持生成历史的顺序多轮调用。

要求：

1. 第一次模型请求由冻结 tokenizer/template 生成 prompt token，并将这组 token 原样送给 verl。
2. 每轮保存真正送入推理的完整 prompt token、原生生成 token、原生 logprob。
3. 已生成 token 永久保留；后续工具结果和模板分隔符作为上下文追加。
4. 输出 token 的 mask 为 1；工具观察、模板补充和 padding 的 mask 为 0。
5. 观察位置的 logprob 可用 0 作张量占位，但必须明确不是模型概率，并保证所有损失和统计都按 mask 排除。
6. Pi 的旧消息变化必须被检测；不得把压缩、删改历史后的请求硬接到旧序列。
7. 初版不支持 context compaction、多 agent 分支、回滚历史或多模态；遇到这些行为直接拒绝。
8. 文本解码仅用于向 Pi 返回响应、解析工具协议，不能回写替换原始训练 token。
9. 复用上游 Qwen/Hermes 工具解析器完成协议转换，不自行修复模型生成的非法 JSON。
10. 截断、超长或请求预算耗尽时不伪造 assistant 结束消息，不允许用不完整轨迹更新。

执行前用真实两轮工具调用验证模板边界。如果无法在不改变生成历史的情况下构造下一轮请求，状态为 `BLOCKED_TOKEN_CONTEXT`，禁止退回“最终 messages 重新 tokenize”。

### 3.5 分组、奖励和策略身份

- 一个 task 的四次独立 Pi 执行为一个 GRPO group。
- episode ID 必须唯一；不能把一次执行里的四个模型调用当成四个样本。
- 保持 verl 原有 `uid` 分组语义；group 审计身份同时包含本轮策略。
- 奖励固定为 APPS verifier 的全测试通过 `1`、有效失败 `0`。
- 模型生成的程序错误可为正常失败；manifest 损坏、verifier 启动失败、服务中断和证据缺失属于基础设施错误。
- 正式训练出现任一无效样本，整批停止，不静默丢样本、补零或缩小 group。
- 新策略身份绑定 base、adapter/checkpoint 内容、tokenizer、template、工具 schema 和采样参数。
- 保存每轮 checkpoint 与上游同步调用完成的证据；单独递增 `policy_generation` 不算同步证明。
- 每次更新只使用当前批一次，禁止跨版本重用 rollout。
- 使用原生同步训练顺序；不得在一条 Pi episode 中途更新模型。

## 4. 执行阶段与逐阶段验收

### 阶段 A：冻结输入、环境与执行基线

**不占用 GPU。**

1. 保存当前 Git 状态、差异摘要和现有测试基线，保护已有修改。
2. 等 SFT 进程正常退出，核对最终训练记录与 adapter 文件。
3. 从本次训练的最终输出记录解析 checkpoint；不使用模糊的 `latest`，不默认使用目前仅存在的 `epoch0`。
4. 创建只读输入 manifest，记录权重、adapter、tokenizer、chat template 和 SFT 数据包校验和。
5. 抽查 SFT token 包与当前 tokenizer 的一致性。Base tokenizer 缺少 template 时，只能使用数据包可追溯的训练 template；不能随手拿别的 Instruct template 替换。
6. 将 verl 固定到 `v0.7.1` 对应 commit。
7. 新建独立运行环境。候选基线采用该 tag 官方 Dockerfile 对应的 Python 3.12、CUDA 12.9、PyTorch 2.10.0、vLLM 0.17.0；只安装 FSDP 路径所需依赖。
8. 先做依赖解析和 CPU import 检查，再生成完整 lock。PEFT 必须可读取当前 0.19.1 产出的 adapter 配置。
9. 禁止执行带未固定 `latest/main` 的启动配置；解析出的实际版本与镜像 digest 写入 lock 后使用。
10. 检查 Pi 0.84.2 的真实可执行文件及安装来源；若不存在，在任务私有环境安装该固定版本，不替换系统 Pi。

该候选环境参考官方文件，但尚未在本机验证；不能把 B200 支持直接当作 RTX PRO 6000 的内核兼容证明。[固定版本环境参考](https://github.com/verl-project/verl/blob/v0.7.1/docker/Dockerfile.stable.vllm)

**通过条件：** 冻结模型可追溯、依赖解析成功、配置可加载、GPU 启动命令明确且未执行。

### 阶段 B：CPU 开发与真实类型契约测试

**不占用 GPU。**

完成 bridge、转换器、admission、Manager、生命周期清理和配置入口。

必须测试：

- 正常的两轮工具调用。
- 原生生成 token 与解码再编码结果不一致时，仍保留原生 token。
- 工具观察只进上下文，不进入策略 loss。
- 工具调用与结果缺失、重复、错配。
- 混入其他策略版本或其他 episode。
- token/mask/logprob 长度不一致、非有限数值。
- reward-zero 正常失败与基础设施错误的区分。
- 模型请求、Pi、verifier 超时及取消后的清理。
- 四次独立执行的 group；禁止重复 episode 充数。
- `AgentLoopOutput` 经真实 verl 后处理得到的 mask、reward、uid 和元数据未丢失。
- 缺少 verl 时，项目原有核心模块和 CLI 仍可导入。

测试可模拟推理响应，但必须使用已锁版本的真实 verl 类型和处理接口。模拟产物须标记，不能复用为 GPU 验收产物。

**通过条件：** 新增测试和受影响既有测试通过，完整基线没有新增回归。

### 阶段 C：14B 框架环境验证

**GPU 预算：最多 60 分钟。**

先不接 Pi，验证外部框架本身：

- 当前 14B Base + 冻结 SFT adapter 能在 verl 中加载。
- vLLM 正确使用该 adapter。
- FSDP2 只训练 LoRA 参数。
- 实际执行一次原生 optimizer update，保存 checkpoint。
- 原生权重同步后再次推理。
- 保存训练参数计数、显存峰值、checkpoint 数值变化和同步日志。

这个阶段允许使用小型确定性诊断任务，但产物必须标记 `framework-smoke`，不得视为项目接入验收。

固定初始设置：

| 项目 | 设置 |
|---|---|
| GPU | 两卡共置分时 |
| 模型精度 | BF16 |
| FSDP | 双卡分片 |
| vLLM TP | 1，每个副本一张卡 |
| LoRA | 加载已有 rank 8 / alpha 16 配置 |
| 训练 micro batch | 每 GPU 1 |
| Gradient checkpointing | 开启 |
| Attention | 训练侧优先原生 SDPA，关闭 remove-padding 路径 |
| Actor 参数/优化器 offload | 开启 |
| vLLM 显存比例 | 0.40 |
| 额外模型 | 不启用 critic、reward model；初次验收 KL 系数为 0 |
| 精度补救 | 不启用 FP8、QLoRA 或新量化 |

OOM 只允许一次受控重试：减小序列预算，并将 vLLM 显存比例降至 0.30。记录差异；仍失败立即停止，不做无限调参。

**通过条件：** 有数值变化的 adapter、真实更新和加载证据。框架退出成功本身不够。

### 阶段 D：真实 Pi rollout 与认证

**GPU 预算：最多 60 分钟。**

1. 使用现有 APPS stdin/stdout 任务，不加入 SWE-bench。
2. 任务工作区只暴露题目、初始 `solution.py` 和公开示例。
3. Pi 使用 `read/bash/write/edit/ls`，禁用个人扩展、skills、持久会话和自动压缩。
4. 每次执行独立工作区；Pi 运行在 CPU sandbox 中，不能读取宿主密钥、训练 checkpoint、完整 APPS 答案或 Docker socket。
5. verifier 在另一个干净执行环境中运行，只接收生成的 solution 和只读测试输入。
6. 每条正式合格轨迹必须至少包含一次实际工具调用、对应结果及后续模型请求。
7. 保留原始 Pi NDJSON、模型请求证据、工具事件、verifier 输出、episode、bundle、认证决定。

固定资源上限：

- 同时最多 2 个 Pi episode。
- 每个 episode 最多 8 次模型请求、8 分钟。
- 总训练上下文初始上限 8192 token：初始 prompt 不超过 4096，后续生成与观察合计不超过 4096。
- 每次生成最多 1024 token，并受剩余预算限制。
- 采样 temperature 1、top-p 1，禁用额外 top-k 截断，固定随机种子。
- 不为了凑够上下文预算截断真实证据；超限直接标记失败。

从训练任务池中固定选择最多 4 个候选任务，每题 4 条轨迹做诊断。优先使用已有训练任务记录，按任务 ID 确定排序；排除评测任务。

允许根据这批训练诊断结果选取最多 2 个具有组内奖励差异的任务用于正式两轮更新，但必须保存完整候选结果及选择规则。这属于训练选择，不能报告为效果评测。

若全部失败、全部成功或工具调用不成立，记录 `BLOCKED_NO_LEARNING_SIGNAL` 或 `BLOCKED_HARNESS_BEHAVIOR`；不得造奖励差异。

**通过条件：** 至少一个完整 group 通过真实多轮轨迹认证，并产生可用于 GRPO 的组内奖励差异。

### 阶段 E：正式两轮闭环

**GPU 预算：最多 90 分钟。**

使用阶段 D 冻结的训练任务重新采样，不能重用诊断轨迹。

- 每轮 1–2 个 task，每题 4 条独立 episode。
- 由任务数自动确定 batch，必须满足双卡和 GRPO 分组整除条件。
- 学习率固定 `1e-6`，每批一个更新轮次，使用 verl 原生 GRPO/PPO-clip。
- 每轮保存同步完成的 checkpoint，禁用异步 checkpoint。
- 更新前检查整批认证、策略一致性和非零组内奖励方差。
- 保存 P0→P1、P1→P2 的 adapter 数值差异、梯度与上游训练指标。
- P1/P2 都必须通过真实 Pi 的后续执行证明可用。
- 在第一次更新后完成一次原生 checkpoint 边界的恢复验证；不重放已消费 batch，不恢复半条 episode。

对更新前相同 token 的训练侧 logprob 与 rollout logprob 做数值对照。初始门限：平均绝对差不超过 0.05 nat，P99 不超过 0.5 nat；完整分布入报告。超限停止定位，不能自动放宽阈值或用重算值覆盖原始记录。

**通过条件：** 两次有效更新、完整 lineage、更新后真实 rollout、恢复验证均有产物。

### 阶段 F：固定小集评测与收尾

**GPU 预算：最多 30 分钟。**

- 预先从现有 holdout 中固定 4 个未参与本次 RL 训练的任务。
- P0 与 P2 各执行一次真实 Pi rollout，使用相同工具、上下文、超时和确定性采样配置。
- 单独报告 verifier 成功率、工具调用、执行无效率、token 和耗时。
- 未核验与 SFT 数据的任务交集时，不得称为“SFT 未见集”。
- 4 题结果只作为冒烟评测，不宣称统计显著提升。
- 到期清理本次 GPU 进程和容器，留下资源释放证据。

**通过条件：** 接入结论与效果结论分开，报告链接能定位到原始证据。

## 5. 产物、报告与最终验收规则

### 5.1 必须交付的文件

除实现代码和测试外，交付：

- 固定版本和环境 lock。
- 一份可直接执行的完整配置。
- 一个分阶段入口，支持 `preflight / smoke / rollout / cycle / evaluate / report / cleanup`。
- 操作说明，覆盖冻结 checkpoint、使用窗口、执行、恢复和清理。
- `acceptance.json`：机器可读的阶段状态和证据路径。
- `acceptance.md`：人可读的结论、限制、耗时与剩余工作。
- 每阶段的命令、配置、退出码、日志及 SHA-256 manifest。
- P0/P1/P2 及 episode/group/batch 的 lineage。
- 两轮实际使用的训练 batch token、mask、reward 和认证引用。

入口默认只做 preflight；GPU 阶段必须显式指定冻结输入、GPU UUID 和截止时间，不能无参数自动占卡。

### 5.2 报告状态

只允许使用：

- `NOT_RUN`
- `PASSED`
- `FAILED`
- `BLOCKED`

阻塞时必须给出：

1. 失败阶段和最小复现命令。
2. 原始日志及关键错误。
3. 已确认事实与尚未确认假设。
4. 已尝试的受控修复。
5. 超出本文范围的下一步建议。
6. GPU 和容器是否已经释放。

### 5.3 必须全部满足才能宣布接入完成

- [ ] 使用冻结的当前 14B Base SFT checkpoint。
- [ ] 实际训练由 verl 执行。
- [ ] Pi 真正执行工具循环。
- [ ] 每轮实际推理上下文与训练序列对齐。
- [ ] 工具观察及 padding 不参与策略损失。
- [ ] 认证在训练前执行，失败不能旁路。
- [ ] group 按独立 episode 构成，策略版本一致。
- [ ] 两次更新有真实参数变化。
- [ ] 新权重经框架同步后被 Pi 使用。
- [ ] checkpoint 边界可恢复。
- [ ] 固定小集评测完成，效果结论独立。
- [ ] 运行没有遗留 GPU 服务。
- [ ] 文档宣称与本次证据一致。

### 5.4 最后才更新项目宣称

验收通过后，项目描述可以写：

> 实现了真实 Pi Harness 到 verl 的训练适配，将执行轨迹经 token 对齐、verifier 和策略身份认证后送入原生 GRPO LoRA 训练，并在双卡 RTX PRO 6000 上完成两轮参数更新及更新后再采样验证。

若只完成部分阶段，应逐项描述，不得写“完整 Agentic RL 闭环”。

本次不承诺训练效果提升，不扩展框架数量，不做大规模 benchmark、调参、分布式调度或通用多 Harness 平台。完成上述验收后即停止扩张，后续工作进入 Backlog。
