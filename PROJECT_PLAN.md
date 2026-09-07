# Agent Improvement Data Plane：实施计划

## P0：收敛和可信内核

- [x] 归档旧自定义 local-model/GRPO/SFT 主链路；
- [x] 新增统一 `ExecutionIdentity`；
- [x] 新增 trainer-neutral `RolloutProducer` 协议；
- [x] 新增统一 consumer certification 入口；
- [x] 将旧 `TrainingCandidateView` 收敛为统一 eligibility 的兼容投影；
- [x] 建立 dataset manifest/compiler、split leakage 与 preference pairing Gate。

## P1：真实 Producer

- [x] 建立 Local Docker Launcher 契约、identity 注入与安全资源边界；
- [x] 建立 Model Proxy request/response/token/logprob/policy evidence 契约；
- [x] 实现 OpenAI-compatible Model Proxy HTTP forwarder、安全证据日志和 execution bearer token；
- [x] 建立 `ExecutionRunManifest` 与 Local Run finalizer；
- [x] 实现 `prepare-local` 自动冻结 Manifest/launch plan；
- [x] 实现 `execute-local` 一键事务：proxy → Harness → verifier → finalize → cleanup；
- [x] 建立带 execution bearer token 的通用 Harness event ingress/SDK；
- [x] 将 stdout/stderr/workspace diff/patch/verifier report 写入内容寻址 ArtifactStore；
- [x] 超时或 Ctrl-C 时按 identity-bound container name 做确定性清理；
- [x] 用真实 Docker 容器跑通无 GPU Harness/model-proxy/verifier/finalizer 全链路；
- [x] 用目标 Harness 和租用 GPU 模型跑通同一全链路；
- [x] 将 Pi direct capture 包装为 capability-limited producer；
- [x] 实现 Polar 官方 HTTP client 与 `PolarLiveBatchProducer`；
- [x] 严格导入 Polar TaskResult/SessionResult/Trajectory，并为每个 session 分配独立身份；
- [x] 建立 raw Episode 与 policy trace 的 `ExecutionBundle` checksum 契约；
- [x] 加入 malformed/missing/nonterminal/timeout capability 单元 Gate；
- [ ] 在实际 Polar 服务上完成 crash/timeout/retry integration tests；
- [x] 实现 Episode/producer/verifier 的离线严格 bundle assembler；
- [ ] 在真实 live session 上验证 Harness events、policy trace 和 verifier report 的 bundle join。

## P2：Harness 闭环

- [ ] 将所有真实实验统一经 Producer 执行；
- [x] 为 Local Execution 自动固定 task/environment/model/harness/evaluator manifests；
- [ ] 通过统一 certification 后进入 compare/gate；
- [ ] 建立 failure regression bank。
- [ ] 冻结模型、任务、sampling 与 verifier，建立 Harness baseline；
- [ ] 从认证 Episode 自动区分模型、Harness、Serving、Sandbox 与 Verifier 失败；
- [ ] 对 Harness candidate 进行同任务 A/B 重放，比较成功率、infra-invalid、步数、token、耗时与恢复率；
- [ ] 只有通过固定模型 regression gate 的 Harness candidate 才能发布。

## P3：有限资源单卡模型闭环

- [x] 建立单 GPU rollout/train/evaluate phase 与 evidence handoff 契约；
- [x] 建立原子状态持久化、checksum CAS、恢复和 dry-run command plan；
- [x] 为 ProducerArtifact、EligibilityDecision、DatasetManifest 建立严格磁盘重载；
- [x] 建立 producer/certification 数据质量报告 CLI；
- [x] 建立从持久化证据严格重建 Slime admission 的离线 smoke CLI；
- [ ] Polar rollout-only；
- [x] certified Slime admission envelope（不复制 Slime Sample/trainer）；
- [ ] 将 admission envelope 接入官方 Polar–Slime bridge；
- [ ] Slime train-only/replay；
- [x] policy fingerprint/checkpoint handoff；
- [x] policy-v0/v1 固定 holdout 对比（candidate 未提升，Gate 正确 REJECT）。

### P3-A：重新建立可信 Agent SFT 闭环

- [x] 修复训练视图的 action–observation closure：按 `tool_call_id` 将 canonical tool result 插回下一次 assistant 决策的上下文；
- [x] 将 closure 加入 fail-closed SFT 序列化/训练包准入；旧 `teacher-sft-example/v1` 不再进入正式训练；
- [x] 将旧 MBPP LoRA 标记为 missing-observation ablation，不作为正式候选；
- [x] 封存完整 MBPP 与 HumanEval，仅用于最终评测；
- [x] 使用公开 APPS train split 作为正式训练任务源；MBPP、HumanEval、BigCodeBench、LiveCodeBench 等 Qwen2.5-Coder 报告 benchmark 全部封存为评测集；
- [x] 让强 Teacher 经 Harness 重新生产、验证并认证 Agent 轨迹；
- [x] 抽样和自动验证 `assistant action → tool observation → next action`，并检查 benchmark contamination；
- [x] 用新训练包重新进行 LoRA SFT；
- [x] 在冻结 Harness/Serving 下完成 base、错误 ablation、正确 SFT 的完整 MBPP/HumanEval 对照。

### P3-B：完成 Agentic RL 闭环

- [x] 定义算法相关 RL Consumer Profile，区分 GRPO 重算 logprob、PPO 直接消费 behavior logprob 等要求；
- [x] 让当前学生 policy 而非 Teacher 产生带 `group_id`、`policy revision`、token/mask/reward 的 rollout；
- [x] 建立跨 MODEL_REQUEST/RESPONSE、tool call/result、reward 的状态转移闭合认证；
- [x] 实现 trainer-neutral RL training view 与 verl/Slime adapter，不在 Data Plane 内复制 Actor/Critic/优化器；
- [x] 用 3 个非评测任务完成 policy-v0 rollout → trainer update → policy-v1 rollout 的真实微型闭环；
- [x] 微型闭环通过后再扩大任务量和 rollout group（已完成 20 题 80 条轨迹 apps-rl-cycle-002 与 HumanEval 50 题对比评测）。

### P3-A / P3-B 并行与单卡约束

- SFT 的公开任务选择、Teacher 数据生产、训练视图验证，可与 RL contract、adapter 和无 GPU/小样本集成测试并行开发；
- 正确 SFT 模型是正式 RL 初始 policy 的依赖，但 RL adapter 可提前用合成 fixture/base policy 验证；
- 单张 A6000 上的大规模 Teacher rollout、SFT、学生 rollout 与 RL 更新采用 time-sliced 排队，不并发争抢显存；
- RL 正式效果评测必须等待正确 SFT policy-v0 产出，不能使用 missing-observation ablation 作为正式起点。

## P4：扩容

- [ ] 在单卡闭环证明学习信号后再进入多卡 async；
- [ ] 加入 weight staleness、backpressure、resampling 和恢复指标；
- [ ] 扩展多 Harness、更多任务分布和数据集版本。
