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

## P4：扩容

- [ ] 在单卡闭环证明学习信号后再进入多卡 async；
- [ ] 加入 weight staleness、backpressure、resampling 和恢复指标；
- [ ] 扩展多 Harness、更多任务分布和数据集版本。
