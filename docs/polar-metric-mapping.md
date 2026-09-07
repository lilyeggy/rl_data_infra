# Polar 指标与 Local Data Plane 指标映射

## Polar 实际使用的指标

当前使用的 Polar 官方仓库在 Slime bridge 中向 W&B 输出以下聚合指标：

- `polar/session_ms/register_to_init_queue_mean`
- `polar/session_ms/init_mean`
- `polar/session_ms/run_mean`
- `polar/session_ms/postrun_mean`
- `polar/reward_mean`、`polar/reward_std`、`polar/reward_mean_completed`
- `polar/rollout_success_rate`
- `polar/eval/resolved_rate`
- `polar/staleness/mean`（有异步 trainer 时）

其 Gateway heartbeat 还暴露 `init_queue_depth`、`init_inflight`、`ready_depth`、`run_inflight`、`postrun_queue_depth`、`postrun_inflight`。调度器按 run、post-run、init 的压力顺序选择节点。官方 Rollout 文档也明确把 `REGISTERED → INIT` 定义为排队时间，`INIT`、`RUN`、`POSTRUN` 作为不同阶段。[Polar Rollout Service](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server/blob/stable/src/polar/rollout/README.md)

另外，session trajectory metadata 包含 `record_count` 与 `trace_count`。这两个字段说明 Polar 自己也不把 session 当作固定工作量：一个 session 可以包含不同数量的模型 completion/trajectory trace。

## 与我们的当前映射

| Polar 指标 | Polar 语义 | Local Data Plane 对应事实 | 当前状态 |
|---|---|---|---|
| `register_to_init_queue_ms` | Gateway admission 后、runtime init 前的等待 | 从任务提交到 Docker launcher 实际开始的等待 | **缺失**；当前 benchmark 只测线程提交后的 wall time |
| `init_ms` | runtime startup + workspace prepare | Docker Sandbox 创建、workspace mount、Pi bootstrap、至首个模型请求 | 部分具备：`SANDBOX_STARTED → MODEL_REQUEST` |
| `run_ms` | Harness 执行 Agent | Sandbox span 中的模型请求、工具调用、Harness 逻辑 | 具备，但需要直接输出统一 `run_ms` |
| `postrun_ms` | trajectory build、eval、teardown | workspace snapshot、Verifier、quality gate、ExecutionBundle finalize | 部分具备；目前 event 外还有一小段未命名开销 |
| `reward_mean` / `resolved_rate` | task/verifier 质量 | verifier pass rate、task reward | 具备，且能区分 verifier failure 与 infrastructure-invalid |
| `rollout_success_rate` | 非 placeholder 的 completed session 占比 | producer completed + integrity complete rate | 具备 |
| `record_count` / `trace_count` | 模型 completion / trajectory 条数 | `model_call_count`、event count、RL-usable call/token count | 具备，且我们额外有 native token/logprob 完整性 |
| `policy_staleness` | 异步训练中 rollout policy 与当前 policy 的差 | policy revision 与训练消费 revision 的差 | **缺失**；当前只有 immutable policy revision，不计算 age/staleness |
| stage pressure | Gateway 的 queue/inflight/ready pool | local rollout pool、proxy、verifier 各阶段 queue/inflight | **缺失**；当前单机 ThreadPoolExecutor 没有导出 gauges |

## 当前 c1/c4/c8 数据：只能作为阶段观察，不能直接宣称 speedup

下表来自三轮既有真实 Agent benchmark。Local 与 Polar 的 session 仍有不同模型调用长度，因此只用于发现瓶颈，不用于证明某一方工程更快。

| 指标（每 session 均值） | c1 Local / Polar | c4 Local / Polar | c8 Local / Polar |
|---|---:|---:|---:|
| Local bootstrap 至首个模型请求 / Polar init | 0.63s / 0.73s | 0.77s / 0.82s | 1.02s / 1.02s |
| Local 模型调用数 / Polar record count | 2.67 / 2.33 | 3.00 / 2.83 | 3.29 / 2.75 |
| Local 模型响应总时长 / Polar run stage | 11.12s / 10.90s | 14.12s / 11.62s | 15.08s / 9.59s |
| Local Verifier+Finalize / Polar postrun | 0.15s / 0.14s | 0.16s / 0.30s | 0.16s / 0.15s |
| Local verifier pass / Polar resolved rate | 100% / 100% | 100% / 91.67% | 100% / 100% |

其中“Local 模型响应总时长”和“Polar run stage”不是同一层级：前者只累计模型请求，后者包含 Polar Harness run；再加上模型调用数不同，不能做比值结论。它们只说明 c8 的主要差异发生在 Agent/模型路径，而不是 verifier。

## 我们应该在 Polar 指标之上新增的指标

Polar 的指标适合 Rollout 服务调度，但本项目是 Agent **Data Plane**，还必须证明训练数据真的可用：

| 新增指标 | 定义 | 为什么需要 |
|---|---|---|
| `durable_verified_episode_goodput` | verifier pass、integrity complete、ExecutionBundle 已 durable 的 episodes/s | 防止只完成内存中的 session，却没有可复用证据 |
| `SFT_goodput` | verified episode 中被质量 Gate 接收的 assistant-action tokens/s | SFT 不要求 token ids/logprobs |
| `RL_goodput` | verified episode 中同时有 native token ids、aligned logprobs、policy revision 的 sampled tokens/s | 这是 RL 真正可消费的数据单位 |
| evidence completeness rate | 完整模型/工具/Sandbox/Verifier 证据的 episode 占比 | 可观测不等于可回溯、可训练 |
| `proxy_added_latency` | client 观察模型请求延迟 - upstream vLLM 处理延迟 | 隔离我们 Proxy 的热路径成本 |
| durable write bytes / accepted token | Evidence 与 Bundle 的存储成本 | 避免以昂贵持久化换取吞吐表面优势 |
| policy staleness | rollout revision 与 trainer 当前/消费 revision 的版本距离或时间距离 | 连接 Agentic RL 数据生产和训练闭环 |

## 下一版统一指标面板

先复刻 Polar 的 stage timing 和 stage pressure，以得到可比的服务运行视图；再在其上增加 data-plane-only 的 quality/evidence/goodput。对外报告必须同时显示：

1. queue-inclusive E2E（生产容量）；
2. active-only init/run/postrun（定位瓶颈）；
3. record/model-call/output-token 分布（证明工作量相同）；
4. verified/SFT/RL goodput（证明产物可训练）；
5. p50/p95/p99 与资源成本（避免只报平均值）。

只有在 task、Agent、model、seed、workload distribution、Evidence 语义都匹配时，才给出 Local/Polar 比值；否则分别展示能力与观测数据。
