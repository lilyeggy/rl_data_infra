# Rollout Data Plane 公平比较协议

## 为什么不以 `session/s` 作为唯一指标

一个 Agent session 不是固定工作量：它可能走 2 次或 8 次模型调用，生成不同数量的 token，调用不同数量的工具，并触发不同长度的 verifier。因而 `session/s` 是端到端交付指标，但不能单独解释为编排、Sandbox 或 Proxy 的性能。

推理系统通常同时报告 token throughput、TTFT、TPOT/ITL 和端到端延迟，而不是只报告 request throughput。vLLM 的持续基准也将 Output Throughput、TTFT、TPOT 与满足 SLA 的最大并发一起比较；NVIDIA GenAI-Perf 同样将 output-token throughput、TTFT、ITL 与 request throughput 分列。[vLLM Performance Dashboard](https://docs.vllm.ai/en/latest/benchmarking/dashboard/) [NVIDIA GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2640/user-guide/docs/perf_analyzer/genai-perf/README.html)

对 Agent Rollout，必须把“做了多少工作”“产出了多少可用训练数据”“花了多少时间”分开记录。

## 先冻结公平性合同

每次比较创建一份不可变的 `comparison contract`。未通过合同只能展示观测值，禁止写 speedup。

| 维度 | 必须相同或明确声明 |
|---|---|
| 任务 | task set、workspace 初始快照、verifier、任务数量与难度分层 |
| Agent | Harness 版本、system prompt、工具 schema、最大模型调用数、超时与重试策略 |
| 模型 | checkpoint revision、tokenizer、模板、采样参数、seed、最大上下文与最大输出 |
| 服务 | 同一 vLLM revision/参数、同一 GPU、相同空闲/预热状态、没有其他 GPU workload |
| Sandbox | 镜像 digest、CPU/内存/PID/network 限制、workspace 挂载方式 |
| 数据语义 | 是否请求 token ids/logprobs、是否持久化 response、是否要求 policy revision、质量 Gate 定义 |
| 计时边界 | 提交时刻、排队、Sandbox init、Agent run、验证、持久化、最终 bundle durable 的精确定义 |

Polar 的公开设计本身将 session 排队、init、run 和 post-run 区分，并说明 timeout 从 INIT 而不是调度时开始。因此本项目也必须同时保存 queue-inclusive 和 active-only 两种时间。[Polar Rollout Service](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server/blob/stable/src/polar/rollout/README.md)

## 三个必须分离的 benchmark

### A. 推理内核 benchmark：回答“模型服务快不快”

使用固定的请求语料，而不是 Agent loop。请求语料冻结 prompt token 长度、输出上限、工具 schema 和 seed；两套系统走同一个 vLLM endpoint。

报告：

- output tokens/s、request/s；
- p50/p95/p99 TTFT、TPOT、E2E latency；
- queue delay 与最大可接受并发（在明确 TTFT/TPOT SLO 下）；
- GPU 利用率、VRAM、GPU-seconds per output token。

这层不衡量 Agent，也不宣称 Data Plane 优势；它用于排除 vLLM 本身或 GPU 负载导致的差异。

### B. 固定轨迹回放 benchmark：回答“Data Plane 热路径快不快”

录制一批固定的 `model request → tool result` 序列，在两边按相同次序回放。每个工作单元有 `trajectory_workload_id`，包含 prompt/tool hash、输入/输出 token 数、模型调用数和工具调用数。

报告：

- model request forward latency；
- Proxy added latency = client-observed request latency - upstream vLLM latency；
- Evidence serialization/fsync latency、写入字节数、对象存储字节数；
- Sandbox cold/warm init、Verifier、Finalization、ExecutionBundle durable latency；
- 每个固定 workload 的 p50/p95/p99，而非跨不同长度 session 的平均值。

这才是 Local Data Plane 与 Polar Gateway/Rollout 路径能进行“编排开销”比较的层级。固定 seed 是辅助措施；因为并发调度仍可能改变工具交互，真正隔离变量要依赖回放。

### C. 端到端数据生产 benchmark：回答“每秒产出多少可信训练数据”

使用真实 Agent loop 和真实任务集。`session/s` 可以保留，但只是次级运营指标。主指标是 goodput：

```text
verified_episode_goodput = verifier-passed + integrity-complete episodes / wall time

SFT_goodput = accepted assistant-action tokens from verified episodes / wall time

RL_goodput = sampled response tokens with native token ids + aligned logprobs
             + immutable policy revision, from verified episodes / wall time
```

一个 failed verifier、infrastructure-invalid、证据不完整、或越过数据质量 Gate 的 session 贡献 0 goodput，但要单独计入失败原因。vLLM 将满足 SLO 的吞吐称为 goodput；这里把同一思想扩展为“满足数据质量与训练契约”的 token goodput。[vLLM performance discussion](https://vllm-project.github.io/2025/09/05/anatomy-of-vllm.html)

报告：

- verified episode/s、SFT tokens/s、RL-usable tokens/s；
- verifier pass rate、integrity-complete rate、RL-usable call/token rate、infra-invalid rate；
- 每条 episode 的模型调用数、input/output token 数、工具调用数分布；
- p50/p95/p99 queue-inclusive E2E 与 active-only E2E；
- GPU/CPU seconds、VRAM peak、storage bytes per accepted training token。

## Evidence 对齐是比较前提

当前 Local Data Plane 的 Controlled 模式会请求 native logprob、持久化模型证据并生成 ExecutionBundle；Polar 的现有 smoke 路径没有证明它承担相同的 token/logprob 持久化语义。因此两者当前只能进行两类诚实比较：

1. **common-denominator mode**：双方都关闭 RL Evidence，只比较任务完成、Sandbox、Agent 与 verifier 的端到端效率；
2. **evidence-equivalent mode**：双方都请求并持久化 token ids、aligned logprobs、policy revision，再比较 RL_goodput。

不得把“Polar light trace”的 `session/s` 与“Local RL evidence”的 `session/s` 解释为同一工作量。NVIDIA 对 agent RL 基础设施的描述同样将模型服务所保存的 token/logprob metadata 视为 RL 所需能力，而不是普通推理服务默认具备的字段。[Nemotron infrastructure report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)

## 负载模型、样本量与统计规则

- **Offline / closed batch**：一次提交固定任务集，测 makespan 与 goodput，适合夜间数据生产；
- **Server / open load**：固定 arrival rate 或 Poisson 到达，测在 p99 E2E SLO 下的最大 RL_goodput，适合生产容量规划；MLPerf 也将 offline measured throughput 与 server latency-constrained throughput 分为不同场景。[MLPerf Inference Rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
- 每个点至少 20 个独立 task seeds；报告 mean、median、p95 和 bootstrap 95% CI，禁止只报单次最大值；
- 对 Agent loop 按 `model_calls`、`output_tokens`、`tool_calls`、`verifier_time` 分层，再在相同分层内比较；
- 每次运行记录模型服务启动参数、GPU 状态、背景进程、镜像 digest 与代码 revision；
- 任何一侧成功率显著不同，都必须同时报告 completed throughput 与 verified/RL goodput，不能只选有利分母。

## 本项目后续采用的结论模板

每次报告按下列顺序给结论：

1. 合同是否通过；
2. A 层模型服务是否等价；
3. B 层固定工作量下的 Data Plane added latency；
4. C 层 verified/SFT/RL goodput 与可靠性；
5. 资源成本与尾延迟；
6. 不能解释的变量及下一步实验。

这样项目的亮点就不是“宣称比 Polar 快”，而是：**将 Agent 轨迹、验证证据和训练可用性纳入统一的、可审计的性能与质量合同。**
