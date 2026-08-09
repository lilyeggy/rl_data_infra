# Day 2：Polar Calculator Rollout 参考复现

> 本阶段第一次运行 Polar，但只复现 rollout。Slime/Megatron 不参与。

## 1. 阶段目标

1. 按官方 Calculator 示例跑通最小 Polar rollout；
2. 理解 Rollout Server、Gateway、Harness、SGLang 和 evaluator 的调用边界；
3. 保存完整 raw artifacts，形成第一个不可变 golden fixture；
4. 列出 Polar 输出中可用于 Agentic RL 的字段和缺失字段；
5. 为 Day 4 的 `PolarSourceAdapter` 提供真实输入。

## 2. 最小拓扑

```text
1 × Polar Rollout Server
1 × Polar Gateway
1 × SGLang backend
1 × Calculator runtime
1 × built-in Harness
1 × task
```

模型继续使用 Day 1 已验证的 Qwen3-4B。context 和并发以稳定完成为准，不照搬官方大 GPU 参数。

## 3. 执行顺序

### 3.1 固定 Polar 版本

使用 stable/pinned commit，记录本地补丁。任何针对 SGLang token metadata 的 patch 必须单独列出来源和 checksum。

### 3.2 构建 Calculator runtime

记录 Docker/Apptainer backend、image digest、构建命令、工作目录和 evaluator 入口。

### 3.3 启动 SGLang

验证 health、model revision、token/logprob 返回、tool parser 和实际显存。保存启动参数，不使用未记录的 shell 环境变量。

### 3.4 启动 Polar 服务

依次启动 Rollout Server 和 Gateway，保存 stdout/stderr、端口、topology 与 Polar config 的渲染后版本。

### 3.5 提交一个 Calculator task

确认完整路径：

```text
submit
→ rollout server
→ gateway
→ runtime/harness
→ model API proxy
→ SGLang
→ tool/edit
→ evaluator
→ response/summary
```

### 3.6 字段审计

对 request/response/summary 和 Gateway trace 标注：

- task/rollout/session/request ID；
- model、tokenizer、policy revision；
- input/output token IDs；
- old logprobs（如果源确实提供）；
- tool call/result；
- termination/timeout/error；
- reward、evaluator result；
- patch/final output；
- source file 与字段路径。

禁止在此阶段人为补齐不存在的训练字段。

## 4. Golden Fixture

保存小型、去密、可提交的 fixture：

```text
tests/fixtures/polar/calculator_success/
├── source-manifest.json
├── request.json
├── response.json
├── summary.json
├── normalized-logs/
└── README.md
```

大日志和 runtime layers 只存 artifact manifest/checksum，不提交 Git。

## 5. 本阶段不做

- 不实现通用 canonical schema；
- 不修改 Polar 核心；
- 不接 Slime；
- 不做 GRPO；
- 不把 Calculator 当作 Coding/SWE 能力实验；
- 不实现大而全 Quality Gate。

## 6. 故障记录

至少人工触发一个非破坏性失败，例如错误 task 参数或 verifier timeout 配置，用于观察 Polar 的 status/error 字段；该 fixture 必须标记 `synthetic_fault=true`。

## 7. 产物

```text
artifacts/day-02/
tests/fixtures/polar/calculator_success/
tests/fixtures/polar/calculator_fault/
notes/polar-field-map.md
configs/polar/calculator/
```

## 8. 验收门

- [ ] Calculator 从提交到 evaluator 完整结束；
- [ ] 所有模型请求经过 Polar Gateway；
- [ ] raw request/response/summary/logs 已保存并校验 checksum；
- [ ] token/logprob 字段来源被实际确认；
- [ ] 至少一个成功 fixture 和一个 synthetic fault fixture；
- [ ] 字段映射笔记区分“源字段存在”和“后续需要派生”；
- [ ] 全程没有启动 Slime/Megatron。

## 9. 执行记录

```text
状态：COMPLETED_WITH_NOTES（2026-08-09/10，run 20260809T164046Z-calculator）
Polar commit：f0e8343a7870abf6ec2366890f685881ceab92cb（stable）
SGLang 配置：0.5.13（28b095c0…）；Qwen/Qwen3-4B-Instruct-2507@cdbee75f…；context 32768（8K/16K 因 harness 请求 22846 tokens 超限），mem-fraction-static 0.35，GPU 0
Harness：qwen_code（@qwen-code/qwen-code@0.14.5）
success fixture：tests/fixtures/polar/calculator_success/ —— calculator-qwen_code-20260809T165213Z / sk-polar-0d2971a8…，evaluator 正常完成但 reward=0.0、resolved=false（模型未解出，VALID_FAILURE 语义）；原生 token_ids/logprobs 确认
fault fixture：tests/fixtures/polar/calculator_fault/ —— calculator-qwen_code-20260809T165634Z-fault / sk-polar-61663888…，synthetic runtime INIT failure（prepare action 4 exit 1，4 session ERROR），分类 INVALID_INFRASTRUCTURE；evaluator timeout-only 尝试因 empty patch 短路（BasePatchEvaluator 提前返回）不可观察，最终未成为 fault 注入方式
缺失训练字段：policy_version、group_id、old_logprobs；fault 另缺 output_token_ids/sampled_logprobs/reward
未解决问题：qwen_code 单轮 tool_call 后退出（原因未定位）；SGLang 结构化 message.tool_calls 为空（工具调用仅在 reasoning_content 文本）；8080 被系统账户 server 占用（已改用 8081）；本地镜像无 RepoDigest
```