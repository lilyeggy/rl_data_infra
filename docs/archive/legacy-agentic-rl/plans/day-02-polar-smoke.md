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
状态：NOT_STARTED
Polar commit：
SGLang 配置：
Harness：
success fixture：
fault fixture：
缺失训练字段：
未解决问题：
```
