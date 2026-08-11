# Day 3：Polar Coding/SWE Rollout 与 Verifier Fixture

> 本阶段把 Day 2 的最小示例升级为真实 Coding Agent rollout，仍然不运行 Trainer。

## 1. 阶段目标

1. 选择少量稳定 SWE-Gym/SWE-bench 风格任务；
2. 通过 Polar 运行真实多轮 model/tool/code-edit/test 轨迹；
3. 保存 verifier、patch、termination 和 policy metadata；
4. 建立 success、valid failure、invalid infrastructure 三类真实/合成 fixture；
5. 明确哪些状态可以成为 reward=0，哪些必须从训练中排除。

## 2. 任务选择

先选 3–10 个小任务，优先：

- Python 仓库；
- runtime image 可稳定获取；
- tests 较短；
- 无外部在线服务；
- base commit 可重复 reset；
- 基础模型至少偶尔成功，便于后续形成 reward variance。

对每个任务记录 task ID、repo、base commit、image digest、test spec、cold/warm runtime、排除原因。

## 3. Reference Harness

第一版只选一个 Polar 已支持且能稳定运行的 Harness。外部 Harness 项目仍保持独立；此阶段不要求实现两个 Harness Adapter，也不复制其源码。

## 4. Rollout Capture

每条轨迹至少记录：

```text
task_id
rollout/session/request IDs
policy_version
model/tokenizer revision
input/output token metadata
tool calls/results
files/patch
termination reason
runtime/harness/model/verifier statuses
reward and evidence
timings
```

至少保留一条包含多个模型请求、文件读取、代码修改、测试执行和最终 patch 的轨迹。

## 5. 三类 Outcome Fixture

### VALID_SUCCESS

环境和 verifier 正常，任务 reward 成功。

### VALID_FAILURE

环境和 verifier 正常，模型的 patch/答案没有通过任务验证。它是合法负样本。

### INVALID_INFRASTRUCTURE

至少覆盖其中两类：runtime prep failure、Harness crash、model backend failure、verifier crash/timeout、trace incomplete。自然产生困难时可以注入 synthetic fault，但不得伪装成真实线上分布。

## 6. Verifier Evidence

第一版只要求足够区分任务失败和基础设施失败：

```text
base commit
patch hash
runtime image digest
test command/spec
stdout/stderr/exit code
timeout/crash state
reward
```

Clean replay 只做最小验证：将 patch 应用到干净 workspace，确认 verifier 能重复执行。暂不建设通用 replay 平台。

## 7. Pilot Sampling

对少量任务各生成 2–4 条 rollout，统计：

- success/failure/invalid 比例；
- reward variance；
- 平均 token/turn/tool 数；
- verifier latency；
- 是否适合作为 Day 6 小规模 GRPO 训练任务。

## 8. 本阶段不做

- 不实现数据 Processor；
- 不设计多 Session 表示或信用分配；
- 不运行 GRPO；
- 不追求大量任务和高吞吐；
- 不修改 Polar 内核来承载本项目逻辑；
- 不把基础设施失败强行映射为 reward=0。

## 9. 产物

```text
artifacts/day-03/
tests/fixtures/polar/coding_success/
tests/fixtures/polar/coding_valid_failure/
tests/fixtures/polar/coding_invalid_infra/
configs/polar/coding/
notes/polar-coding-field-map.md
notes/task-pilot-report.md
```

## 10. 验收门

- [ ] 至少 3 个候选任务完成 runtime baseline；
- [ ] 至少一条真实多轮 Coding rollout 完整保存；
- [ ] success 与 valid failure fixture 都存在；
- [ ] 至少两类 invalid infrastructure fixture 存在；
- [ ] verifier evidence 足以复核 outcome；
- [ ] policy version 和 task/group identity 可追踪；
- [ ] pilot report 能选出后续训练任务；
- [ ] 本阶段没有依赖 Slime/Megatron。

## 11. 执行记录

```text
状态：EXECUTED_AS_FAILURE（2026-08-11 run 20260811T030617Z-swebench；runbook docs/runbooks/day-03-polar-coding.md）
Harness：qwen_code（@qwen-code/qwen-code@0.14.5）
候选/有效任务：3 个 baseline 通过（sphinx-doc__sphinx-8595、sympy__sympy-20916、pytest-dev__pytest-5809）；排除 pylint-4661/sklearn-14141（FAIL_TO_PASS 在 base 上已通过）、django-12419（镜像传输失败）
真实 rollout 数量：12（3 候选×2 + sympy 补采 4 + fault 2）
success/failure/invalid：success=0 / valid failure=10 / invalid infra=1（另 1 次 verifier-timeout 注入不可观察）
reward variance：无（全部单轮 empty_generation、reward=0；无正样本）
fixture：tests/fixtures/polar/coding_valid_failure + coding_invalid_infra（验证均 exit 0）；coding_success 缺失（无真实 success，不伪造）
训练候选任务：无（qwen_code 单轮退出限制，详见 notes/task-pilot-report.md）
未解决问题：qwen_code 单轮退出（SGLang 结构化 tool_calls 为空 + content 为空）；swebench 官方镜像/make_test_spec 网络依赖（已用代理+fallback 绕过）
```

执行入口：`docs/runbooks/day-03-polar-coding.md`
