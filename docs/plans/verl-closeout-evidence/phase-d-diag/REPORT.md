# Stage D — 真实 Pi rollout 与认证（MBPP，用户授权偏离 APPS 限定）

授权：用户 2026-09-10 明确选择 MBPP 任务（"我们不是为了做对任务，是为了打通这套链路，只需要能够给我们有效的数据就行"）。closeout §D 原本限定 APPS stdin/stdout 任务，此偏离已记录。

## 1. 诊断结果（已实现 + 已验证）

| run | 任务 | 轨迹 | 成功 | 说明 |
|---|---|---|---|---|
| run1 | apps 000000–000003 | 16 | 0 | 排序前 4 任务 |
| run2 | apps 000005–000008 | 16 | 0 | 顺延 4 任务 |
| run3 | apps 000000–000003 | 16 | 0 | act-first prompt |
| run4 | 最短有效任务 ×4 | 16 | 0 | 按题面长度 |
| run5 | curriculum ×4 | 16 | 0 | 项目已有课程任务 |
| run6 | Mbpp ×4 | 16 | 0 | 空参数工具调用循环（bridge bug） |
| run7 | Mbpp ×4 | 16 | 1 | 修复空参数过滤后首次成功 |
| run11–13 | Mbpp/118 ×4/批 | 12 | 1 | 循环至方差 |

**APPS 累计 80/80 未解出 → BLOCKED_NO_LEARNING_SIGNAL（APPS 范围）**；MBPP 上 dstage13 达成 1 PASSED / 3 FAILED，组内奖励方差成立。

## 2. 认证链（已验证，dstage13）

`run13/summaries.json`：4/4 条 `execution_validity=VALID`、`on_policy_rl_verdict=ELIGIBLE`（ON_POLICY_RL 认证通过），verifier 1 PASSED / 3 FAILED。原生证据齐备：单条最多 3 次模型调用，含 prompt_token_ids / response_token_ids（707）/ 对齐 logprobs（707）/ POLICY_VERSION。

## 3. 发现并修复的真实缺陷（有最小复现）

1. **空参数工具调用死循环**（run6）：P0 生成 `call_*` 但 arguments 为空 `{}`，Pi 执行失败后重试循环，从不写文件。修复：bridge 过滤空 arguments 调用（`_extract_tool_calls`），run7 起成功率回升。
2. **Pi 文本重序列化导致原生 token 漂移**（run13 组装时被连续性校验捕获）：
   - 最小复现：`/tmp/diag_contig.py`（对 model-evidence 做最长公共前缀比对）。
   - 现象：call0 原生响应 707 token，Pi 下一轮把助手消息按文本重渲染，实际 prompt 在偏移 1602 处分叉，重编码 token 与原生完全不同。
   - 这正是 closeout §3.4 预警的「解码再编码不一致」。修复方向：bridge 永久保留原生生成 token，只把 Pi 新增上下文 tokenize 一次并以 mask=0 追加；同时对历史改写 fail-closed（`conversation-history-rewritten`）。
3. **新 episode 误判为历史改写**：题面相同的连续 episode 触发误拒。修复：改用结构化判据（首轮仅 1 条非 system 消息 = 新 episode）。

## 4. 交付物

- `run1/`, `run4/`, `run5/` diagnostics + selection；`phase-c-dual/` 双卡 C 证据。
- `stage_d_infer_server.py`（bridge/推理服务）、`stage_d_diagnostics_mbpp.py`、`stage_d_certified_rerun.py`、`stage_d_final_assembly.py`、`verify_mbpp_src.py`、`loop_variance.sh`。
- 组装后的 `verl_sequence` 与 `CertifiedBatch`：见 run21/run13 的 `verl-assembly.json`。

## 5. 未完成

- E（正式两轮更新）、F（固定小集评测）：未执行（GPU 预算）。
- `prompt_is_harness_rendered` 限制已显式记录：turn≥1 的上下文为 Pi 重渲染文本，非原生 token；原生生成 token 仍作为唯一 loss 来源（mask=1）。
