# Day 2 执行记录（Polar Calculator rollout）

状态：`DAY2_STATUS=COMPLETED_WITH_NOTES`
run_id：`20260809T164046Z-calculator`
执行日期：2026-08-09T16:40Z – 2026-08-10T00:59Z（本地 2026-08-10 00:40–08:59）

## 版本与身份

```text
project_commit（执行时 HEAD）: 91402a0f8fff35aeb0e8ab0001fe6eda7c42a6d6
project_commit（结果 commit）:  cb456c0（"Complete Day 2 Polar calculator fixtures"，已 push）
polar_commit:                 f0e8343a7870abf6ec2366890f685881ceab92cb（stable，工作树干净）
sglang_commit/version:        28b095c01005d4a3a2a5b637b7d028b07fba31b2 / 0.5.13
model_id:                     Qwen/Qwen3-4B-Instruct-2507
model/tokenizer revision:     cdbee75f17c01a7cc42f958dc650907174af0554
runtime image identity:       polar-localhost-calculator:latest @ sha256:0a967a5c33ee206f7524181cdd4ff866389016651645336eee93de1a7d970f49（layout 3）
gpu_uuid（SGLang 用卡）:      GPU-4288d6c9-0384-25dc-1071-9bea701b9374（GPU 0）
```

## Topology 与实际启动命令

- 渲染配置：`configs/topology.rendered.sgl.yaml`（官方原件存 `configs/topology.official.sgl.yaml`）。
- **端口偏差**：8080 被系统账户 `server` 的进程占用（非本 run，未确认归属、未终止）→ rollout 端口改为 8081，已记录。
- 结构：Rollout Server 127.0.0.1:8081；Gateway localhost-node-01 127.0.0.1:8100；SGLang 127.0.0.1:8000；completion_persistence enabled；`TopologyConfig.load` 校验通过。

SGLang（最终 32K context；8K/16K 均因 qwen_code 请求 14846+8000 tokens 超限失败，见 notes）：

```bash
cd /data/day-01-workspace/src/polar
CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  uv run python -m sglang.launch_server \
  --model-path /data/day-01-workspace/hf-cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554 \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
  --host 127.0.0.1 --port 8000 --context-length 32768 --mem-fraction-static 0.35 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --trust-remote-code
```

Rollout Server / Gateway：

```bash
uv run polar serve_rollout -c <run>/configs/topology.rendered.sgl.yaml
uv run polar serve_gateway  -c <run>/configs/topology.rendered.sgl.yaml --node-id localhost-node-01
```

任务提交基于 pinned `examples/calculator/run.py` 的 `build_task_payload()`（`submit_calculator.py`），因端口偏差连接 8081，未修改 Polar 任何文件。

## 任务与产物 ID

```text
success task:     calculator-qwen_code-20260809T165213Z
success session:  sk-polar-0d2971a8-8dcb-4087-a8cd-f9824a5d07a4（COMPLETED，4 session 均 COMPLETED）
success completion: msg_ff634f2fb0a0（0001-msg_ff634f2fb0a0.json，Gateway 持久化）

fault task:       calculator-qwen_code-20260809T165634Z-fault
fault session:    sk-polar-61663888-df74-4d97-b2c6-b77e801123fe（ERROR，4 session 均 ERROR）
```

## Success evaluator 结果

- strategy：`test_on_output`；report：`empty_generation=true, resolved=false, failed_apply_patch=false, error_eval=false, test_timeout=false`；
- `outcome_reward=0.0` —— evaluator 正常完成但模型未解出（VALID_FAILURE 语义），不是基础设施失败；
- 原生字段确认：prompt_token_ids=14846（`response.json $.choices[0].input_token_ids` / `summary.json $.trajectory.traces[0].prompt_ids`）、output_token_ids=99（`$.choices[0].token_ids` / `traces[0].response_ids`）、sampled_logprobs=99（`$.choices[0].logprobs.content` / `traces[0].response_logprobs`）、loss_mask=99、finish_reason=stop；
- Gateway 重写证据：original_request.model=`qwen3-coder-plus` → transformed_request.model=`Qwen/Qwen3-4B-Instruct-2507`。

## Fault 注入点与错误阶段

- 注入：`request.json $.runtime.prepare` 追加 `{"type":"exec","command":"echo SYNTHETIC_FAULT_INJECTED_AT_INIT && false"}`（prepare 索引 4）；同时 `$.evaluator.config.test_timeout` 60.0→0.1；
- **test_timeout 注入不可观察**：agent 空 patch 时 `BasePatchEvaluator.evaluate` 提前返回，test_command 不执行（首次尝试归档于 `raw/fault-test_timeout-attempt/`）；最终 fault fixture 采用 prepare 注入；
- 错误阶段：runtime INIT；4 session 均 `status=ERROR`，`$.error="runtime initialization failed: prepare action 4 failed with exit code 1"`，run_ms=0、record_count=0、无模型调用；
- 分类：**INVALID_INFRASTRUCTURE**（synthetic，非破坏性；未停 SGLang/Gateway、未改模型与 runtime 镜像）；不得作为模型 reward=0 训练数据。

## 验证结果

- `python3 scripts/verify_polar_fixture.py tests/fixtures/polar/calculator_success` → PASS（10 项全 PASS，exit 0）
- `python3 scripts/verify_polar_fixture.py tests/fixtures/polar/calculator_fault` → PASS（exit 0）
- `python3 -m unittest discover -s tests -v` → 29 tests OK
- `python3 scripts/verify_day01_evidence.py ...` → PASS_WITH_NOTES（exit 0，仅保留 sglang_numeric_output_token_ids WARN——已在 Day 2 completion record 核实原生 token_ids，fixture 已含）
- verifier：Calculator 示例无 verifier（evaluator=test_on_output），verifier 相关字段按 missing 记录。

## Missing training fields

```text
policy_version（不得从时间戳/文件名推断）
group_id（不得从目录名猜测）
old_logprobs（训练语义；源仅有当前采样 logprobs）
fault 额外缺失：output_token_ids、sampled_logprobs、reward（INIT 前失败，无模型调用）
```

## 服务 PID 与停止

```text
gateway:  4163521   （uv run polar serve_gateway）
rollout:  4163468   （uv run polar serve_rollout）
sglang:   4174683   （python -m sglang.launch_server，32K）
```

停止顺序（依赖逆序，仅用 pids/*.pid 精确 PID，停止前已核对进程命令行）：

```text
Gateway(4163521) → TERM，正常退出（~4s）
Rollout Server(4163468) → TERM，正常退出（~4s）
SGLang(4174683) → TERM，正常退出（~6s）
```

停止后检查：

```text
nvidia-smi: GPU0 17MiB / GPU1 17MiB 空闲
端口: 8000、8081、8100 全部释放；8080 仍为 server 用户进程（本 run 从未使用/触碰）
docker ps: 无残留容器
```

## 未启动

```text
未启动 Slime、Megatron-LM、GRPO；未修改 Polar 核心代码；未修改验证器迁就数据；
未伪造 token IDs/logprobs/reward/policy_version；未重新 tokenize 文本冒充采样 token IDs。
```

## Remaining notes

1. qwen-code CLI 单轮 tool_call 后静默退出（run 阶段 ~2.7s，1 completion）；原因未定位；
2. SGLang response 结构化 `message.tool_calls=[]`，工具调用仅出现在 reasoning_content 文本；
3. 本地镜像以 image ID 标识，无 registry RepoDigest（干净复现需 tar checksum）；
4. 8K/16K context 尝试失败（input 14846 + completion 8000 超限），最终 32K 生效；
5. 8080 被系统账户 server 占用（持续存在），Day 3+ 需继续使用非 8080 端口或协调释放。
