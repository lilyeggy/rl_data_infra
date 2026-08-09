# Day 2：Polar Calculator Rollout Runbook

> 状态：`EXECUTED_WITH_NOTES`
>
> 已在服务器执行（2026-08-09/10，run `20260809T164046Z-calculator`，polar f0e8343a）。
> 本阶段运行锁定版本的 Polar Calculator 示例，取得真实 rollout artifacts，
> 为 Day 4 的数据契约与 `PolarSourceAdapter` 提供输入样本。本阶段不重新实现
> Polar，也不启动 Slime/Megatron。

## 1. 完成目标

服务器上只完成以下闭环：

```text
pinned Polar + pinned SGLang/model/runtime
→ one Calculator success task
→ one explicitly marked synthetic fault task
→ preserve raw session/completion artifacts
→ prepare reviewed staging copies
→ package success/fault fixtures
→ verify manifests/checksums
→ audit actual Polar fields
```

最终产物：

```text
artifacts/day-02/<run-id>/
tests/fixtures/polar/calculator_success/
tests/fixtures/polar/calculator_fault/
notes/polar-field-map.md
```

通过后立即结束 Day 2，不扩展成 Polar 部署项目。

## 2. 边界

本阶段使用但不实现：

- Polar Rollout Server、Gateway、runtime pool；
- Calculator runtime、built-in Harness、trajectory builder 和 evaluator；
- SGLang inference backend。

本阶段自行维护：

- 单节点、单 Gateway、单 SGLang 的渲染后 topology；
- 命令、PID、日志和 artifact provenance；
- fixture staging、打包、验证与字段审计。

禁止：

- 启动 Slime/Megatron 或运行 GRPO；
- 修改 Polar 核心以迁就本项目 schema；
- 从生成文本重新 tokenize 后冒充 sampled token IDs；
- 给缺失 logprobs、reward 或 policy version 填零或猜值；
- 用 `pkill`、模糊进程名或未知 PID 停止服务器进程；
- 提交 secrets、完整大日志、模型权重或私有基础设施信息。

## 3. 锁定事实与上游依据

以 `configs/upstream-lock.yaml` 为唯一版本事实：

```text
Polar commit:
f0e8343a7870abf6ec2366890f685881ceab92cb

SGLang commit/version:
28b095c01005d4a3a2a5b637b7d028b07fba31b2 / 0.5.13

Model:
Qwen/Qwen3-4B-Instruct-2507

Model/tokenizer revision:
cdbee75f17c01a7cc42f958dc650907174af0554

Calculator runtime image:
polar-localhost-calculator:latest
sha256:0a967a5c33ee206f7524181cdd4ff866389016651645336eee93de1a7d970f49
layout version: 3
```

锁定 commit 的官方依据：

- [Calculator README](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server/blob/f0e8343a7870abf6ec2366890f685881ceab92cb/examples/calculator/README.md)
- [SGLang topology](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server/blob/f0e8343a7870abf6ec2366890f685881ceab92cb/examples/calculator/topology.sgl.yaml)
- [Calculator runner](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server/blob/f0e8343a7870abf6ec2366890f685881ceab92cb/examples/calculator/run.py)
- [Gateway capture semantics](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server/blob/f0e8343a7870abf6ec2366890f685881ceab92cb/src/polar/gateway/README.md)
- [Rollout persistence semantics](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server/blob/f0e8343a7870abf6ec2366890f685881ceab92cb/src/polar/rollout/README.md)

若服务器 checkout、模型 revision、镜像身份或关键包与 lock 不同，先停止并记录
drift；不得在未说明的情况下换用 stable 最新代码。

## 4. 目录与一次运行的身份

从项目仓库根目录执行。以下变量只在当前 Day 2 shell 使用：

```bash
PROJECT_ROOT="$(pwd)"
POLAR_ROOT="/data/day-01-workspace/src/polar"
MODEL_ROOT="/data/day-01-workspace/hf-cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
DAY02_ROOT="/data/day-01-workspace/artifacts/day-02"
DAY02_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-calculator"
DAY02_RUN_DIR="${DAY02_ROOT}/${DAY02_RUN_ID}"
```

创建：

```text
<run-dir>/
├── configs/
├── logs/
├── pids/
├── raw/
├── staging/
│   ├── success/
│   └── fault/
└── validation/
```

```bash
mkdir -p "${DAY02_RUN_DIR}/configs"
mkdir -p "${DAY02_RUN_DIR}/logs"
mkdir -p "${DAY02_RUN_DIR}/pids"
mkdir -p "${DAY02_RUN_DIR}/raw"
mkdir -p "${DAY02_RUN_DIR}/staging/success"
mkdir -p "${DAY02_RUN_DIR}/staging/fault"
mkdir -p "${DAY02_RUN_DIR}/validation"
```

不要使用 `latest` 目录覆盖旧运行。`DAY02_RUN_ID`、项目 commit、Polar commit、
GPU UUID、模型 revision、镜像 identity 和所有实际命令必须写入执行记录。

## 5. 启动前检查

### 5.1 项目与 Polar 版本

```bash
git status --short
git rev-parse HEAD
git -C "${POLAR_ROOT}" status --short
git -C "${POLAR_ROOT}" rev-parse HEAD
```

预期 Polar 输出必须严格等于锁定的 40 位 commit，且 Polar 工作树干净。

先确认 Day 1 仍通过：

```bash
python3 scripts/verify_day01_evidence.py \
  --lock configs/upstream-lock.yaml \
  --artifacts artifacts/day-01 \
  --project-config configs/project.yaml
```

允许 `PASS_WITH_NOTES`；其中 output token IDs 缺失将在本阶段通过 Gateway
completion record 继续核实。

### 5.2 GPU、端口和未知进程

```bash
nvidia-smi
ss -ltnp
docker ps
```

本阶段只需要一张 GPU。确认 `8000`、`8080`、`8100` 未被占用。发现未知 GPU、
端口或容器占用时停止并报告，不终止无法确认归属的进程。

### 5.3 Runtime image

```bash
docker image inspect polar-localhost-calculator:latest
docker image inspect \
  --format '{{ index .Config.Labels "io.polar.calculator-image-version" }}' \
  polar-localhost-calculator:latest
```

预期 layout version 为 `3`，image ID 与 lock 一致。身份一致时不要重建。镜像缺失
时才使用官方命令：

```bash
cd "${POLAR_ROOT}"
uv run python examples/calculator/build_image.py
```

重建后 identity 若变化，先更新执行证据，不得继续声称使用旧 digest。

## 6. 生成单 Gateway topology

先保存官方文件原件：

```bash
cp "${POLAR_ROOT}/examples/calculator/topology.sgl.yaml" \
  "${DAY02_RUN_DIR}/configs/topology.official.sgl.yaml"
```

在 `<run-dir>/configs/topology.rendered.sgl.yaml` 创建以下最小配置，将
`<ABSOLUTE_RUN_DIR>` 替换为当前运行目录的绝对路径：

```yaml
rollout:
  host: 127.0.0.1
  port: 8080
  public_url: http://127.0.0.1:8080
  save_dir: <ABSOLUTE_RUN_DIR>/raw/rollout_results

gateway:
  heartbeat_interval_seconds: 30
  completion_persistence:
    enabled: true
    max_field_bytes: 1048576
    queue_size: 128
  nodes:
    - id: localhost-node-01
      host: 127.0.0.1
      port: 8100
      public_url: http://127.0.0.1:8100
      max_init_workers: 1
      max_run_workers: 1
      max_postrun_workers: 1
      model_served: Qwen/Qwen3-4B-Instruct-2507
      inference:
        engine: sglang
        base_url: http://127.0.0.1:8000
```

该结构来自锁定 commit 的严格 `TopologyConfig`。加载验证：

```bash
cd "${POLAR_ROOT}"
uv run python -c "from polar.config import TopologyConfig; print(TopologyConfig.load('${DAY02_RUN_DIR}/configs/topology.rendered.sgl.yaml'))"
```

禁止添加未出现在 pinned schema 中的键。

## 7. 启动一张卡的 SGLang

复用 Day 1 已验证环境，从 `--mem-fraction-static 0.35`、8K context 和单任务开始。
先用 `--help` 确认当前版本支持所写参数，尤其 `--served-model-name`。启动命令和
最终 `/v1/models` 输出必须保存。

```bash
cd "${POLAR_ROOT}"
CUDA_VISIBLE_DEVICES=0 nohup uv run python -m sglang.launch_server \
  --model-path "${MODEL_ROOT}" \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
  --host 127.0.0.1 \
  --port 8000 \
  --context-length 8192 \
  --mem-fraction-static 0.35 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  > "${DAY02_RUN_DIR}/logs/sglang.log" 2>&1 &

SGLANG_PID=$!
printf '%s\n' "${SGLANG_PID}" > "${DAY02_RUN_DIR}/pids/sglang.pid"
```

健康检查：

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
nvidia-smi
```

若当前 SGLang 不支持 `--served-model-name`，删除该参数，并把 `/v1/models` 返回的
真实 model id 与 topology/Gateway 行为一起记录；不得静默假设名称一致。

## 8. 启动 Rollout Server 和 Gateway

每个服务使用独立日志和 PID 文件。先 Rollout Server，健康后再 Gateway。

```bash
cd "${POLAR_ROOT}"
nohup uv run polar serve_rollout \
  -c "${DAY02_RUN_DIR}/configs/topology.rendered.sgl.yaml" \
  > "${DAY02_RUN_DIR}/logs/rollout-server.log" 2>&1 &

ROLLOUT_PID=$!
printf '%s\n' "${ROLLOUT_PID}" > "${DAY02_RUN_DIR}/pids/rollout-server.pid"
```

然后检查：

```bash
curl -fsS http://127.0.0.1:8080/health
```

启动唯一 Gateway：

```bash
nohup uv run polar serve_gateway \
  -c "${DAY02_RUN_DIR}/configs/topology.rendered.sgl.yaml" \
  --node-id localhost-node-01 \
  > "${DAY02_RUN_DIR}/logs/gateway.log" 2>&1 &

GATEWAY_PID=$!
printf '%s\n' "${GATEWAY_PID}" > "${DAY02_RUN_DIR}/pids/gateway.pid"
```

检查：

```bash
curl -fsS http://127.0.0.1:8100/health
curl -fsS http://127.0.0.1:8080/rollout/status
```

只有 rollout status 显示该 Gateway healthy 后才提交任务。

## 9. Success task

优先使用 pinned 官方 runner，并只选择一个 built-in Harness：

```bash
cd "${POLAR_ROOT}"
uv run python examples/calculator/run.py --harness qwen_code \
  > "${DAY02_RUN_DIR}/logs/submit-success.log" 2>&1
```

锁定版本的 runner 会提交一个 task，默认产生多个 session。Day 2 只需从中选择一条
`COMPLETED` 且 evaluator 正常完成的 session 作为 success fixture；不要把多个
session 拼成一条 trajectory。

必须确认：

- task 从 submit 到 terminal status 完整结束；
- 至少一个 model completion 经过 Gateway 并持久化；
- session artifact 有 trajectory/evaluator/reward 证据；
- SGLang 日志、Gateway completion 数量和 session 时间窗口相互对应。

如果 `qwen_code` 在该服务器环境不可用，只能换成 pinned 示例实际支持的另一个
Harness，并在执行记录写明原因、版本和安装命令；不得修改 Polar 核心绕过。

## 10. Synthetic fault task

成功路径完成后才运行故障样本。复制相同 task payload，只改变一个明确记录的、
非破坏性参数。首选：把 evaluator `test_timeout` 调整为足以稳定触发 evaluator
timeout 的小正数，并保持模型、Harness、runtime 和其他字段不变。

要求：

- task metadata 或外部执行记录明确写 `synthetic_fault=true`；
- 保存修改前后的 payload diff；
- 不停止 SGLang、不杀 Gateway、不破坏 Docker daemon；
- fault 结果不能被解释成模型正常答错；
- 若超时未稳定触发，换用明确错误的 evaluator command，但仍需记录注入点；
- 不把 synthetic fault 用作模型 reward=0 训练数据。

提交方式必须基于 pinned `examples/calculator/run.py` 的
`build_task_payload()`，保存实际 POST payload 后再提交；不要手写一个结构不同的
假任务。任务的原始 POST payload、terminal API response 和 session artifact 全部
保存到 `<run-dir>/raw/fault/`。

## 11. Polar 原生 artifacts 与 fixture staging

锁定版本原生持久化结构包括：

```text
<save_dir>/task_<task-id>/ses_<session-id>.json
<save_dir>/task_<task-id>/sessions/<session-id>/completions/<NNNN>-<id>.json
```

其中 completion record 同时保存 agent original request、Gateway served request 和
backend response；session artifact 保存 terminal status、trajectory 和 evaluator/reward
信息。不要预设实际 JSON key，必须先查看真实文件。

我们的 fixture 文件名是项目侧 staging contract，不保证是 Polar 原生文件名：

```text
request.json   从已确认的 completion request 字段无损提取
response.json  从同一 completion response 字段无损提取
summary.json   对应完整 session artifact 或其去密副本
```

每个提取动作必须记录：

- 原始文件相对路径和 SHA256；
- 原始 JSON path；
- `direct`、`derived` 或 `missing` provenance；
- redaction rule 和 count；
- 提取后文件 SHA256。

如果真实 completion record 没有预期 request/response 字段，停止打包并更新
capture policy/schema；禁止为了让验证器通过而自行构造内容。

Staging 只允许：

```text
request.json
response.json
summary.json
gateway.log / rollout-server.log / sglang.log / runtime.log / evaluator.log（可选）
normalized-logs/（可选）
```

原始完整目录留在服务器 `<run-dir>/raw/`，大文件不提交 Git。

## 12. 打包与验证

Success 示例：

```bash
cd "${PROJECT_ROOT}"
python3 scripts/package_polar_fixture.py \
  --source-dir "${DAY02_RUN_DIR}/staging/success" \
  --output-dir tests/fixtures/polar/calculator_success \
  --fixture-type calculator_success \
  --fixture-id "polar-calculator-success-${DAY02_RUN_ID}" \
  --polar-commit f0e8343a7870abf6ec2366890f685881ceab92cb \
  --model-id Qwen/Qwen3-4B-Instruct-2507 \
  --model-revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --tokenizer-revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --runtime-image-identity sha256:0a967a5c33ee206f7524181cdd4ff866389016651645336eee93de1a7d970f49 \
  --harness qwen_code
```

Fault 使用相同 provenance，修改：

```text
--source-dir <run-dir>/staging/fault
--output-dir tests/fixtures/polar/calculator_fault
--fixture-type calculator_fault
--fixture-id polar-calculator-fault-<run-id>
```

`--missing-field` 只能在字段审计确认缺失后添加。`--redactions-json` 必须指向实际
redaction 记录，不能用空数组掩盖已发生的去密。

分别验证：

```bash
python3 scripts/verify_polar_fixture.py \
  tests/fixtures/polar/calculator_success

python3 scripts/verify_polar_fixture.py \
  tests/fixtures/polar/calculator_fault
```

两次都必须退出 `0`。失败时修复 staging/provenance，不修改验证器迁就数据。

## 13. 字段审计

根据真实 completion/session 文件填写 `notes/polar-field-map.md`：

- task、session、request/rollout identity；
- original request 与 served request 的差异；
- model/tokenizer/policy identity；
- prompt/output token IDs 和 sampled logprobs 的实际位置；
- tool call/result、patch、final output；
- terminal status、timeout、runtime/Harness/model backend error；
- evaluator command、exit、evidence 和 reward；
- 每个逻辑字段的 source file、JSON path 与 provenance。

未观察到的字段写 `missing`，不写“应该有”。

## 14. 停止服务

先确认 PID 文件属于本次 run，再按依赖逆序发送 `TERM`：

```text
Gateway
→ Rollout Server
→ SGLang
```

仅使用 `<run-dir>/pids/*.pid` 中记录的精确 PID。停止前再次读取进程命令行，避免
PID 复用。等待正常退出并保存退出状态；不要使用 `pkill` 或模糊匹配。

停止后检查：

```bash
nvidia-smi
ss -ltnp
docker ps
```

只处理能确认属于本次 Polar run 的容器或进程。

## 15. 验收门与退出条件

- [ ] Polar、SGLang、model 和 runtime identity 与 lock 一致；
- [ ] 单 Rollout Server、单 Gateway、单 SGLang topology 加载成功；
- [ ] success task 至少一条 session 正常完成 evaluator；
- [ ] synthetic fault 注入点、payload diff 和错误状态清楚；
- [ ] 所有模型调用经过 Gateway，至少一个 completion record 被持久化；
- [ ] raw session/completion 路径与 checksum 已记录；
- [ ] success/fault fixture 均通过验证器；
- [ ] 字段表区分 direct、derived 和 missing；
- [ ] 没有启动 Slime/Megatron；
- [ ] 服务按精确 PID 停止，GPU/端口释放。

完成时记录：

```text
DAY2_STATUS=COMPLETED_WITH_NOTES 或 FAIL
project_commit=
polar_commit=
run_id=
topology=
gpu_uuid=
success_task/session=
fault_task/session=
success_fixture_sha=
fault_fixture_sha=
missing_training_fields=
remaining_notes=
```

达到验收门后停止 Day 2，进入 Day 3 的最小 Coding/SWE rollout；不要继续增加
Calculator 并发、Harness 数量、Dashboard 或生产部署功能。
