# Day 3：Polar Coding/SWE 最小真实 Rollout Runbook

> 状态：`READY_AFTER_DAY2_CAPTURE`
>
> 目标是取得可区分任务成败与基础设施故障的 Coding fixtures。本阶段不运行
> Slime/Megatron，不追求 benchmark 分数，也不修改 Polar 核心。

## 1. 加速后的完成目标

```text
3 个可复现候选任务 runtime baseline
→ qwen_code + Qwen3-4B 单卡顺序 rollout
→ 1 条真实多轮 Coding trajectory
→ VALID_SUCCESS + VALID_FAILURE
→ 至少 2 种 INVALID_INFRASTRUCTURE 证据
→ clean replay
→ 去密、打包、验证、字段审计
```

先对 3 个候选各运行 1–2 个 sample。只对最有希望形成 reward variance 的任务补采，
不扩展为 500-task evaluation。若锁定模型没有产生真实 success，必须记录 `FAIL`，
不能把人工 patch、gold patch 或 Calculator success 冒充 Coding success。

## 2. 权威上游

只使用锁定 Polar commit：

```text
f0e8343a7870abf6ec2366890f685881ceab92cb
```

已核对的该 commit 官方文件：

- `examples/swebench_verified/README.md`
- `examples/swebench_verified/build_images.py`
- `examples/swebench_verified/dataset.py`
- `examples/swebench_verified/submit_swebench_tasks.py`
- `src/polar/trajectory/evaluator/swebench_harness.py`
- `src/polar/trajectory/evaluator/_patch_utils.py`

官方 runner 支持 `qwen_code`，task payload 由 `build_task_request()` 产生，evaluator
使用 `swebench_harness`。Evaluator 会提取 `patch.diff`，可在 fresh runtime 应用并执行
官方 test spec。

本项目继续使用 Day 1 锁定模型：

```text
Qwen/Qwen3-4B-Instruct-2507
revision cdbee75f17c01a7cc42f958dc650907174af0554
```

不要静默换成官方 README 的 Qwen3.6-27B，也不要把已下载但未验证的 Qwen3.5-4B
当成锁定模型。

## 3. 进入门槛

从项目仓库根目录执行：

```bash
python3 scripts/verify_polar_fixture.py tests/fixtures/polar/calculator_success
python3 scripts/verify_polar_fixture.py tests/fixtures/polar/calculator_fault
python3 -m unittest discover -s tests -v
```

两个 Day 2 fixture 必须退出 0。若仍只有 README，先完成 Day 2，不跳过。

检查工作树，禁止覆盖服务器未提交文件：

```bash
git status --short
git rev-parse HEAD
```

## 4. 本次运行目录

```bash
PROJECT_ROOT="$(pwd)"
POLAR_ROOT="/data/day-01-workspace/src/polar"
MODEL_ROOT="/data/day-01-workspace/hf-cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
DAY03_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-swebench"
DAY03_RUN_DIR="/data/day-01-workspace/artifacts/day-03/${DAY03_RUN_ID}"
```

创建：

```bash
mkdir -p "${DAY03_RUN_DIR}"/{configs,logs,pids,raw,validation}
mkdir -p "${DAY03_RUN_DIR}"/staging/{success,valid-failure,invalid-verifier-timeout,invalid-runtime-prepare}
```

所有实际命令、commit、包版本、GPU UUID、镜像 ID、task/session ID 和 checksum 写入
`<run-dir>/execution-record.md`。Raw artifacts 保留在服务器，不提交 Git。

## 5. 冻结 SWE evaluator 环境

```bash
git -C "${POLAR_ROOT}" status --short
git -C "${POLAR_ROOT}" rev-parse HEAD
cd "${POLAR_ROOT}"
uv pip install -e ".[swebench]"
uv pip freeze > "${DAY03_RUN_DIR}/validation/polar-swe-packages.txt"
```

Polar HEAD 必须严格等于锁定 commit且工作树干净。把实际安装的 `swebench`/`swegym`、
`datasets`、`qwen-code` 版本写入执行记录；`configs/upstream-lock.yaml` 中当前
`swegym_evaluator=NOT_FROZEN`，本阶段必须用真实安装结果替换，不能猜版本。

## 6. 候选任务与 runtime baseline

先让官方 loader 生成/复用缓存：

```bash
cd "${POLAR_ROOT}/examples/swebench_verified"
uv run python -c "from dataset import load_swebench_verified; print(len(load_swebench_verified()))"
```

然后从项目根目录生成确定性 shortlist：

```bash
cd "${PROJECT_ROOT}"
python3 scripts/select_swebench_candidates.py \
  --dataset-json "${HOME}/.cache/polar/swebench_verified.json" \
  --limit 10 \
  --distinct-repos \
  --output "${DAY03_RUN_DIR}/configs/candidate-shortlist.json"
```

从 shortlist 依次尝试，至少留下 3 个通过 baseline 的候选。对每个候选：

1. 用官方 `build_images.py --instance-id <ID>` 构建 runtime；
2. 保存 base image、runtime image ID 和 layout label；
3. 在容器中确认 `/testbed`、完整 base commit、Python 和 Git 可用；
4. 记录 cold/warm startup；
5. 不符合 base commit、镜像不可取或 baseline 不稳定时明确排除并继续下一项。

官方构建示例：

```bash
cd "${POLAR_ROOT}"
uv run python examples/swebench_verified/build_images.py --instance-id <INSTANCE_ID>
docker image inspect "polar-swebench-runtime:<SANITIZED_INSTANCE_ID>"
```

禁止为了凑够 3 个候选而跳过 runtime baseline。

## 7. 单 Gateway topology

保存官方 topology 原件，并创建：

```text
<run-dir>/configs/topology.rendered.sgl.yaml
```

内容：

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
    queue_size: 256
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

将 `<ABSOLUTE_RUN_DIR>` 替换为真实绝对路径，并验证：

```bash
cd "${POLAR_ROOT}"
uv run python -c "from polar.config import TopologyConfig; print(TopologyConfig.load('${DAY03_RUN_DIR}/configs/topology.rendered.sgl.yaml'))"
```

## 8. 启动服务

先确认 GPU 和 `8000/8080/8100` 无未知占用。只使用一张 GPU，候选任务顺序执行：

```bash
cd "${POLAR_ROOT}"
CUDA_VISIBLE_DEVICES=0 nohup uv run python -m sglang.launch_server \
  --model-path "${MODEL_ROOT}" \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507 \
  --host 127.0.0.1 \
  --port 8000 \
  --context-length 8192 \
  --mem-fraction-static 0.55 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  > "${DAY03_RUN_DIR}/logs/sglang.log" 2>&1 &
printf '%s\n' "$!" > "${DAY03_RUN_DIR}/pids/sglang.pid"
```

健康后启动 Rollout Server 和唯一 Gateway，命令与 Day 2 相同，只替换 topology 路径，
并把精确 PID 分别写入：

```text
pids/rollout-server.pid
pids/gateway.pid
```

只有 `/health` 和 `/rollout/status` 均确认 Gateway healthy 后才提交任务。

## 9. 真实 success / valid failure 采样

对 3 个 baseline 候选逐个调用锁定 commit 的官方 runner：

```bash
cd "${POLAR_ROOT}"
uv run python examples/swebench_verified/submit_swebench_tasks.py \
  --harness qwen_code \
  --instance-id <INSTANCE_ID> \
  --num-samples 2 \
  --timeout-seconds 1800 \
  --model-name Qwen/Qwen3-4B-Instruct-2507
```

每个任务完成后立即保存 task terminal response、session artifact 路径和 completion
数量，不并发提交下一个。官方 runner 的 topology 文件只用于得到 rollout URL；真正
运行的服务必须是本 run 的 rendered topology。

从结果中选择：

- `VALID_SUCCESS`：verifier 正常完成、`resolved=true`、reward=1；
- `VALID_FAILURE`：verifier 正常完成、`resolved=false`、reward=0。

两类都必须来自真实模型 rollout。若 3 个候选没有 success，只对一个 verifier 稳定、
成本最低的候选补采，整个 Day 3 最多再增加 4 个 sample；仍无 success 就记录失败，
不无限采样。

至少一条所选 trajectory 必须包含多个 Gateway completion、文件读取、代码修改、测试
执行和最终 patch。

## 10. 两类 infrastructure fault

成功/失败路径稳定后，基于官方 `build_task_request()` 复制同一 payload，只修改一个
明确参数，实际 POST payload 必须先写入 raw 目录。

### A. Verifier timeout

把 `evaluator.config.test_timeout` 改为稳定的小正数。要求结果明确显示 evaluator timeout。

### B. Runtime prepare failure

在 payload 的 `runtime.prepare` 末尾添加一个明确失败且无副作用的命令，例如
`exit 42`。不得破坏镜像、Docker daemon、共享目录或正在运行的服务。

两者都必须记录：

```text
synthetic_fault=true
原 payload SHA256
修改后 payload SHA256
唯一修改 JSON path
预期/实际失败阶段
terminal status/error
```

两类 infrastructure fixture 的 normalized evidence 必须保留：

```json
{"resolved": null, "reward": null}
```

禁止写成 reward 0。

## 11. Staging contract

从同一 session 的原始 artifact 无损提取：

```text
request.json
response.json
summary.json
patch.diff
verifier-evidence.json
replay.json                 # success / valid failure
fault-injection.json        # synthetic invalid
```

`verifier-evidence.json` 必须符合 `coding-verifier-evidence/v1`，其字段定义见
`configs/polar/coding/capture-policy.yaml`。`patch_sha256` 必须针对最终 `patch.diff`。

Valid outcome 的 `replay.json` 使用 `coding-clean-replay/v1`，至少记录：

```json
{
  "schema_version": "coding-clean-replay/v1",
  "clean_workspace": true,
  "patch_applied": true,
  "verifier_completed": true,
  "resolved": true
}
```

Failure 的 `resolved` 为 false。每个 derived staging 字段都要记录原始文件、JSON path、
转换规则和 checksum；未观察到的训练字段写 missing。

## 12. 打包与验证

使用通用打包器：

```bash
python3 scripts/package_polar_fixture.py \
  --source-dir "${DAY03_RUN_DIR}/staging/success" \
  --output-dir tests/fixtures/polar/coding_success \
  --fixture-type coding_success \
  --fixture-id "polar-coding-success-${DAY03_RUN_ID}" \
  --polar-commit f0e8343a7870abf6ec2366890f685881ceab92cb \
  --model-id Qwen/Qwen3-4B-Instruct-2507 \
  --model-revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --tokenizer-revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --runtime-image-identity <ACTUAL_RUNTIME_IMAGE_ID> \
  --harness qwen_code \
  --policy-version policy-v0
```

Valid failure 修改 source/output/type/id。Infrastructure fixture 使用：

```text
--fixture-type coding_invalid_infra
--synthetic-fault
```

一个目录只能保存一条 invalid fixture；第二种 invalid fault 放在服务器 raw evidence 和
pilot report 中，并记录 checksum。第一版 Git golden fixture 选择证据最完整的一种。

分别执行：

```bash
python3 scripts/verify_polar_fixture.py tests/fixtures/polar/coding_success
python3 scripts/verify_polar_fixture.py tests/fixtures/polar/coding_valid_failure
python3 scripts/verify_polar_fixture.py tests/fixtures/polar/coding_invalid_infra
python3 -m unittest discover -s tests -v
```

## 13. 字段审计与 pilot report

填写：

```text
notes/polar-coding-field-map.md
notes/task-pilot-report.md
```

字段表必须逐项给出 source file、JSON path 和 `direct/derived/missing`：identity、policy、
native token IDs、mask、sampled logprobs、tool events、patch、各组件状态、verifier、reward。

Pilot report 至少包含 3 个候选的 baseline、rollout 数量、success/failure/invalid、token/
turn/tool 数、verifier latency、可观察 Harness 行为，以及 Day 6 是否采用该任务做 paired comparison。

## 14. 停止与验收

仅使用本 run 的 PID 文件，先核对命令行，再按：

```text
Gateway → Rollout Server → SGLang
```

发送 TERM。禁止 `pkill`。检查 GPU 和端口释放。

完成门：

- [ ] Day 2 fixture 先通过；
- [ ] 3 个候选完成 runtime baseline；
- [ ] 至少一条真实多轮 Coding trajectory；
- [ ] success 与 valid failure 都有正常 verifier 证据；
- [ ] 两种 infrastructure fault 已记录且未映射 reward 0；
- [ ] valid outcome clean replay 一致；
- [ ] 三个 Git fixture 验证退出 0；
- [ ] 字段表和 pilot report 完成；
- [ ] 没有启动 Slime/Megatron；
- [ ] 精确 PID 停止且 GPU/端口释放。

结束记录：

```text
DAY3_STATUS=COMPLETED_WITH_NOTES 或 FAIL
project_commit=
polar_commit=
run_id=
swebench_evaluator_versions=
candidates=
real_rollouts=
success_task/session=
valid_failure_task/session=
invalid_faults=
fixture_verification=
missing_training_fields=
services_stopped=
remaining_notes=
```
