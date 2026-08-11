# Day 3 执行记录（Polar Coding/SWE rollout）

状态：`DAY3_STATUS=FAIL`（无真实 VALID_SUCCESS；valid failure 与 invalid infra 证据完整保存）
run_id：`20260811T030617Z-swebench`
执行日期：2026-08-11T03:06Z – 04:2xZ（本地 2026-08-11 11:06–12:2x）

## 版本与身份

```text
project_commit:      c29cfed48b8dee8d0c9108762c6ecfe8233b0966
polar_commit:        f0e8343a7870abf6ec2366890f685881ceab92cb（工作树干净）
sglang_commit/ver:   28b095c01005d4a3a2a5b637b7d028b07fba31b2 / 0.5.13
model:               Qwen/Qwen3-4B-Instruct-2507
model/tokenizer rev: cdbee75f17c01a7cc42f958dc650907174af0554
harness:             qwen_code（@qwen-code/qwen-code@0.14.5）
evaluator:           swebench_harness（swebench 4.1.0，datasets 5.0.1）
gpu_uuid:            GPU-4288d6c9-0384-25dc-1071-9bea701b9374（GPU 0）
sglang context:      32768（preflight：qwen_code 实测请求 ~14684–14907 input + max_tokens 8000，
                     Day 2 已证 8K/16K 不足；mem-fraction-static 0.55）
```

## Runtime baseline（validation/baseline-*.json）

- sphinx-doc__sphinx-8595：base b19bce97…，/testbed OK，py3.9.19，FAIL_TO_PASS 在 base 上 rc=4 → **PASS**
- sympy__sympy-20916：base 82298df6…，test_super_sub rc=1 → **PASS**
- pytest-dev__pytest-5809：base 8aba863a…，test_create_new_paste rc=4 → **PASS**
- pylint-4661 / sklearn-14141：FAIL_TO_PASS 在 base 上 rc=0（已通过）→ **EXCLUDE**
- django-12419：base 镜像经代理 30+ 分钟传输停滞 → **EXCLUDE**

镜像：SWE-bench 官方 docker.io 不可达；swebench 4.1.0 make_test_spec 联网卡死 → 用 dataset.py fallback 镜像约定（xingyaoww/sweb.eval.x86_64.<instance>），经代理 docker.1ms.run 拉取，构建 polar-swebench-runtime（layout v1）。

## 服务（pids/）

```text
sglang:   2534902（GPU 0，--served-model-name Qwen/Qwen3-4B-Instruct-2507 --context-length 32768 --mem-fraction-static 0.55）
rollout:  2536171（8081；8080 被系统账户 server 进程占用 → 端口偏差记录）
gateway:  2536257（8100，completion_persistence enabled）
```

## Rollout 结果

- 12 次提交：sphinx×2、sympy×6（含补采 4）、pytest×2、fault-rprep×1、fault-vtimeout×1
- 10 个 normal session：均 COMPLETED、单轮（prompt 14684–14907 / response 56–194）、`empty_generation=true`、resolved=false、reward=0.0
- **无 VALID_SUCCESS**（qwen_code 单轮退出：1 次模型请求后 CLI 静默结束，无工具执行/无 patch）
- fault-rprep：ERROR `runtime initialization failed: prepare action 1 failed with exit code 42`（INIT，run_ms=0）
- fault-vtimeout：注入 test_timeout=0.1 不可观察（empty patch 短路，report.test_timeout=false）

## Evidence（raw/）

- task-terminal / submit-payload(+sha256) / fault-injection.json 均保存
- valid failure（pytest caf2bff2）：completion record `0001-msg_b3ffe0123480.json`（input_token_ids 14907、token_ids 191、logprobs 191）
- clean replay：fresh runtime + base commit + 空 patch + verifier → exit 4、resolved=false、matches_original=true

## Fixture

- tests/fixtures/polar/coding_valid_failure（6 files）verify exit 0
- tests/fixtures/polar/coding_invalid_infra（6 files，synthetic_fault=true，resolved=null reward=null）verify exit 0
- coding_success：未打包（无真实 success，不伪造）
- Calculator fixtures 仍通过；77 单元测试通过；scripts/tests compileall 通过（src/**/.venv 内 3.12 第三方包在系统 3.10 下 SyntaxError，环境差异已记录）

## 停止

Gateway(2536257) → Rollout(2536171) → SGLang(2534902)，TERM 等待退出；GPU/端口检查见执行后快照。

## 未启动

未启动 Slime/Megatron/GRPO；未修改 Polar 核心；未伪造 success/patch/reward。
