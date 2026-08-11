# Day 3 Coding Task Pilot Report

状态：`EXECUTED`（2026-08-11，run `20260811T030617Z-swebench`）。

> 第一阶段（无补丁）无 VALID_SUCCESS → 记录 FAIL。随后定位根因并给 SGLang v0.5.13 打本地补丁
> `patches/sglang/qwen3-tool-call-fix.patch`（sha256 `136fb0af…`，修复思考块内工具调用剥离 +
> JSON 风格 tool_call 解析），重跑 pytest-5809 → **2/2 session VALID_SUCCESS**。
> 最终：DAY3_STATUS=COMPLETED_WITH_NOTES（valid success + valid failure + invalid infra 齐备）。

## 候选任务

| Instance | Repo | Base commit | Runtime image | Baseline | Exclusion reason |
|---|---|---|---|---|---|
| sphinx-doc__sphinx-8595 | sphinx-doc/sphinx | b19bce971e82f2497d67fdacdeca8db08ae0ba56 | polar-swebench-runtime:sphinx-doc-sphinx-8595 | PASS（test_empty_all 在 base 上失败 rc=4） | 保留 |
| sympy__sympy-20916 | sympy/sympy | 82298df6a51491bfaad0c6d1980e7e3ca808ae93 | polar-swebench-runtime:sympy-sympy-20916 | PASS（test_super_sub 在 base 上失败 rc=1） | 保留 |
| pytest-dev__pytest-5809 | pytest-dev/pytest | 8aba863a634f40560e25055d179220f0eefabe9a | polar-swebench-runtime:pytest-dev-pytest-5809 | PASS（test_create_new_paste 在 base 上失败 rc=4） | 保留（fixture 选择） |
| pylint-dev__pylint-4661 | pylint-dev/pylint | 1d1619ef913b99b06647d2030bddff4800abdf63 | polar-swebench-runtime:pylint-dev-pylint-4661 | FAIL | FAIL_TO_PASS 在 base commit 上已通过（rc=0），无 reward variance 空间 |
| scikit-learn__scikit-learn-14141 | scikit-learn/scikit-learn | 3d997697fdd166eff428ea9fd35734b6a8ba113e | polar-swebench-runtime:scikit-learn-scikit-learn-14141 | FAIL | 同上（test_get_deps_info rc=0） |
| django__django-12419 | django/django | 7fa1a93c6c8109010a6ff3f604fda83b604e0e97 | polar-swebench-runtime:django-django-12419 | **PASS（后补）** | 当时镜像并发下载慢被误判为停滞；复查确认镜像已拉全（4.1GB），构建后 baseline 通过（test_middleware_headers rc=2） |

镜像说明：SWE-bench 官方镜像位于 docker.io（不可达）。官方 `make_test_spec`（swebench 4.1.0）在本机联网卡死 → 使用 dataset.py 的 fallback 镜像约定（xingyaoww 社区镜像 `sweb.eval.x86_64.<instance>`），经镜像代理 `docker.1ms.run` 拉取后构建 `polar-swebench-runtime`（layout v1，node 22 覆盖层）。base 镜像 ID：sphinx `95b17bea…`、sympy `3f4a…`、pytest `…`（详见 raw validation/baseline-*.json）。

## Rollout 统计（补丁前）

| Instance | Samples | Success | Valid failure | Invalid | Mean turns/tools | Verifier latency |
|---|---:|---:|---:|---:|---:|---:|
| sphinx-doc__sphinx-8595 | 2 | 0 | 2 | 0 | 1 turn / 0 tool | 短路（empty_generation） |
| sympy__sympy-20916 | 6 | 0 | 6 | 0 | 1 turn / 0 tool | 短路 |
| pytest-dev__pytest-5809 | 2 | 0 | 2 | 0 | 1 turn / 0 tool | 短路 |
| pytest（fault-rprep） | 1 | 0 | 0 | 1 | 0 turn | 未执行（INIT 失败） |
| pytest（fault-vtimeout） | 1 | 0 | 1* | 0 | 1 turn | 未执行（短路） |
| **补丁前合计** | **12** | **0** | **10** | **1** | — | — |

- 10 个补丁前 session 均为**单轮**（prompt 14684–14907 tokens，response 56–194 tokens），无工具执行、无 patch（`empty_generation=true`）。*fault-vtimeout 注入不可观察，不计入。

## Rollout 统计（SGLang 补丁后）

| Instance | Samples | Success | Valid failure | Invalid | Mean turns/tools | Verifier latency |
|---|---:|---:|---:|---:|---:|---:|
| pytest-dev__pytest-5809 | 2 | **2** | 0 | 0 | 9-10 turns / 9 tools | ~11-13s（含 fresh eval runtime） |
| **补丁后合计** | **2** | **2** | **0** | **0** | — | — |

- 补丁后 session `sk-polar-0f8c6689…`：**10 轮模型调用**（todo_write/read_file/grep×2/edit×2/write_file/run_shell + todo 更新），9 个结构化工具调用 + 1 个 content 文本 edit；`resolved=true`、`reward=1.0`；verifier `FAIL_TO_PASS test_create_new_paste` 通过 + `PASS_TO_PASS` 3 项无回归（exit 0）；clean replay（重建 patch）4 passed、resolved=true。
- 另一 session `sk-polar-c0e38472…`：6 轮，同样 resolved=true、reward=1.0。

- 10 个 valid-failure session 均为**单轮**：prompt 14684–14907 tokens，response 56–194 tokens，`finish_reason=stop`，无工具执行、无 patch（`empty_generation=true`）。
- *fault-vtimeout 的 session 状态为 COMPLETED/reward 0，但注入的 test_timeout=0.1 **不可观察**（空 patch 短路，测试未执行），不计入真实 valid failure 证据。

## Infrastructure faults

| Fault | Natural/synthetic | Injection path | Observed status | Reward | Evidence SHA256 |
|---|---|---|---|---|---|
| verifier timeout | synthetic | `evaluator.config.test_timeout` 60→0.1（SWE-bench 无显式默认，harness 内部默认 1200） | session COMPLETED；report.test_timeout=false（empty_generation 短路，**未触发**） | null（verifier 未执行） | 见 raw/pytest-dev--pytest-5809-fault-verifier-timeout/fault-injection.json |
| runtime prepare failure | synthetic | `runtime.prepare[+]` 追加 `exit 42` | session ERROR：`runtime initialization failed: prepare action 1 failed with exit code 42`（INIT 阶段，run_ms=0） | **null** | 见 raw/pytest-dev--pytest-5809-fault-runtime-prepare/fault-injection.json |

Git golden fixture `coding_invalid_infra` 选择证据最完整的 **runtime prepare failure**（resolved=null、reward=null 已入 fixture）。

## Clean replay（valid failure，pytest-5809）

- fresh runtime 容器 → `git checkout 8aba863a…` → 应用空 patch.diff（empty）→ 执行 verifier 命令（pytest test_create_new_paste）
- 结果：`exit_code=4, verifier_completed=true, timed_out=false, resolved=false, reward=0`；`matches_original=true` ✓（与原 session outcome 一致）

## Day 6 选择

```text
训练候选：pytest-dev__pytest-5809（补丁后 2/2 success，可形成 reward variance）
选择原因：SGLang 补丁后 qwen_code 多轮工作正常（10 轮工具循环），FAIL_TO_PASS/PASS_TO_PASS
          稳定；success（resolved=true）与 failure（resolved=false）样本均可产生。
排除任务：pylint/sklearn（baseline 不成立）；django（baseline 后补通过，可作后备）
是否观察到 reward variance：是（补丁后 success reward=1 vs failure reward=0）
预计单 rollout 成本：~30-60s（10 轮工具 + evaluator + fresh eval runtime；远低于预算 1800s）
仍需解决的问题：
  1. SGLang 补丁为本地维护（v0.5.13 + qwen3-tool-call-fix.patch）；后续升级/换版需重验。
  2. 最后一次 edit 调用的结构化 tool_calls 为空（content 文本携带）——qwen-code 自己解析成功，
     但说明 tool parser 对超大 JSON 的流式解析仍不完整（第 10 轮），已记录。
  3. patch 重建依赖工具调用参数精确性；evaluator 原始 patch 随 session 目录清理（Polar 侧改进空间）。
```

## 执行记录要点

```text
DAY3_STATUS=COMPLETED_WITH_NOTES（SGLang 本地补丁后：valid success 2 + valid failure 10 + invalid 1）
project_commit=c29cfed48b8dee8d0c9108762c6ecfe8233b0966（执行时 HEAD）
polar_commit=f0e8343a7870abf6ec2366890f685881ceab92cb
run_id=20260811T030617Z-swebench
swebench_evaluator_versions=swebench 4.1.0 / datasets 5.0.1（uv pip freeze 见 validation/polar-swe-packages.txt）
sglang_local_patch=patches/sglang/qwen3-tool-call-fix.patch（sha256 136fb0af…；reasoning 剥离 + JSON tool_call 解析）
candidates=sphinx-8595(PASS) sympy-20916(PASS) pytest-5809(PASS) pylint-4661(EXCLUDE) sklearn-14141(EXCLUDE) django-12419(PASS 后补)
real_rollouts=14（补丁前 12 + 补丁后 2）
success=2 / valid_failure=10 / invalid=1（补丁后全部 success）
fixtures=三套齐备：coding_success + coding_valid_failure + coding_invalid_infra（验证均 exit 0）
services_stopped=已停止（Gateway→Rollout→SGLang，精确 PID）
```
