# Day 3 Coding Task Pilot Report

状态：`EXECUTED`（2026-08-11，run `20260811T030617Z-swebench`）。**无真实 VALID_SUCCESS** → DAY3_STATUS=FAIL（如实记录，不伪造）。

## 候选任务

| Instance | Repo | Base commit | Runtime image | Baseline | Exclusion reason |
|---|---|---|---|---|---|
| sphinx-doc__sphinx-8595 | sphinx-doc/sphinx | b19bce971e82f2497d67fdacdeca8db08ae0ba56 | polar-swebench-runtime:sphinx-doc-sphinx-8595 | PASS（test_empty_all 在 base 上失败 rc=4） | 保留 |
| sympy__sympy-20916 | sympy/sympy | 82298df6a51491bfaad0c6d1980e7e3ca808ae93 | polar-swebench-runtime:sympy-sympy-20916 | PASS（test_super_sub 在 base 上失败 rc=1） | 保留 |
| pytest-dev__pytest-5809 | pytest-dev/pytest | 8aba863a634f40560e25055d179220f0eefabe9a | polar-swebench-runtime:pytest-dev-pytest-5809 | PASS（test_create_new_paste 在 base 上失败 rc=4） | 保留（fixture 选择） |
| pylint-dev__pylint-4661 | pylint-dev/pylint | 1d1619ef913b99b06647d2030bddff4800abdf63 | polar-swebench-runtime:pylint-dev-pylint-4661 | FAIL | FAIL_TO_PASS 在 base commit 上已通过（rc=0），无 reward variance 空间 |
| scikit-learn__scikit-learn-14141 | scikit-learn/scikit-learn | 3d997697fdd166eff428ea9fd35734b6a8ba113e | polar-swebench-runtime:scikit-learn-scikit-learn-14141 | FAIL | 同上（test_get_deps_info rc=0） |
| django__django-12419 | django/django | 7fa1a93c6c8109010a6ff3f604fda83b604e0e97 | 未构建 | — | base 镜像经代理下载 30+ 分钟停滞未完成，排除 |

镜像说明：SWE-bench 官方镜像位于 docker.io（不可达）。官方 `make_test_spec`（swebench 4.1.0）在本机联网卡死 → 使用 dataset.py 的 fallback 镜像约定（xingyaoww 社区镜像 `sweb.eval.x86_64.<instance>`），经镜像代理 `docker.1ms.run` 拉取后构建 `polar-swebench-runtime`（layout v1，node 22 覆盖层）。base 镜像 ID：sphinx `95b17bea…`、sympy `3f4a…`、pytest `…`（详见 raw validation/baseline-*.json）。

## Rollout 统计

| Instance | Samples | Success | Valid failure | Invalid | Mean turns/tools | Verifier latency |
|---|---:|---:|---:|---:|---:|---:|
| sphinx-doc__sphinx-8595 | 2 | 0 | 2 | 0 | 1 turn / 0 tool | 短路（empty_generation） |
| sympy__sympy-20916 | 6 | 0 | 6 | 0 | 1 turn / 0 tool | 短路 |
| pytest-dev__pytest-5809 | 2 | 0 | 2 | 0 | 1 turn / 0 tool | 短路 |
| pytest（fault-rprep） | 1 | 0 | 0 | 1 | 0 turn | 未执行（INIT 失败） |
| pytest（fault-vtimeout） | 1 | 0 | 1* | 0 | 1 turn | 未执行（短路） |
| **合计** | **12** | **0** | **10** | **1** | — | — |

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
训练候选：无（本 run 无 VALID_SUCCESS，无 reward variance 可观察）
选择原因：qwen_code@0.14.5 + Qwen3-4B-Instruct-2507 环境下 agent 单轮退出（仅 1 次模型请求），
          无法产生多轮 trajectory / patch / resolved=true；reward 恒为 0，无正负样本区分。
排除任务：全部 3 候选（原因同上）；pylint/sklearn（baseline 不成立）；django（镜像不可得）
是否观察到 reward variance：否（0 正样本）
预计单 rollout 成本：~15–30s（INIT 3s + 单轮 agent 3s + evaluator 短路；远低于预算 1800s）
仍需解决的问题：
  1. harness-模型输出格式兼容：SGLang 返回 message.tool_calls=[] 且 content 为空，
     qwen-code CLI 无后续动作即退出（Day 2/3 跨两次复现）。需在 harness 配置或模型输出侧修复。
  2. swebench 官方镜像/ make_test_spec 的网络依赖（docker.io、swebench 库联网）——镜像已用代理
     + fallback 绕过；make_test_spec 卡死用 dataset.py fallback 约定替代。
  3. 若后续换用可多轮工作的 harness（如 opencode/codex）或修复输出格式，可重新评估
     sphinx/sympy/pytest 三个 baseline 通过的任务作为 Day 6 训练候选。
```

## 执行记录要点

```text
DAY3_STATUS=FAIL（无真实 VALID_SUCCESS；valid failure 与 invalid infra 证据已保存）
project_commit=c29cfed48b8dee8d0c9108762c6ecfe8233b0966（执行时 HEAD）
polar_commit=f0e8343a7870abf6ec2366890f685881ceab92cb
run_id=20260811T030617Z-swebench
swebench_evaluator_versions=swebench 4.1.0 / datasets 5.0.1（uv pip freeze 见 validation/polar-swe-packages.txt）
candidates=sphinx-8595(PASS) sympy-20916(PASS) pytest-5809(PASS) pylint-4661(EXCLUDE) sklearn-14141(EXCLUDE) django-12419(EXCLUDE)
real_rollouts=12（10 valid-failure + 1 invalid + 1 vtimeout-不可观察）
success=0 / valid_failure=10 / invalid=1
fixtures=coding_valid_failure + coding_invalid_infra（coding_success 缺失：无真实 success）
services_stopped=待执行
```
