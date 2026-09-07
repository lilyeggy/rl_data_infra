# v3 — SWE-bench Verified 真实 benchmark 集成（已归档）

> 版本：v3-swebench · 状态：已完成 · 日期：2026-08-19
> 目标：把**真实第三方编码 benchmark**（SWE-bench Verified）接入我们的数据面，
> 证明"真实 harness 数据 → 产生有效数据 → 可用于 agentic RL"不只在自造任务上成立。

---

## 1. 一句话

> 用真实 Pi + deepseek-v4-flash 求解 10 个 SWE-bench Verified 真实 GitHub issue，
> 通过完整数据面捕获轨迹，用**真实隐藏测试**（FAIL_TO_PASS）做 verifier，
> **7/10 被真实测试验证通过**；7 条验证成功的 teacher 轨迹直接成为 SFT 候选（offline agentic RL 数据）。

---

## 2. 为什么做这个

| | 之前（自造任务 suite） | 现在（SWE-bench） |
|---|---|---|
| 任务来源 | 我们自己生成的 read/find/count | **真实 GitHub issue + 真实测试** |
| verifier | 自写 JSON 精确匹配 | **真实 FAIL_TO_PASS 测试** |
| 说服力 | 机制验证充分 | 面试官认识 SWE-bench，真实 benchmark 背书 |
| 数据价值 | 3 条 teacher 轨迹 | **10 条真实 agent 轨迹，7 条测试验证通过** |

---

## 3. 方法

### 3.1 实例选择（10 个，轻量纯 Python）

约束：服务器**无 Docker**，用 venv + pytest 评估 → 选 Python 3.12 兼容、依赖少、diff 小的实例：

```text
pallets__flask-5014      flask 2.3    <15min
psf__requests-5414       requests 2.26 <15min    ✅
psf__requests-6028       requests 2.27 15min-1h  ✅
pytest-dev__pytest-10081 pytest 7.2   <15min
pytest-dev__pytest-10051 pytest 7.2   15min-1h
sympy__sympy-23534       sympy 1.11   <15min    ✅
sympy__sympy-24539       sympy 1.12   <15min    ✅
sympy__sympy-23824       sympy 1.12   15min-1h  ✅
sympy__sympy-24213       sympy 1.12   15min-1h  ✅
sympy__sympy-23950       sympy 1.12   15min-1h  ✅
```

选择文件：`experiments/swebench/manifest-v3.json`（redacted，无 gold patch）。
完整数据来自 `princeton-nlp/SWE-bench_Verified`（validation split, 500 实例）。

### 3.2 执行（`experiments/swebench/run_swebench.py`）

```text
1. 镜像克隆：gitee / gh-proxy / gitcode（服务器访问不了 github.com）
2. 每个实例：git worktree @ base_commit + venv + pip install -e . （含 test deps）
3. 真实 Pi（opencode-go/deepseek-v4-flash）+ read/bash/write/edit/grep/find/ls/glob
   在 worktree 里求解，NDJSON 全量捕获（决策 + 工具调用 + token）
4. 评估：
   - 新建 clean worktree @ base_commit
   - 应用 agent diff（tracked）+ 未跟踪文件快照
   - 应用 test_patch（gold，对 agent 隐藏）
   - venv + pytest 跑 FAIL_TO_PASS（+ PASS_TO_PASS 前 2 条 sanity）
5. 数据面组装（src/real_swebench.py）→ canonical 事件/Episode/metrics/SFT 候选
```

### 3.3 防假阳性验证

```text
挑选已解决实例跑基线对照（clean base + test_patch，不加 agent patch）：
  psf__requests-5414 : 基线 FTP 测试 FAIL  ➜ 修复是 agent 真实完成的 ✅
gold patch 与 test_patch 全程对 agent 隐藏
```

agent patch 抽查确认为真实代码修复：
```diff
# requests-5414 (1 行修复)
-        elif host.startswith(u'*'):
+        elif host.startswith((u'*', u'.')):
# sympy-23950 (实现 Contains.as_set)
+    def as_set(self):
+        x, s = self.args
+        return s.as_relational(x).as_set()
```

---

## 4. 结果

```text
实例数           : 10
verified pass    : 7  (70% on this curated easy subset)
FAIL_TO_PASS     : 全部通过（含 PASS_TO_PASS sanity）
SFT candidates   : 7  （验证成功的 teacher 轨迹）
on-policy 候选   : 0  （deepseek 是远程 API，无 logprobs —— 预期边界）
```

数据面产物（`artifacts/swebench-v3/package-v3/`）：

```text
raw-events.jsonl        6.8MB   canonical TraceEvents（全部事件，append-only）
episodes.jsonl          6.9MB   组装后的 Episode（决策谱系、token、工具序列）
metrics.json                    每 Episode 的 token/耗时/工具调用数
verifier-evidence.json          每实例真实测试结果
training-candidates.json        7 条验证成功的 SFT 候选轨迹
summary.json                    checksums + claim boundary
```

---

## 5. 诚实边界（面试表述）

```text
- 这是"真实 benchmark × 我们的数据面"的集成证据，不是 benchmark 声称
  （10 个刻意选的轻量实例，pass rate 不代表 SWE-bench 整体水平）
- verifier 是真实测试，但无 Docker → venv + pytest（部分环境简化）
- 7 条成功轨迹是 offline（SFT 蒸馏）候选；on-policy RL 仍需本地可训练模型
- deepseek 内部决策仍是黑盒（无 HARNESS_DECISION 运行时信号）
- 实例含 difficulty 分级，但整体偏易（为保证无 Docker 可评估）
```

---

## 6. 与整体故事的连接

```text
真实 benchmark 数据 → 数据面（canonical + verifier + gate）
                            ↓
7 条验证成功的 teacher 轨迹 → offline SFT/偏好数据（直接可用）
真实测试 = 最严格的 verifier（无假阳性，基线对照验证）
```

下一步（Track C）：把这些真实 agent 轨迹作为 SFT 训练数据，微调本地 Qwen-7B，
验证"真实 benchmark 数据 → 训练本地模型"。同时回到 15-task suite 做 Track A
（真实 harness 策略迭代）和 RL 机制验证。
