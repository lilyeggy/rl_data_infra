# Agent Improvement Data Plane

把一次真实 Agent/Harness 执行转化为可验证、可归因、可训练的证据，分别驱动：

1. Harness 改进：诊断、受控 A/B、回归 Gate；
2. 模型改进：认证后的 SFT、Preference 和 Agentic RL 数据。

```text
Task + Environment + Harness + Policy
                  │
        Local Docker Launcher (default)
        Pi Direct / Polar Live (plugins)
                  │
       Model Proxy + Tool/Sandbox + Hook
                  │
      immutable execution evidence
                  │
      assemble → verify → certify
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
Harness analysis         Learning datasets
compare / gate           SFT / pref / RL
                              │
                         Slime trainer
```

## 当前状态

已完成且保留的可信内核：

- 不可变 `TraceEvent` 与确定性 `AgentEpisode`；
- append-only storage、checksum、lineage、quarantine；
- task failure 与 infrastructure invalid 分离；
- verifier-backed Episode certification；
- Harness metrics、attribution、paired comparison、regression gate；
- policy fingerprint 与 fail-closed training eligibility；
- 统一 `ExecutionIdentity`、`ExecutionBundle`、producer 边界和 consumer certification 入口；
- content-addressed `DatasetManifest` / compiler，并阻断 task split leakage 与残缺 preference pair；
- Polar 官方 HTTP boundary、batch producer 和 stable `TaskResult/SessionResult/Trajectory` 严格适配。
- Local Docker Launcher：统一 identity 注入、资源限制、只读 root、cap-drop、no-new-privileges 与显式网络策略；
- Model Proxy evidence：外部 API 保持 observability-only，受控服务原生 token/logprob 才能声明 RL capability；
- OpenAI-compatible Model Proxy HTTP forwarder、逐 execution bearer token 和耐久 model-evidence log；
- `ExecutionRunManifest` 与 Local Run finalizer，将 events/model evidence/launcher artifact 严格 join 为 Episode/Bundle；
- `execute-local` 单事务编排器，统一负责 proxy、Harness、verifier、terminal event、finalize 与 cleanup；
- Harness event ingress/SDK：Harness 只提交事实描述，identity、sequence、timestamp 和 event ID 由宿主生成；
- stdout/stderr、workspace diff/patch 与 verifier report 的内容寻址落盘；
- 单 GPU phase/evidence 状态契约，阻断 serving/training 资源重叠和越级训练。
- `ExecutionBundle` 严格组装器与 bundle-bound RL certification；
- Slime admission envelope：重新核验 manifest/decision/bundle/artifact 后才输出 token-faithful traces；
- 单 GPU 原子状态存储、checksum CAS、断点恢复和不执行命令的 dry-run plan。
- A6000 上的真实 Pi → Model Proxy → 14B LoRA → Verifier → ExecutionBundle live 闭环；
- base 与 SFT candidate 的固定 unseen DEV 对照，以及读取报告结论的 fail-closed supervisor Gate；
- APPS clean-v2 训练包（含数据质量复盘后的 fail-closed 清洗）与 14B LoRA SFT；
- 单卡 on-policy GRPO LoRA trainer（`grpo-lora-trainer/v1`）与 apps-rl-cycle-001/002/003 三轮 policy 更新；
- 官方 EvalPlus/BigCodeBench 三基准对比报告与固定 APPS holdout 50 题评测。

正在建设：

- 在真实 Polar 服务上验证可选 live rollout，而不只依赖 transport unit test；
- 在真实 live session 上验证 Harness capture 与 Polar artifact 的 bundle join；
- 将单 GPU phase contract 接到真实进程启停与 artifact 持久化；
- 把已 admission 的 envelope 交给官方 Polar–Slime bridge 并完成正式训练验证。

当前没有完成、不得宣称完成：

- Slime 端到端模型更新；
- 通过固定 unseen DEV Gate 的 policy-v1；当前 canonical-v2 SFT candidate 为 `REJECT / NO_IMPROVEMENT`，apps-rl-cycle-001/002/003 candidate 在固定 APPS holdout 上未超过 SFT-v0（50% vs 50%），HumanEval 与 base 持平；
- 大规模 Agentic RL；
- benchmark 泛化提升。

## 快速验证

核心代码只依赖 Python 3.10+ 标准库：

```bash
python3 -m unittest discover -s tests -v
python3 -m src.cli inspect \
  --episodes artifacts/v1-observability/episodes.jsonl \
  --episode-id episode-tool-loop
python3 -m src.cli quality-report \
  --artifacts artifacts/current/producer-artifacts.jsonl \
  --decisions artifacts/current/eligibility-decisions.jsonl \
  --output artifacts/current/data-quality-report.json
python3 -m src.cli admit-slime \
  --manifest artifacts/current/dataset-manifest.json \
  --decisions artifacts/current/eligibility-decisions.jsonl \
  --bundles artifacts/current/execution-bundles.jsonl \
  --artifacts artifacts/current/producer-artifacts.jsonl \
  --output artifacts/current/slime-admission.json
python3 -m src.cli prepare-local \
  --spec configs/local-execution.example.json \
  --output-dir artifacts/prepared-run
python3 -m src.cli execute-local \
  --spec configs/local-execution.example.json \
  --output-dir artifacts/run-001 \
  --upstream-auth-env MODEL_SERVER_AUTHORIZATION
```

`quality-report` 会严格重载每条版本化记录；未知字段、旧 schema、损坏 JSON
都会直接失败，不会被当作可训练数据跳过。
`admit-slime` 会从磁盘重新核验整条 checksum/identity/policy lineage，并通过
原子替换写出 trainer-neutral admission envelope。

旧 demo 已迁到 `examples/legacy_scenarios/`，只由回归测试直接调用，不再暴露为生产 CLI 命令。旧硬件 lock、主机脚本和自定义训练代码见 `archive/`。

## 权威阅读顺序

1. [项目范围](PROJECT_SCOPE.md)
2. [实施计划](PROJECT_PLAN.md)
3. [架构决策 ADR-0002](docs/adr/0002-agent-improvement-data-plane.md)
4. [数据契约](docs/data-contract.md)
5. [历史归档索引](archive/README.md)

## 组件边界

- Local Docker Launcher：默认运行真实 Harness 和隔离 workspace；
- Polar：可选地运行批量 Harness rollout、重建 token-faithful trajectory；
- 本项目：不可变证据、验证、认证、数据集、Harness/模型双闭环；
- Slime：正式 GRPO/PPO 训练、Megatron、SGLang 和权重同步；
- 旧自定义 GRPO/SFT：已归档；当前单卡 RL 更新使用独立极简 GRPO LoRA trainer（`scripts/train_grpo_lora.py`），正式 trainer 仍以 Slime 接入为目标。
