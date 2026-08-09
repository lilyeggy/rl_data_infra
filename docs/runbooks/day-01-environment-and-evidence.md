# Day 1 环境基线与证据复现 Runbook

> 当前状态：服务器操作延期。本文件用于冻结操作步骤；当前阶段优先完成无需 GPU/服务器的本地校验工具。

## 1. 范围

Day 1 只负责建立可信、可追溯的运行基线：

- 主机、GPU、驱动、CUDA、磁盘和容器事实；
- Polar、SGLang、Slime、Megatron-LM、mbridge 与模型 revision；
- SGLang inference smoke；
- Megatron TP2 forward/backward、optimizer step 和 checkpoint reload smoke；
- rollout/train staged sharing 决策；
- artifact checksum 与已知风险。

Day 1 不实现 `RolloutRecord`、Source Adapter、Processor、GRPO 或 Trainer Adapter。

## 2. 自研与上游边界

我们自己维护的只有：

```text
scripts/check_gpu_host.sh
scripts/capture_day01_evidence.sh
scripts/verify_day01_evidence.py
configs/upstream-lock.yaml
docs/adr/0001-freeze-upstream-baseline.md
```

Polar、SGLang、Slime 和 Megatron 的运行逻辑使用锁定的上游版本，不在 Day 1 重写。

## 3. 服务器目录约定

```text
/data/day-01-workspace/
├── src/
│   ├── polar/
│   ├── sglang/
│   ├── slime/
│   ├── Megatron-LM/
│   ├── mbridge/
│   └── mbridge-iseekyan/
├── models/
├── hf-cache/
├── artifacts/day-01/
├── configs/
├── patches/
└── scripts/
```

模型、Docker data-root、缓存和训练输出均放在 `/data`。

## 4. 服务器恢复后执行

### 4.1 拉取代码并确认 GPU 空闲

```bash
cd /data/day-01-workspace
git pull
nvidia-smi
```

不要终止无法确认归属的 GPU 进程。

### 4.2 重新采集轻量事实

```bash
bash scripts/capture_day01_evidence.sh /data/day-01-workspace
```

预期生成：

```text
artifacts/day-01/host-preflight.txt
artifacts/day-01/gpu-topology.txt
artifacts/day-01/source-versions.txt
artifacts/day-01/python-packages.txt
artifacts/day-01/container-images.txt
artifacts/day-01/paths.txt
```

该步骤只读检查环境，不启动模型、不运行训练。

### 4.3 审计版本漂移

检查：

- 核心上游仓库是否仍为完整的锁定 commit；
- dirty 仓库是否有可复现 patch 和 checksum；
- 模型/tokenizer snapshot revision 是否变化；
- PyTorch、CUDA、SGLang、Slime/Megatron 关键版本是否变化；
- runtime image identity 是否变化。

只有发生相关漂移时才重跑对应 smoke：

| 漂移 | 需要重跑 |
|---|---|
| SGLang/PyTorch/model revision | SGLang smoke |
| Slime/Megatron/训练 patch | Megatron TP2 smoke |
| 驱动/CUDA | 两侧 smoke |
| Calculator runtime image | Day 2 Calculator |
| 只有文档变化 | 不重跑 GPU |

### 4.4 保存本地 Megatron SDPA patch

```bash
mkdir -p patches/megatron
git -C src/Megatron-LM diff HEAD \
  -- megatron/core/transformer/dot_product_attention.py \
  > patches/megatron/sdpa-sm120-8k.patch

git -C src/Megatron-LM apply \
  --check \
  /data/day-01-workspace/patches/megatron/sdpa-sm120-8k.patch

sha256sum patches/megatron/sdpa-sm120-8k.patch
```

将 patch 相对路径与 SHA256 写入 `configs/upstream-lock.yaml`。文档只能声称该 patch 的目标语义一致；TP 数值等价性和 dropout RNG 一致性仍需单独验证。

### 4.5 运行机器验收

```bash
python scripts/verify_day01_evidence.py \
  --lock configs/upstream-lock.yaml \
  --artifacts artifacts/day-01 \
  --project-config configs/project.yaml
```

退出码：

```text
0 = PASS 或 PASS_WITH_NOTES
1 = 验收失败
2 = 配置或证据无法解析
```

## 5. 当前已知事实

已有服务器证据支持：

- 2 × RTX PRO 6000 Blackwell，驱动 580.159.03，CUDA 13.0；
- Qwen/Qwen3-4B-Instruct-2507；
- SGLang TP1 8K smoke；
- Megatron TP2 4096/8192 optimizer step 与 checkpoint reload；
- 锁文件列出的主要 artifact checksum 一致。

当前保留的 notes：

1. SGLang response 有 token 文本和 logprob，但缺少完整数值 output token IDs；不得通过重新 tokenize 文本伪造。
2. 本地 SDPA patch 需要提交实际 patch 文件，而不只是 checksum。
3. `backward-smoke-512-tp2.json` 的 reload 校验失败，必须明确标记为失败实验，不能计入通过结果。
4. rollout/process/train/reload 的 staged GPU 运行方式必须保留在项目配置和 ADR 中。
5. 本地 Docker image ID 可标识同机镜像，但干净复现还需要 registry RepoDigest 或导出镜像 tar checksum。

## 6. Day 1 完成定义

满足以下条件后标记：

```text
DAY1_STATUS=COMPLETED_WITH_NOTES
```

- 轻量环境事实无未解释漂移；
- lock 中所有关键 commit/revision/image identity 已冻结；
- TP2 smoke 的有效通过结果满足 `A != B` 且 `B == C`；
- evidence checksum 全部一致；
- dirty patch 可从仓库内容重现；
- 所有已知风险显式保留；
- 未因 Day 1 工作提前实现 Day 4–6 的业务逻辑。

完成后停止扩展 Day 1，进入 Day 2 Polar Calculator rollout。
