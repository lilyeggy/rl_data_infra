# Day 1 收口：只冻结证据，不重跑昂贵 smoke

Day 1 的 SGLang 与 Megatron smoke 已在 2026-08-03 完成。当前只需确认服务器没有发生版本漂移，并把完整 commit、包版本和镜像 digest 固定下来。

## Step 1：采集服务器事实

在双 RTX PRO 6000 服务器上，将本仓库放到可访问位置，然后执行：

```bash
cd <本仓库目录>
bash scripts/capture_day01_evidence.sh /data/day-01-workspace
```

脚本只读检查环境，不会启动或停止服务，也不会修改上游源码和模型权重。输出写入：

```text
/data/day-01-workspace/artifacts/day-01/
├── host-preflight.txt
├── gpu-topology.txt
├── source-versions.txt
├── python-packages.txt
├── container-images.txt
└── paths.txt
```

## Step 2：人工审计

先检查 `source-versions.txt`：

- 六个上游仓库都必须有完整 40 位 commit；
- Polar 应对应 stable 基线，Slime 应对应 v0.3.0 基线；
- 任何 `dirty=true` 都必须说明本地补丁及其 checksum；
- 缺失仓库不能用文档中的缩写 commit 猜测补齐。

再检查：

- `python-packages.txt` 中 rollout/training 两套环境是否仍分别存在；
- `container-images.txt` 是否包含 Day 2 Calculator runtime 所需镜像及 digest；
- 模型与 tokenizer 的精确 revision 是否能从本地缓存或下载 manifest 反查；
- Day 1 原始 smoke JSON/log 是否仍存在且可计算 checksum。

## Step 3：生成版本锁

审计完成后再写 `configs/upstream-lock.yaml`。锁文件至少包含：

```yaml
captured_at_utc: <UTC timestamp>
hardware:
  gpu: NVIDIA RTX PRO 6000 Blackwell Server Edition
  count: 2
software:
  polar: {ref: stable, commit: <full commit>, dirty: false}
  slime: {ref: v0.3.0, commit: <full commit>, dirty: false}
  megatron_lm: {commit: <full commit>, patch_checksum: <sha256>}
  sglang: {ref: v0.5.13, commit: <full commit>, dirty: false}
  mbridge: {version: 0.15.1, commit: <full commit>}
model:
  id: Qwen/Qwen3-4B-Instruct-2507
  revision: <immutable revision>
runtime_images:
  calculator: <repository@sha256:digest>
```

只有完整事实齐全后，Day 1 才标记为 `COMPLETED`。若 GPU/驱动、关键 commit、模型 revision 或 runtime image 已变化，则只重跑受影响的 smoke，不做全量返工。

## Day 1 → Day 2 门槛

满足以下条件即可进入 Polar Calculator：

- 版本锁无 `unknown`/缩写 commit；
- 本地 patch 有 diff/checksum；
- SGLang 原始响应能证明 token ID 与 logprob 来自实际采样；
- Megatron smoke 能证明 optimizer step、checkpoint save/load 与 checksum 变化；
- Day 2 使用的 runtime image digest 已冻结。
