# ADR 0001: 冻结上游基线（freeze-upstream-baseline）

状态：Accepted（2026-08-03，Day-01 阶段）
范围：Day-01 基线阶段 4.1/4.2 已冻结内容；4.3 发现的模型内核问题及降级决策

## 背景

在双 RTX PRO 6000 Blackwell（sm_120，驱动 580.159.03 / CUDA 13.0）上建立可复现基线，
优先复现 Polar stable 示例，禁止随意组合 main/nightly。

## 决策 1：上游源码冻结（4.2 完成）

| 组件 | 来源 | ref/commit | 依据 |
|---|---|---|---|
| Polar | NVIDIA-NeMo/ProRL-Agent-Server | stable @ f0e8343a7870abf6ec2366890f685881ceab92cb | 文档要求 stable 分支 |
| Slime | THUDM/slime | v0.3.0 @ bf14dc21f9500746447f2572d0692e981c4d2a7e | Polar stable src/slime_bridge/README.md |
| Megatron-LM | NVIDIA/Megatron-LM | 1dcf0dafa884ad52ffb243625717a3471643e087 | slime v0.3.0 docker/Dockerfile MEGATRON_COMMIT |
| SGLang | sgl-project/sglang | v0.5.13 @ 28b095c01005d4a3a2a5b637b7d028b07fba31b2 | Polar stable README：uv pip install "sglang==0.5.13" |

配套：uv venv（python 3.12，唯一环境工具链），torch 2.11.0+cu130、flash-attn-4 4.0.0b19、
flashinfer-python 0.6.12。机器可读 lock 见 configs/upstream-lock.yaml。

## 决策 2：Qwen3.5-4B 在 sm_120 上内核不稳定 → 降级 dense 模型（4.3 发现）

现象（SGLang 0.5.13 + flashinfer 0.6.12 + torch 2.11.0+cu130，GPU sm_120）：

1. flashinfer attention/cuda-graph：`FlashInfer requires GPUs with sm75 or higher`——
   flashinfer 认为本机 CUDA < 12.9（其 JIT 检查 `is_cuda_version_at_least("12.9")` 为假，
   尽管 torch 为 cu130；flashinfer 0.6.12 构建的 CUDA 版本认知低于 12.9）。
2. 换 triton attention + 禁用 cuda graph 后，Qwen3.5-4B（hybrid GatedDeltaNet/mamba）在
   warmup forward 崩溃：nvidia_cutlass_dsl/cutlass MLIR `Expected an MLIR object`（SIGABRT），
   触发点为 flashinfer rmsnorm DSL JIT。

### 根因（2026-08-03 定位）
flashinfer 的 `get_cuda_version()` 通过 `nvcc --version` 探测 CUDA 版本，而系统默认 PATH 中
`/usr/bin/nvcc` 是 CUDA 11.5 → 误判 CUDA < 12.9 → 拒绝 sm120，进而引发系列 JIT/DSL 崩溃。
**修复**：启动服务时注入 `CUDA_HOME=/usr/local/cuda-13.0` 且 `PATH` 前置 `/usr/local/cuda-13.0/bin`。
冻结栈无需改动（torch 2.11.0+cu130 / flashinfer 0.6.12 / sgl-kernel 0.4.3 / nvidia-cutlass-dsl 4.5.2）。

### 决策
按 4.3 止损条款降级为成熟 dense 模型完成 SGLang 验证：
`Qwen/Qwen3-4B-Instruct-2507`（dense、官方 chat 模型、snapshot cdbee75f17c01a7cc42f958dc650907174af0554）。
验证结果：GPU1/TP1/8K，预热后 53ms 返回，logprobs/token ID 齐全，详见 artifacts/day-01/inference-response.json。
Qwen3.5-4B 的 GDN 内核在 CUDA13 环境注入后是否可用尚未重测，留待后续（届时可优先复测，
若仍不稳定则维持 dense 基线）。

## 未决

- mbridge（ISEEKYAN/mbridge@89eb1088，已冻结）与 SWE-Gym/evaluator 尚未冻结（4.2 剩余：仅 SWE-Gym/evaluator）。
- Megatron 训练侧在 slime venv 内使用 --transformer-impl local（TE 未装）；若要 TE 需构建 transformer-engine-torch wheel（GitHub Releases 直连不通，需代理，见 4.4 记录）。
- SGLang 在 sm_120 上的最终可用 attention 后端组合需以 dense 模型验证结果为准。

## 决策 3：训练侧（4.4）内存与容量结论（2026-08-03 更新）

- 训练环境为独立 uv venv（slime），与 rollout venv（polar+sglang）隔离，符合"只用 uv、不混环境"。
- 4B dense / BF16 / full recompute / Adam / 双卡 TP2：最大稳定 context = 4096（完整 step+ckpt reload 校验通过）；8K 最初 OOM。
- **8K 打通方案（本地 patch，数学等价）**：把 megatron DotProductAttention 的
  baddbmm+scale_mask_softmax+bmm 经典实现替换为 torch `scaled_dot_product_attention`
  （flash/mem-efficient，is_causal，不物化 [b,np,s,s] 分数矩阵）。改动仅
  `megatron/core/transformer/dot_product_attention.py`（本地，git 可回滚），
  不改变 loss 语义（4096 验证 loss 11.33 vs 原实现 11.51，同量级）。
  已知差异：SDPA dropout 使用全局 RNG（TP 下 dropout mask 不再逐 rank 对齐），正式训练复现性需评估。
- 结果：8K / TP2：loss=11.36，grad_norm=70.0，step 3.64s，**peak 45.3GB/卡**（4096 亦从 82.8GB 降至 44.3GB），
  checksum 变化+恢复 ✓（backward-smoke-8192-tp2.json）。
- 未用方案（记录）：`--optimizer-cpu-offload`/`--offload-optimizer-states` 均依赖 TE（FusedAdam），TE 缺席不可用；
  不混装 nightly；LoRA 为最后方案未使用。
