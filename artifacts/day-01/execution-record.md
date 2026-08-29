# Day-01 执行记录（仅 14:00 前的准备部分）

状态：PREP_DONE（14:00 前内容全部完成，等待 14:00 推理/训练阶段）
开始时间：2026-08-02T17:23+08:00
阶段完成时间：2026-08-02T18:3x+08:00

## 已完成
- [x] 宿主机预检：2× RTX PRO 6000 Blackwell（sm_120，97887MiB/卡），驱动 580.159.03 / CUDA 13.0，无 Xid，ECC 0 错误 → artifacts/day-01/host-preflight.txt
- [x] GPU 拓扑 → artifacts/day-01/gpu-topology.txt（两卡 NODE 互联，同 NUMA）
- [x] Docker 29.1.3 + nvidia-container-toolkit 1.19.1 安装，data-root 迁至 /data，容器内可见双卡 → artifacts/day-01/docker-gpu-test.txt
- [x] 路径规划落盘（全部在 /data）→ artifacts/day-01/paths.txt
- [x] 源码冻结：polar stable f0e8343 / slime v0.3.0 bf14dc2 / Megatron-LM 1dcf0da / sglang v0.5.13 28b095c → artifacts/day-01/source-versions.txt
- [x] uv venv（python 3.12）+ Polar 可编辑安装
- [x] Qwen/Qwen3.5-4B 下载完成（8.8G，snapshot 851bf6e8，hf-mirror，禁用 xet）→ /data/day-01-workspace/hf-cache
- [x] venv 内安装 sglang==0.5.13（含 torch 2.11.0+cu130、flash-attn-4 4.0.0b19），torch 可见双卡、cap=(12,0) → artifacts/day-01/python-packages.txt

## 未做（14:00 之后的内容，按要求暂停）
- SGLang 单卡推理服务启动与验证（inference-response.json / sglang-server.log）
- GPU 1 backward smoke test（backward-smoke.json / memory-before/after.txt）
- decision.md 冻结决策

## 已知问题与备注
1. huggingface.co 直连不通：全程用 HF_ENDPOINT=https://hf-mirror.com，且必须 HF_HUB_DISABLE_XET=1（xet 后端 401）。
2. github.com git 协议 TLS 不稳：克隆走 gh-proxy.com 前缀，remote 已统一改回 github.com 原始地址。
3. docker.io registry 不通：镜像从 nvcr.io（NGC）拉取；后续如需 Docker Hub 镜像需配 registry mirror。
4. cxr 已加入 docker 组，但当前会话未刷新组成员，docker 命令暂需 sudo（重新登录后免 sudo）。
5. GPU 0 上有用户自己的训练进程（train_c3_v2_ceiling.py）曾占用 ~2.2GB；Day-1 后续启动 SGLang 前需确认其已退出。
6. 系统盘 / 实际有 973G 空闲（非文档假设的 30GB），但所有缓存/源码/Docker 仍按要求落在 /data。
7. Megatron-LM 未应用 slime docker/patch/v0.5.12.post1/megatron.patch（属于训练环境安装步骤，留到后续）。
8. 安装 sglang==0.5.13 需 `uv pip install --prerelease=allow`（依赖 flash-attn-4 预发布版）。
EOF

## 4.3 SGLang 验证结果（2026-08-03 补充）

- 服务：SGLang 0.5.13 @ GPU 1（端口 30000），Qwen3-4B-Instruct-2507（dense 降级），TP=1，context 8192，mem-fraction 0.70
- 启动：19:44:52 启动 → 19:45:12 加载权重（1.74s）→ cuda graph 捕获 39.6s → 19:46:14 Uvicorn ready（约 82s）
- 显存：GPU1 峰值 ~70.9GB（KV cache 58GB + 权重 7.7GB + graph）
- 首次请求 64.7s（JIT/预热），预热后 53ms；响应 `2`，finish_reason=stop，带 logprobs（token "2"、<|im_end|>）
- 响应保存：artifacts/day-01/inference-response.json
- 服务日志：artifacts/day-01/sglang-server.log

### 根因与修复（重要）
Qwen3.5-4B / flashinfer 在 sm_120 上一系列崩溃（`FlashInfer requires sm75+`、cutlass DSL `Expected an MLIR object`）
根因是 flashinfer 的 get_cuda_version() 通过 `nvcc --version` 探测 CUDA 版本，而系统默认 PATH 里
/usr/bin/nvcc 是 CUDA 11.5 → 误判 CUDA<12.9 → 拒绝 sm120。
修复：启动时注入 CUDA_HOME=/usr/local/cuda-13.0 且 PATH 前置 cuda-13.0/bin。冻结栈无需改动
（torch 2.11.0+cu130 / flashinfer 0.6.12 / sgl-kernel 0.4.3 / cutlass-dsl 4.5.2 全部恢复）。
Qwen3.5-4B 的 GDN 内核在注入后是否可用尚未重测（当前用 dense 完成验证），留待后续。

## 4.4 Megatron backward smoke 结果（2026-08-03 补充）

训练环境（/data/day-01-workspace/src/slime/.venv，uv/py3.12）：
- torch 2.11.0+cu130 / triton 3.6.0 / numpy 1.26.4（sitecustomize 补 np.long/np.ulong）
- Megatron-LM 1dcf0da（已应用 slime megatron.patch）+ slime v0.3.0（editable）
- mbridge 0.15.1 = ISEEKYAN/mbridge@89eb1088（editable）；radixark Megatron-Bridge 不需要（slime 运行时不 import）
- transformers 4.57.3；TE 未安装 → --transformer-impl local + --no-rope-fusion --no-persist-layer-norm --no-gradient-accumulation-fusion --no-masked-softmax-fusion

HF→Megatron 转换：Qwen3-4B-Instruct-2507 → torch_dist（TP1 版 + TP2 版，后者训练时用 --no-load-optim 从 TP1 载权重）。需要自定义 mbridge 桥映射 local-spec 的 input_layernorm/pre_mlp_layernorm（scripts/local_qwen3_bridge.py）。

Smoke 结果（scripts/smoke_megatron_step.py，真实 forward/backward/optimizer step + save/mutate/reload 校验）：
- 512 / TP1 / Adam BF16：loss=2.064，grad_norm=31.05，lr=1e-4，step 2.41s，peak 88.67GB，checksum 变化+恢复 ✓（backward-smoke-512.json）
- 4096 / TP2 / Adam BF16：loss=11.51，grad_norm=127.0，step 5.10s，peak 82.83GB/卡，checksum 变化+恢复 ✓（backward-smoke-4096-tp2.json）
- 8192 / TP2 / Adam BF16：OOM（~89GB/卡，差 ~5GB）。根因：local spec 的 torch SDPA math 后端在 8K 物化 (16,8192,8192) 分数/softmax 矩阵；flash-attn 已装但不改变 DotProductAttention 路径。8K 需 optimizer 状态 offload（--optimizer-cpu-offload 要求 TE）或换 attention backend —— 留待 Day 2。

结论：当前配置（dense 4B / BF16 / full recompute / Adam）单卡最大稳定 ~512-1K，双卡 TP2 最大稳定 4096；8K 为容量缺口（记录在案）。

### 4.4 更新：8K 打通（2026-08-03，SDPA 补丁）
把 megatron DotProductAttention 经典实现（baddbmm+softmax+bmm，物化 [b,np,s,s]）替换为
torch scaled_dot_product_attention（is_causal，flash/mem-efficient，不物化分数矩阵）。
数学等价（4096 loss 11.33 vs 原 11.51；grad_norm 116.9 vs 127）。
效果：4096 峰值 82.8→44.3GB/卡；**8192 从 OOM 变为 45.3GB/卡稳定通过**（step 3.64s，
checksum 变化+恢复 ✓）。产物 backward-smoke-8192-tp2.json。
注意：SDPA dropout 用全局 RNG（TP 下逐 rank 不复现），正式训练需评估。
--offload-optimizer-states / --optimizer-cpu-offload 均需 TE，未采用。

## Day 1 收口（2026-08-05，对应 day-01-closeout.md）

状态：**COMPLETED**（SGLang 与 Megatron smoke 为 2026-08-03 完成；本次只冻结证据、重跑受影响的镜像拉取/构建，未重跑昂贵 smoke）

### Step 1：采集
- 新增只读采集脚本 `scripts/capture_day01_evidence.sh <workspace>`（只读检查，不启停服务、不改源码与权重）。
- 执行后 6 个证据文件已刷新（artifacts/day-01/），原始 08-02/08-03 版本归档于 `artifacts/day-01/baseline-2026-08-03/`。
- 本次为冻结 runtime image 新增环境事实：docker.io 仍不可达，经镜像代理 `docker.1panel.live` 拉取 `python:3.12-slim-bookworm` 与 `node:22-bookworm-slim`（已有 digest 记录），tag 为本地名供 Dockerfile FROM 解析。

### Step 2：人工审计结论
无漂移（关键事实全部一致）：
- GPU/驱动/CUDA：2× RTX PRO 6000（UUID 不变）、580.159.03 / CUDA 13.0 / sm_120，无变化。
- 六个上游仓库 commit 全部完整 40 位、无缩写：
  - polar        stable   f0e8343a7870abf6ec2366890f685881ceab92cb（clean）
  - slime        v0.3.0   bf14dc21f9500746447f2572d0692e981c4d2a7e（clean）
  - Megatron-LM           1dcf0dafa884ad52ffb243625717a3471643e087（**dirty**，见下）
  - sglang       v0.5.13  28b095c01005d4a3a2a5b637b7d028b07fba31b2（clean）
  - mbridge(ISEEKYAN) v0.15.1 89eb10887887bc74853f89a4de258c0702932a1c（clean）
  - megatron_bridge(radixark) bridge 7f0fb3456f8ffe47599b5fd167b454605d85f932（clean；slime 运行时未 import，仅记录）
- Megatron-LM dirty 说明（两笔本地补丁，均有 checksum）：
  1. slime megatron.patch（v0.5.12.post1）：文件 sha256 `976fd8bb…c5b77`，以 staged 应用，diff sha256 `c30f39e6…818c1`；
  2. 本地 SDPA attention 补丁（dot_product_attention.py，8K 容量打通）：unstaged，diff sha256 `eb95f109…30cb8`；
  全量 `git diff HEAD` checksum `2a2af7be…ddd64` 已入锁。
- 模型 revision 可反查：hf-cache `main` ref → `cdbee75f17c01a7cc42f958dc650907174af0554`（Qwen3-4B-Instruct-2507，snapshot 含 config/tokenizer）；Qwen3.5-4B `851bf6e8…` 已下载未验证，如实标注。
- 两套 venv 仍分别存在：polar/.venv（sglang 0.5.13 / torch 2.11.0+cu130 / flash-attn-4 4.0.0b19 / polar 0.1.0 editable）、slime/.venv（torch 2.11.0+cu130 / megatron-core 0.16.0rc0 editable / slime 0.3.0 editable / mbridge 0.15.1 editable / transformers 4.57.3 / numpy 1.26.4）。
- Day 1 原始 smoke 证据仍存在且可 checksum（见 lock `evidence.checksums`）：
  - inference-response.json `270b3078…25c7`：token ID + logprob 来自实际采样（token "2"、<|im_end|>，logprob 0.0，matched_stop 151645，HTTP 200；model 字段引用 cdbee75f… snapshot）
  - sglang-server.log `3379b0bb…1dd96`（19:45:02 启动，19:47:36/43 两次 POST /v1/chat/completions 200 OK）
  - backward-smoke-512.json / 4096-tp2 / 8192-tp2：`checksum_changed_after_step=true`、`checksum_restored_after_reload=true`，证明 optimizer step + checkpoint save/mutate/reload。

发现并记录（不阻塞门槛）：
- polar venv 存在传递依赖小幅变动（08-02 安装态 vs 当前）：numpy 2.3.5→2.5.1、cuda-bindings/cuda-python 13.3.1→13.0.3、nvidia-cuda-nvcc 13.2.86→13.2.51、fsspec 2026.6.0→2026.7.0、apache-tvm-ffi 0.1.9→0.1.13.post0，新增 nccl4py/cutlass-dsl-libs-core/cu12。关键栈（torch/sglang/sgl-kernel/flash-attn-4/flashinfer/xgrammar/polar）逐项一致；08-03 smoke 运行态与当前相同，锁以当前 freeze 为准。
- SWE-Gym/evaluator 仍 NOT_FROZEN（4.6 分段基线前冻结，如实保留在锁中）。

### Step 3：版本锁
- 重写 `configs/upstream-lock.yaml`（closeout schema：captured_at_utc / hardware / software / model / runtime_images / environment / evidence）。
- Calculator runtime 镜像 digest 已冻结：`polar-localhost-calculator:latest@sha256:0a967a5c33ee206f7524181cdd4ff866389016651645336eee93de1a7d970f49`（本地构建，layout v3，base：python `d97928a6…` + node `d649c27d…`；本地镜像无 RepoDigest，以 image ID 作为内容 digest）。

### Day 1 → Day 2 门槛检查
- [x] 版本锁无 unknown/缩写 commit（六仓库全 40 位，YAML 校验通过）
- [x] 本地 patch 有 diff/checksum（Megatron 两笔补丁 + 全量 diff checksum 入锁）
- [x] SGLang 原始响应证明 token ID 与 logprob 来自实际采样（inference-response.json）
- [x] Megatron smoke 证明 optimizer step、checkpoint save/load 与 checksum 变化（backward-smoke-*.json）
- [x] Day 2 使用的 runtime image digest 已冻结（polar-localhost-calculator@sha256:0a967a…）

## Day 1 最小收口（2026-08-09，分支 codex/sync-authoritative-plans @ 08f0c048）

状态：**DAY1_STATUS=COMPLETED_WITH_NOTES**（验收器输出 PASS_WITH_NOTES，exit 0）

### 收口动作
1. 轻量环境采集 `bash scripts/capture_day01_evidence.sh /data/day-01-workspace`（只读，未启动/停止任何服务）。
2. 版本漂移审计（对照 configs/upstream-lock.yaml）——**无漂移**，未重跑任何 GPU smoke：
   - 六个上游仓库 commit 全 40 位且与 lock 一致：polar f0e8343a…、slime bf14dc21…、Megatron-LM 1dcf0daf…、sglang 28b095c0…、mbridge(ISEEKYAN) 89eb1088…、megatron_bridge(radixark) 7f0fb345…；
   - Megatron dirty 全量 `git diff HEAD` checksum 仍为 `2a2af7be…ddd64`（与 lock 一致）；
   - 模型 snapshot revision `cdbee75f…`（Qwen3-4B-Instruct-2507）不变；
   - 两套 venv 关键包不变（polar: sglang 0.5.13/torch 2.11.0+cu130；slime: mbridge 0.15.1/slime 0.3.0/megatron-core 0.16.0rc0）；
   - Calculator runtime 镜像身份一致：`polar-localhost-calculator:latest` ID `sha256:0a967a5c…`，layout v3。
3. 保存实际 Megatron SDPA patch：`patches/megatron/sdpa-sm120-8k.patch`（113 行）。
   - 干净 HEAD 副本 `git apply --check`：OK（exit 0），可从 HEAD 干净重现；
   - 当前工作树（补丁已应用）反向 `git apply --check --reverse`：OK（exit 0）。
   - SHA256：`eb95f109058bf8e57a9dadac222ac4908b3ebceaf6c26fdc05a1eef0a1a30cb8`。
4. 将 patch 相对路径与 SHA256 写入 `configs/upstream-lock.yaml`（megatron_lm.patches[1].path / .sha256），并更新 captured_at_utc 为 2026-08-09T08:06:57Z。
5. 最终验收 `python3 scripts/verify_day01_evidence.py`（不带 --strict-warnings）：12 PASS / 1 WARN，DAY1_STATUS=PASS_WITH_NOTES，exit 0。

### 失败实验标记（不得计入通过结果）
- `backward-smoke-512-tp2.json`：`checksum_restored_after_reload=false`（reload 后 checksum 回到 A 而非 B）→ **失败实验**，不满足 A≠B 且 B==C，不计入 Megatron 通过证据。
- 有效 Megatron TP2 证据以 4096/8192 为准：backward-smoke-4096-tp2.json 与 backward-smoke-8192-tp2.json 均满足 `A != B`、`B == C`、grad_norm>0（4096: 127.0；8192: 70.0）。

### 保留的 warnings（显式不消除）
1. `sglang_numeric_output_token_ids`：SGLang 原始响应（inference-response.json）含 token 文本与 logprob，但缺少原生数值 output token IDs。**不通过重新 tokenize 文本伪造**；由 Day 2 起在 gateway 层以原生字段采集。
2. `backward-smoke-512-tp2.json` 为失败实验（见上），保留在 artifacts 中作为失败证据，不在 lock 的通过 checksum 列表之外新增标记。
