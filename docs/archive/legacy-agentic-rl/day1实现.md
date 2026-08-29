# Day 1：官方基线、硬件环境与数据契约 — 执行汇总
> **历史记录说明（2026-08-04）：** 本文件保留服务器端原始执行事实，不按新项目边界重写。新计划已将 Polar rollout 复现移到 Day 2–3，并取消“Day 1 必须跑官方 SWE-Gym + Slime 训练”的要求；4.1–4.4 的硬件、SGLang 和 Megatron smoke 结果继续复用。权威计划见 `PROJECT_PLAN.md` 与 `IMPLEMENTATION_PLAN.md`。
> 状态：**4.1–4.4 已完成，4.5–4.7 未开始**
> 环境：Ubuntu 22.04 / 双 NVIDIA RTX PRO 6000 Blackwell（sm_120，96GB×2）/ 驱动 580.159.03 / CUDA 13.0
> 记录日期：2026-08-03
## 一、阶段目标与定位
Day 1 不合并外部 Harness、不做自研 Agent loop / schema / quality gate，只解决一件事：
**在写任何自研逻辑前，证明"官方栈（Polar + SGLang + Slime + Megatron）在这台机器上能跑、版本能复现、坑在哪"。**
RL 系统两条腿——**推理侧（SGLang，出 token/logprob）** 和 **训练侧（Megatron，做梯度更新）**，4.1/4.2 是底座，4.3/4.4 分别验证两条腿。
## 二、硬件与系统环境（4.1）
| 项 | 事实 |
|---|---|
| GPU | 2 × RTX PRO 6000 Blackwell Server Edition，sm_120，97887MiB/卡，无 Xid，ECC 0 错误，persistence 开启 |
| 驱动 / CUDA | 580.159.03 / 13.0（nvcc 13.0.88） |
| 拓扑 | 两卡 NODE 互联、同 NUMA（非 NVLink） |
| CPU / 内存 | 192 核 / 251GB |
| 磁盘 | 系统盘 `/` 3.5T 剩 973G；数据盘 `/data` 3.5T 剩 3.1T ✅ 全量落数据盘 |
| Docker | 29.1.3 + nvidia-container-toolkit 1.19.1，data-root 迁至 `/data/day-01-workspace/docker-root`，容器内双卡可见 |
**网络环境（已绕开，全部记录）**：huggingface.co 直连不通 → `HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1`；github git 协议 TLS 不稳 → 克隆走 `gh-proxy.com`（remote 已改回官方地址）；docker.io 不通 → 镜像走 `nvcr.io`（NGC）。
## 三、版本冻结（4.2）
上游源码（`/data/day-01-workspace/src/`），机器可读 lock 见 `configs/upstream-lock.yaml`：
| 组件 | ref/commit | 依据 |
|---|---|---|
| Polar | stable @ `f0e8343a…` | NVIDIA-NeMo/ProRL-Agent-Server stable |
| Slime | v0.3.0 @ `bf14dc2…` | Polar stable slime_bridge 指定 |
| Megatron-LM | `1dcf0dafa…`（已应用 slime megatron.patch） | slime v0.3.0 docker/Dockerfile MEGATRON_COMMIT |
| SGLang | v0.5.13 @ `28b095c…` | Polar stable README |
| mbridge | 0.15.1 = ISEEKYAN/mbridge@`89eb1088` | slime v0.3.0 docker/Dockerfile |
两个独立 uv venv（python 3.12，唯一环境工具链，不混 conda/venv）：
- **rollout**：`src/polar/.venv` —— sglang 0.5.13 / torch 2.11.0+cu130 / flash-attn-4 4.0.0b19
- **training**：`src/slime/.venv` —— megatron-core（patched，非 TE）/ slime 0.3.0 / mbridge 0.15.1 / transformers 4.57.3 / numpy 1.26.4（sitecustomize 补 `np.long/np.ulong`）
## 四、SGLang 推理验证（4.3，GPU 1）
- 模型：`Qwen/Qwen3-4B-Instruct-2507`（dense，**按文档止损条款降级**），TP=1，context 8192，mem-fraction 0.70
- 启动 ~82s（权重 1.74s + cuda graph 捕获 39.6s）；首次请求 64.7s（JIT 预热），**预热后 53ms**
- 返回正确（`1+1=?` → `2`），finish_reason=stop，logprobs / token ID / usage 字段齐全
- 峰值显存 ~70.9GB；产物：`inference-response.json`、`sglang-server.log`
**关键坑（根因定位）**：Qwen3.5-4B（hybrid GatedDeltaNet）与 flashinfer 的一系列崩溃（`requires sm75+`、cutlass DSL `Expected an MLIR object`）——根因是 **flashinfer 通过 `nvcc --version` 探测 CUDA，而系统 PATH 里 `/usr/bin/nvcc` 是 CUDA 11.5**，误判 CUDA<12.9 拒绝 sm_120。**修复 = 启动时注入 `CUDA_HOME=/usr/local/cuda-13.0` 且 PATH 前置**，冻结栈零改动。Qwen3.5-4B 的 GDN 内核是否可用未重测，降级 dense 完成验证。
## 五、Megatron 训练验证（4.4）
### 5.1 权重转换（HF → Megatron torch_dist）
- Qwen3-4B-Instruct-2507（7.6G）→ torch_dist 7.5G，TP1 / TP2 各一份
- 坑：mbridge 桥映射是 TE 风格命名，local spec 是 `input_layernorm` / `pre_mlp_layernorm` → **自定义桥**（`scripts/local_qwen3_bridge.py`）补映射后转换通过
### 5.2 Smoke 测试（`scripts/smoke_megatron_step.py`）
每轮：载权重 → checksum(A) → 随机数据 forward → backward → **optimizer step**（megatron 0.16 新 API，`step()` 返回 grad_norm）→ checksum(B) → save ckpt → 篡改参数 → reload → checksum(C) 校验恢复。
| 配置 | loss | grad_norm | step | peak/卡 | 校验 |
|---|---|---|---|---|---|
| 512 / TP1 / Adam BF16 | 2.06 | 31.0 ✅ 非零 | 2.4s | 88.7GB | A≠B 且 B==C ✓ |
| 4096 / TP2 / Adam BF16 | 11.51 | 127.0 | 5.1s | 82.8GB | ✓ |
| **8192 / TP2 / Adam BF16** | **11.36** | **70.0** | **3.6s** | **45.3GB** | ✓ |
### 5.3 8K 打通过程（减配 → 无损优化）
1. 8K 最初 OOM（~89GB/卡）：local spec 的注意力是 torch SDPA math 后端，在 8K 物化 `(16, 8192, 8192)` 分数/softmax 矩阵
2. 两条"减配"均被 TE 依赖堵死：`--offload-optimizer-states`（需 FusedAdam）、`--optimizer sgd`（megatron 0.16 只支持 Adam/HybridDevice）
3. **最终方案（本地 patch，数学等价）**：把 `megatron/core/transformer/dot_product_attention.py` 的经典实现（baddbmm+softmax+bmm）替换为 **torch `scaled_dot_product_attention`（flash/mem-efficient，`is_causal`，不物化分数矩阵）**
   - 正确性：4096 loss 11.33 vs 原实现 11.51（同量级），grad_norm 116.9 vs 127
   - 效果：4096 峰值 82.8→44.3GB/卡；**8192 从 OOM → 45.3GB/卡稳定通过**
   - 已知差异：SDPA dropout 用全局 RNG（TP 下逐 rank 不对齐），正式训练复现性需评估（已记 ADR）
   - 改动仅一个文件，git 可回滚
### 5.4 训练栈关键决策（记入 ADR 0001 决策 3）
- 不装 TE → `--transformer-impl local` + 关 4 个融合项（rope-fusion / persist-layer-norm / gradient-accumulation-fusion / masked-softmax-fusion）
- 当前配置（dense 4B / BF16 / full recompute / Adam / TP2）：**最大稳定 context = 8192**（45GB/卡），单卡满 Adam 状态仅 ~512
## 六、踩坑清单（全部已定位并记录）
| 坑 | 根因 | 解法 |
|---|---|---|
| HF 模型下载 401 | xet 后端域名不通 | `HF_HUB_DISABLE_XET=1` + hf-mirror |
| flashinfer 拒绝 sm120 / cutlass DSL 崩溃 | 系统 PATH 的 `nvcc` 是 CUDA 11.5 | 启动注入 `CUDA_HOME=/usr/local/cuda-13.0` + PATH |
| Qwen3.5-4B hybrid 内核不稳 | GDN + flashinfer JIT 组合 | 按文档止损降级 dense Qwen3-4B-Instruct（写 ADR，不静默换） |
| 训练栈 transformers 版本冲突 | mbridge 要 4.x / radixark 要 5.x | 不装 radixark（slime 运行时不 import），transformers 4.57.3 |
| mbridge 缺权重映射 | local spec 命名 ≠ TE 命名 | 自定义 LocalQwen3Bridge |
| 8K OOM | SDPA math 后端物化分数矩阵 | 换 torch SDPA flash/mem-efficient（patch） |
| `nvidia-modelopt` 导入失败 | 用 `np.long`，numpy≥1.24 移除 | sitecustomize 补 `np.long/np.ulong` |
## 七、产物清单
```
/data/day-01-workspace/
├── day-01-environment.md            # 文档（含执行记录 8/8.1）
├── day-01-summary.md                # 本文档
├── configs/upstream-lock.yaml       # 机器可读版本 lock
├── docs/adr/0001-freeze-upstream-baseline.md   # 冻结基线 + 决策1/2/3
├── scripts/
│   ├── check_gpu_host.sh            # 4.1 预检脚本
│   ├── smoke_megatron_step.py       # 4.4 训练 smoke
│   ├── local_qwen3_bridge.py        # 4.4 转换用自定义 mbridge 桥
│   └── convert_qwen3.py             # 4.4 转换包装器
├── src/   (polar / slime / Megatron-LM / sglang / mbridge-iseekyan / mbridge)
├── models/ (qwen3-4b-instruct-2507-td-tp1 / -td-tp2，megatron 权重)
├── hf-cache/ (Qwen3.5-4B, Qwen3-4B-Instruct-2507)
└── artifacts/day-01/
    ├── host-preflight.txt / gpu-topology.txt / docker-gpu-test.txt   # 4.1
    ├── source-versions.txt / python-packages.txt / container-images.txt / paths.txt  # 4.2
    ├── inference-response.json / sglang-server.log                    # 4.3
    ├── backward-smoke-512.json / backward-smoke-4096-tp2.json / backward-smoke-8192-tp2.json  # 4.4
    ├── convert.log / convert-tp1-fresh.log / smoke-*.log              # 4.4 日志
    └── execution-record.md            # 全程执行记录
```
## 八、未开始（4.5–4.7）与下一步
- **4.5 Polar Calculator 官方 smoke**：1 rollout server / 1 gateway / 1 SGLang backend / 1 harness / 1 task，审计模型请求是否经 Gateway、tool/evaluator 是否执行、token/logprob 是否来自实际采样
- **4.6 官方 SWE-Gym + Slime 最小基线**：1 任务 × 最小 rollout group，观察完整字段与调用顺序（SWE-Gym/evaluator 尚未冻结）
- **4.7 数据契约 ADR**：TaskEnvelope / CanonicalTrajectoryBundle 等 11 个类型，external ID 作 opaque metadata
## 九、验收对照（文档 §6 验收门）
- [x] 两张 PRO 6000、Docker、数据盘事实已记录
- [x] SGLang 返回 tokens/logprobs 并保存原始响应
- [x] Megatron 完成真实 optimizer step 和 checkpoint reload（4096/8192 TP2 全绿）
- [ ] Polar Calculator 完整结束并保存 raw artifacts
- [ ] 官方 SWE-Gym/Slime 链路至少分段验证
- [x] 所有源码 commit、镜像 digest、模型 revision 冻结
- [ ] 数据契约 ADR 基于实际字段完成
- [x] 没有引入外部 Harness 源码（mbridge 桥 patch 为本地最小适配，非外部 Harness）
> 当前阶段状态：**PASS_WITH_NOTES**（4.1–4.4 达标，8K 经本地等价 patch 打通；4.5–4.7 待办）
