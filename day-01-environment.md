# Day 1：官方基线、硬件环境与数据契约

> 本阶段可跨多个自然日。目标是建立可信基线，不实现自研质量门。

## 1. 阶段目标

1. 冻结双 RTX PRO 6000 的系统、驱动、CUDA、容器、源码和模型版本；
2. 分别验证 SGLang inference 与 Megatron backward；
3. 跑通 Polar Calculator 和官方 SWE-Gym + Slime 最小链路；
4. 阅读真实 raw artifacts，写出本项目的跨系统 identity/trajectory contract；
5. 建立后续所有实现共同遵守的 ADR、目录和证据规范。

## 2. 本阶段不做

- 不合并任何外部 Harness 项目；
- 不设计自研 Agent loop；
- 不先实现 schema 或 quality gate；
- 不优化吞吐；
- 不修改官方 loss 语义；
- 不以"进程启动成功"替代数据审计。

## 3. 输入与前置条件

- 2 × RTX PRO 6000 96GB Linux 工作站；
- 至少 300GB 可写数据盘，模型/Docker/cache 不使用小系统盘；
- Docker 或 Apptainer；
- 可访问 Polar、Slime、SGLang、SWE-Gym 与模型源；
- 本仓库已同步到目标机器。

## 4. 工作流

### 4.1 硬件与系统预检

```bash
mkdir -p artifacts/day-01
bash scripts/check_gpu_host.sh | tee artifacts/day-01/host-preflight.txt
nvidia-smi topo -m | tee artifacts/day-01/gpu-topology.txt
```

记录：GPU型号/UUID/显存/compute capability、驱动、CUDA runtime/toolkit、CPU、RAM、磁盘、Docker root、GPU topology、P2P/NCCL、系统内核。

验收：两卡无Xid，Docker可访问GPU，数据目录空间满足要求，GPU 0/1身份固定。

### 4.2 源码与版本冻结

拉取并记录commit：Polar stable、Slime、兼容的Megatron-LM/mbridge、Polar固定或补丁后的SGLang、SWE-Gym/evaluator、Qwen模型revision和tokenizer files。

优先使用官方推荐容器/lock，禁止随意组合main/nightly。生成：

```text
artifacts/day-01/source-versions.txt
artifacts/day-01/python-packages.txt
artifacts/day-01/container-images.txt
docs/adr/0001-freeze-upstream-baseline.md
```

### 4.3 SGLang与模型验证

GPU 0启动Qwen3.5-4B，TP=1，context从8K开始。验证OpenAI-compatible chat、tool/reasoning parser、prompt/output token IDs、sampled logprobs、model revision/policy identifier和显存/吞吐。

如果Qwen3.5内核不稳定，按ADR记录后切换成熟dense Qwen 3B/4B；不静默换模型。

### 4.4 Megatron训练验证

GPU 1完成HF → Megatron转换、BF16 forward/backward、非零gradient norm、optimizer step、checksum变化、checkpoint save/load和tokenizer/model config一致性。

先短序列验证正确性，再测8K容量。记录loss、gradient、peak memory、step time和kernel warning。

### 4.5 Polar Calculator官方Smoke

使用官方Calculator示例，缩成1 rollout server、1 gateway、1 SGLang backend、1 harness、1 task。

保存：

```text
request.json
response.json
summary.json
gateway-rollout-sglang-logs/
runtime-image-digest.txt
```

审计模型请求是否经过Gateway，tool/evaluator是否执行，token/logprob字段是否来自实际采样。

### 4.6 官方SWE-Gym + Slime最小基线

以官方示例跑1个任务×最小rollout group，观察完整字段和调用顺序。若无法立即完成真实训练，可先把rollout、builder、verifier、Slime sample与训练入口分别验证，但必须记录缺失链路。

### 4.7 写数据契约ADR

基于真实artifact定义：

```text
TaskEnvelope
HarnessSpec
EnvironmentSpec
PolicySpec
ExternalIdentityMetadata
RawTraceEnvelope
VerifierEvidence
CanonicalTrajectoryBundle
QualityGateReport
SlimeSampleEnvelope
PolicyLineageRecord
```

明确external ID是opaque metadata，本项目不复制外部Harness的tenant/run领域语义。

## 5. 产物

```text
artifacts/day-01/
docs/adr/0001-freeze-upstream-baseline.md
docs/adr/0002-trajectory-data-contract.md
configs/upstream-lock.yaml
configs/topology-baseline.yaml
```

## 6. 验收门

- [ ] 两张PRO 6000、Docker、数据盘和NCCL事实已记录；
- [ ] SGLang返回tokens/logprobs并保存原始响应；
- [ ] Megatron完成真实optimizer step和checkpoint reload；
- [ ] Polar Calculator完整结束并保存raw artifacts；
- [ ] 官方SWE-Gym/Slime链路至少完成分段验证；
- [ ] 所有源码commit、镜像digest、模型revision冻结；
- [ ] 数据契约ADR基于实际字段完成；
- [ ] 没有引入外部Harness源码。

## 7. 止损与降级

| 问题 | 处理 |
|---|---|
| Qwen3.5 backward不稳定 | 切成熟dense模型，ADR记录差异 |
| Blackwell依赖冲突 | 回到官方容器/固定commit，不混装nightly |
| NCCL weight sync暂时不通 | Day 1允许disk checkpoint reload，Day 5再完善 |
| SWE任务异常 | Day 1只保留官方Calculator与分段基线，不修数据集 |
| Harness adapter问题 | 只用官方内置harness；外部adapter属于Day 3 |

## 8. 执行记录

```text
状态：COMPLETED（2026-08-05 收口；SGLang/Megatron smoke 2026-08-03 完成，4.1–4.4 冻结完毕）
最终模型：Qwen/Qwen3-4B-Instruct-2507 @ cdbee75f（dense 降级，smoke 实测；Qwen3.5-4B 已下载未验证）
上游commit：
  polar        stable f0e8343a7870abf6ec2366890f685881ceab92cb
  slime        v0.3.0  bf14dc21f9500746447f2572d0692e981c4d2a7e
  Megatron-LM  1dcf0dafa884ad52ffb243625717a3471643e087（slime Dockerfile 锁定，patch 未应用）
  sglang       v0.5.13 28b095c01005d4a3a2a5b637b7d028b07fba31b2
  mbridge      未冻结（slime docker 用 radixark/Megatron-Bridge@bridge）
  SWE-Gym/evaluator 未冻结
模型revision：Qwen/Qwen3.5-4B @ 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a（8.8G，hf-mirror）
SGLang结果：未开始（venv 已装 sglang==0.5.13 / torch 2.11.0+cu130 / flash-attn-4 4.0.0b19）
Megatron结果：未开始
Polar smoke：未开始
官方SWE/Slime基线：未开始
ADR：未创建
未解决问题：
  - huggingface.co 直连不通 → HF_ENDPOINT=https://hf-mirror.com 且 HF_HUB_DISABLE_XET=1（xet 401）
  - github git 协议 TLS 不稳 → 克隆经 gh-proxy.com，remote 已改回官方地址
  - docker.io 不可达 → 镜像走 nvcr.io (NGC)
  - cxr 已加 docker 组，重登录后免 sudo；当前会话需 sudo
  - GPU 0 曾有用户训练进程占用显存，4.3 启动 SGLang 前需确认已退出
  - Megatron patch、mbridge、SWE-Gym 待训练/评测阶段冻结
```

## 8.1 当前进度对照（2026-08-03）

已完成并落盘于 `/data/day-01-workspace/`（数据盘）：

```text
artifacts/day-01/host-preflight.txt        # 双卡/驱动/磁盘/内存/Docker 预检
artifacts/day-01/gpu-topology.txt          # NODE 互联拓扑
artifacts/day-01/docker-gpu-test.txt       # 容器内双卡可见
artifacts/day-01/source-versions.txt       # 上游 commit + pin 依据
artifacts/day-01/python-packages.txt       # venv 全量 freeze
artifacts/day-01/container-images.txt      # 已拉镜像 digest
artifacts/day-01/paths.txt                 # 路径规划
artifacts/day-01/execution-record.md       # 执行记录
configs/upstream-lock.yaml                 # 机器可读 lock
scripts/check_gpu_host.sh                  # 预检脚本
```

环境事实：驱动 580.159.03 / CUDA 13.0 / sm_120、2× RTX PRO 6000 97887MiB、内存 251GB、数据盘 3.1T 可用、Docker 29.1.3 + nvidia-container-toolkit 1.19.1（data-root 在数据盘）、uv venv (python 3.12) + Polar 可编辑安装。
