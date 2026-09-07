# Day 1：环境、版本与双卡容量基线

> 本阶段只建立可复用的硬件/软件事实，不要求跑 Polar 或官方完整 Agentic RL。已有执行结果见 `day1实现.md`；版本未变化时不重复无意义检查。

## 1. 阶段目标

1. 冻结双 RTX PRO 6000、驱动、CUDA、容器、磁盘和源码版本；
2. 独立验证 SGLang inference；
3. 独立验证 Qwen3-4B 的两卡 Megatron forward/backward/checkpoint；
4. 确认后续采用 staged shared GPU，而不是固定“一卡 rollout、一卡 training”；
5. 把 Polar rollout 环境和 Trainer 环境视为可独立启动的两个参考栈。

## 2. 已有结果的处理

`day1实现.md` 中已经完成的硬件、CUDA、SGLang、模型转换和 Megatron smoke 继续有效。新计划不要求重做旧文档 14 点以前的检查，除非发生以下变化：

- GPU/驱动/CUDA/PyTorch 变化；
- Polar、SGLang、Slime、Megatron commit 变化；
- 模型或 tokenizer revision 变化；
- runtime/container image 变化；
- 现有结果无法提供命令、日志或版本证据。

## 3. 固定基线

```text
GPU: 2 × RTX PRO 6000 96GB
Model: Qwen/Qwen3-4B-Instruct-2507
Inference: SGLang
Reference Trainer: Slime + Megatron-LM
Training precision: BF16
Training parallelism: TP2
Execution: rollout/train staged sharing
```

Qwen3.5-4B 的历史兼容性问题保留在执行记录中，但不再作为第一版必须解决的问题。

## 4. 工作项

### 4.1 主机与 GPU 事实

记录 GPU UUID、拓扑、NUMA、驱动、CUDA runtime/toolkit、CPU、RAM、磁盘、Docker/Apptainer 与 P2P/NCCL。

### 4.2 版本锁

冻结：

```text
Polar commit
SGLang version
Slime commit/version
Megatron-LM/mbridge commit
PyTorch/CUDA dependencies
model/tokenizer revision
runtime image digest
```

### 4.3 SGLang smoke

验证 OpenAI-compatible API、token IDs、logprobs、tool parser、finish reason、context 4K/8K 和显存。`mem-fraction-static` 从约 0.35 开始测量，不把静态 reservation 当成单请求真实显存。

### 4.4 Megatron smoke

两张 GPU TP2 BF16 验证：forward/backward、非零 gradient、optimizer step、checkpoint save/load、模型 checksum 变化。短序列先验证正确性，再测试 4K/8K 容量。

### 4.5 Staged GPU 运行约定

```text
rollout phase: SGLang 使用一张或两张 GPU
process phase: CPU 数据处理，释放 rollout 占用
train phase: Megatron TP2 使用两张 GPU
reload phase: 恢复 SGLang 并加载新 checkpoint
```

## 5. 本阶段不做

- 不跑 Polar Calculator；它属于 Day 2；
- 不跑 SWE-Gym；它属于 Day 3；
- 不复现官方 8-GPU Slime 训练；
- 不实现核心 contract 或 Processor；
- 不优化最终吞吐；
- 不切换到 FP8 training。

## 6. 产物

```text
artifacts/day-01/host-preflight.txt
artifacts/day-01/gpu-topology.txt
artifacts/day-01/source-versions.txt
artifacts/day-01/python-packages.txt
artifacts/day-01/model-smoke/
configs/upstream-lock.yaml
```

## 7. 验收门

- [ ] 双卡硬件、驱动、CUDA、磁盘事实可追溯；
- [ ] SGLang 返回实际 token/logprob；
- [ ] Qwen3-4B TP2 完成真实 optimizer step 和 checkpoint reload；
- [ ] staged execution 决策已记录；
- [ ] 关键 commit/revision/image digest 已冻结；
- [ ] 旧 Day 1 结果中仍未验证的风险被列出，而不是静默忽略。

## 8. 当前执行记录

```text
状态：CORE_BASELINE_COMPLETED（依据 day1实现.md，仍需在服务器端确认版本锁文件）
模型：Qwen/Qwen3-4B-Instruct-2507
SGLang：已完成 TP1 8K smoke
Megatron：已完成 TP2 训练 smoke
已知风险：自定义 SDPA patch 的 TP 数值等价性仍需单独验证
下一阶段：Polar Calculator rollout
```
