# Local Model Experiments（已归档）

这些脚本属于旧的自定义训练原型，不再代表当前推荐架构，仅保留实验历史。

```text
harness.py                  共享 micro-harness：action 协议、安全工具执行、渲染/掩码
baseline_local_harness.py   零样本基线 + SFT 后评估（同一 harness）
build_sft_dataset.py        从真实 Pi v2.1-live candidate fixtures 构建 SFT 数据
sft_train.py                LoRA SFT 训练（GPU 上几十秒）
eval_generalization.py      泛化验证（task-1..5，基线 vs SFT）
grpo_lite.py                GRPO-lite（on-policy RL 冒烟）
```

## 运行顺序

```bash
export PYTHONPATH=/path/to/repo
# 1. 基线（未训练）
python3 baseline_local_harness.py --model /root/models/qwen2.5-1.5b-instruct
# 2. 构建 SFT 数据（需真实 fixtures: tests/fixtures/pi/v2.1-live）
python3 build_sft_dataset.py
# 3. LoRA SFT
python3 sft_train.py --model /root/models/qwen2.5-1.5b-instruct \
  --data experiments/local_model/sft-dataset.jsonl --output /root/models/qwen-sft-adapter
# 4. 基线 vs SFT（task-1..5）
python3 baseline_local_harness.py --model /root/models/qwen2.5-1.5b-instruct \
  --adapter /root/models/qwen-sft-adapter
# 5. GRPO-lite（on-policy）
python3 grpo_lite.py --model /root/models/qwen2.5-1.5b-instruct \
  --sft /root/models/qwen-sft-adapter --output /root/models/qwen-grpo-adapter
```

## 预期结果（真实 fixtures，1.5B）

```text
基线（零样本）：0/5
SFT 后：5/5（含训练时未见 task-4/5；种子方差，最好种子 5/5、其余 4/5）
GRPO-lite 训练后：4~5/5（task-3 计数错误是 1.5B 能力边界）
```

GPU 注意：训练用 `--no-bf16` 时模型也会强制 fp32；评估建议 `LOCAL_MODEL_FP32=1`（eager attention）；
完整踩坑记录见 [docs/releases/v2.5-local-model-experiments.md](../../docs/releases/v2.5-local-model-experiments.md)。
