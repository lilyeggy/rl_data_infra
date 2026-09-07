# 新服务器部署 Runbook（已归档：旧 RTX 4090 实验环境）

> 目标：在 GPU 服务器上完整复现"真实 Harness 实验 + 本地模型实验"。
> 原则：旧服务器在**新服务器全部验证通过前不释放**（作为兜底）。
> 状态标记：✅=可验证 checkpoint；⚠️=注意事项。

## 0. 决策顺序

```text
1. 创建/租用 GPU 实例（RTX 4090，海南）
2. 按本 runbook 部署并逐项验证
3. 新服务器全部 checkpoint 通过后，再释放旧服务器
4. 释放前：备份 /root/agentic-rl-artifacts 与 /root/models 必要产物
```

## 1. 前置安全（旧服务器也要做）

```text
⚠️ SSH 密码已出现在聊天记录中：
   1. 云平台重置 root 密码，改用 SSH key
   2. opencode-go API key 可轮换（如果担心泄露）
   3. 本 runbook 中的所有命令不要内嵌密码明文（用 sshpass 从环境变量读）
```

## 2. 基础环境检查

```bash
hostname; uname -r; uname -m
nproc; free -h; df -h /
nvidia-smi          # ✅ 必须显示 RTX 4090 / 24GB（真 GPU 验证）
python3 --version   # 期望 3.10+（3.12 也可）
```

网络源验证：

```bash
curl -sI https://download.pytorch.org/whl/cu124/ | head -1   # PyTorch GPU 源
curl -sI https://mirrors.aliyun.com/pypi/simple/ | head -1   # pip 镜像
curl -sI https://modelscope.cn | head -1                     # 模型源
curl -sI https://registry.npmmirror.com | head -1            # npm 镜像
```

## 3. 安装 Node.js + Pi 0.84.2

```bash
export PATH=/usr/local/bin:$PATH
cd /tmp
curl -fL --retry 3 -o node.tar.xz \
  https://npmmirror.com/mirrors/node/v22.16.0/node-v22.16.0-linux-x64.tar.xz
tar -xJf node.tar.xz -C /usr/local --strip-components=1
node --version; npm --version                              # ✅ v22.x / 10.x

npm config set registry https://registry.npmmirror.com
npm install -g @earendil-works/pi-coding-agent@0.84.2 --no-fund --no-audit
pi --version                                               # ✅ 0.84.2
```

## 4. 同步 Pi 认证配置

从本机（Mac）执行（不要复制密钥值到聊天/日志）：

```bash
for f in auth.json models-store.json settings.json; do
  base64 < "$HOME/.pi/agent/$f" | sshpass -e ssh -p <PORT> root@<HOST> \
    "umask 077 && mkdir -p /root/.pi/agent && base64 -d > /root/.pi/agent/$f && chmod 600 /root/.pi/agent/$f"
done
sshpass -e ssh -p <PORT> root@<HOST> 'chmod 700 /root/.pi/agent; ls -la /root/.pi/agent'
```

验证：

```bash
pi --provider opencode-go --model deepseek-v4-flash --mode json --print \
  --no-session --no-context-files --no-extensions --no-skills \
  --tools read,grep,find,ls --thinking minimal "Say OK" > /tmp/probe.ndjson
grep -o '"stopReason":"[^"]*"' /tmp/probe.ndjson | sort | uniq -c   # ✅ 出现 stop
```

## 5. Python + ML 依赖（GPU 版！）

```bash
pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
pip3 install --no-cache-dir --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --timeout 60 --retries 5 transformers peft modelscope datasets numpy
python3 - <<'PY'
import torch, transformers, peft
print("torch", torch.__version__, "cuda=", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
# ✅ 期望 cuda=True 且显示 RTX 4090
```

## 6. 部署项目代码

本机执行（排除不需要的文件）：

```bash
tar --exclude='./.git' --exclude='./artifacts' --exclude='__pycache__' \
    --exclude='*/__pycache__' --exclude='./.ruff_cache' \
    --exclude='./project-guide-site' --exclude='./agentic-rl-framework.html' \
    -cf - . | sshpass -e ssh -p <PORT> root@<HOST> \
    'rm -rf /root/agentic-rl && mkdir -p /root/agentic-rl && tar -xf - -C /root/agentic-rl'
sshpass -e ssh -p <PORT> root@<HOST> \
  'cd /root/agentic-rl && find . -name "._*" -type f -delete && find . -type d -name __pycache__ -prune -exec rm -rf {} + && chown -R root:root /root/agentic-rl'
```

验证：

```bash
cd /root/agentic-rl
python3 -m unittest discover -s tests -v          # ✅ 期望 ~185 tests OK
python3 -m src.cli demo-v21-runtime --output /root/agentic-rl-artifacts/v2.1-runtime-evidence
python3 -m src.cli demo-v24 --output /root/agentic-rl-artifacts/v2.4-offline-training-view
```

## 7. 下载本地模型（Qwen2.5）

```bash
python3 - <<'PY'
from modelscope import snapshot_download
print(snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", local_dir="/root/models/qwen2.5-1.5b-instruct"))
PY
du -sh /root/models/qwen2.5-1.5b-instruct          # ✅ ~2.9G

# （可选，GPU 建议直接上 7B）
python3 - <<'PY'
from modelscope import snapshot_download
print(snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_dir="/root/models/qwen2.5-7b-instruct"))
PY
```

## 8. 本地模型实验（experiments/local_model/）

脚本在仓库内，全部支持 GPU（`device = cuda if available`，GPU 用 bf16）：

```bash
cd /root/agentic-rl/experiments/local_model
export PYTHONPATH=/root/agentic-rl

# 基线（零样本）+ SFT 后评估
python3 baseline_local_harness.py --model /root/models/qwen2.5-1.5b-instruct \
  --adapter /root/models/qwen-sft-adapter   # 期望 0/5 -> 5/5

# 构建 SFT 数据集（从真实 fixtures）
python3 build_sft_dataset.py

# LoRA SFT 训练（GPU 上几十秒）
python3 sft_train.py

# 泛化验证（task-1..5）
python3 eval_generalization.py

# GRPO-lite（on-policy；GPU 上每 step 十几秒）
python3 grpo_lite.py
```

GPU 注意：

```text
torch.set_num_threads 仅在 CPU 时设置
模型加载：cuda 时 torch_dtype=torch.bfloat16, model.to('cuda')
采样/评估输入 .to('cuda')
GRPO 的逐 rollout 梯度累积在 GPU 上不再需要（激活内存小），但保留也无害
```

## 9. 真实 Pi 实验（可选，验证数据面与训练闭环的源头）

```bash
cd /root/agentic-rl
export PATH=/usr/local/bin:$PATH
python3 -m src.cli run-real-experiment \
  --output /root/agentic-rl-artifacts/v2.1-live \
  --model deepseek-v4-flash \
  --workspace /root/pi-live-workspace --timeout 600
# ✅ 期望 control 0/3、candidate 3/3、Gate REJECT(latency)
```

## 10. 最终验证清单

```text
[ ] nvidia-smi 显示 RTX 4090 / 24GB
[ ] pi --version = 0.84.2
[ ] torch.cuda.is_available() = True
[ ] 全量测试 ~185 通过
[ ] demo-v21-runtime / demo-v24 产物生成
[ ] Qwen 模型下载完成
[ ] 基线 0/5、SFT 5/5、泛化 5/5、GRPO 5/5
[ ] （可选）真实 Pi run-real-experiment 复现
```

## 11. 释放旧服务器前的备份清单

```text
/root/agentic-rl-artifacts/      实验产物（全部）
/root/models/                    qwen-sft-adapter、qwen-grpo-adapter（如需保留）
/root/pi-probe*.ndjson           真实 Pi 证据（如需要）
```

## 12. 成本与时长预估（GPU 实例）

```text
4090 实例通常 ¥3-8/小时，按量
环境部署：~30-60 分钟
SFT：~1 分钟
GRPO-lite（3 step × 9 rollout）：~5-10 分钟
全量复现：半天内可完成
```
