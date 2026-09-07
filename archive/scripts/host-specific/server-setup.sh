#!/bin/bash
# ARCHIVED: retired server bootstrap; do not run on the current host.
# server-setup.sh — run ON the new GPU server. Idempotent.
# Installs env, downloads the base model, registers the Pi local provider.
# Mirrors + blockers learned on the previous host are baked in:
#   - pypi via tsinghua/aliyun mirror (default pypi unreachable)
#   - model via modelscope (github unreachable)
#   - Pi via npm @earendil-works/pi-coding-agent
set -e
MODEL_DIR=/root/rivermind-data/models
mkdir -p "$MODEL_DIR" /root/.pi/agent

echo "== [1/5] pip mirror + python deps =="
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || true
pip config set global.break-system-packages true 2>/dev/null || true
pip config set global.root-user-action ignore 2>/dev/null || true
python3 -m pip install --quiet torch transformers peft accelerate fastapi "uvicorn[standard]" modelscope

echo "== [2/5] node + Pi CLI =="
if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs
fi
npm install -g @earendil-works/pi-coding-agent@0.84.2
# Pi's find/grep tools are backed by fd/ripgrep; without them find is broken
# (learned the hard way on the previous host).
apt-get install -y ripgrep fd-find 2>/dev/null || true
ln -sf /usr/bin/fdfind /usr/local/bin/fd 2>/dev/null || true

echo "== [3/5] download Qwen2.5-7B-Instruct via modelscope =="
python3 - <<'PY'
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct',
                  local_dir='/root/rivermind-data/models/qwen2.5-7b-instruct')
print("model ready")
PY

echo "== [4/5] register Pi local provider =="
cat > /root/.pi/agent/models.json <<'JSON'
{
  "providers": {
    "local-qwen": {
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "supportsUsageInStreaming": true
      },
      "models": [
        { "id": "qwen2.5-7b-instruct", "name": "Qwen2.5 7B Local", "reasoning": false,
          "input": ["text"], "contextWindow": 32768, "maxTokens": 4096,
          "cost": {"input":0,"output":0,"cacheRead":0,"cacheWrite":0} }
      ]
    }
  }
}
JSON

echo "== [5/5] sanity =="
python3 -c "import torch,transformers,peft,fastapi; print('deps ok, cuda=', torch.cuda.is_available())"
pi --version | head -1
echo "SETUP_DONE"
