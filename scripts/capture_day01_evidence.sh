#!/usr/bin/env bash
#
# Day-01 closeout: read-only environment evidence capture.
#
# Usage:
#   bash scripts/capture_day01_evidence.sh /data/day-01-workspace
#
# The script ONLY inspects the environment. It does NOT start/stop any service,
# does NOT modify upstream sources or model weights, and does NOT touch the
# Docker daemon. Output is written to:
#   <workspace>/artifacts/day-01/{host-preflight,gpu-topology,source-versions,python-packages,container-images,paths}.txt
#
set -euo pipefail

WS="${1:?usage: capture_day01_evidence.sh <workspace_dir>}"
SRC="$WS/src"
OUT="$WS/artifacts/day-01"
mkdir -p "$OUT"

now() { date '+%Y-%m-%dT%H:%M:%S%z (%Z)'; }
now_utc() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

# docker may need sudo if the session group membership is stale
dk() {
  if docker "$@" >/dev/null 2>&1; then
    docker "$@"
  else
    sudo -n docker "$@" 2>/dev/null || docker "$@"
  fi
}

# ---------------------------------------------------------------- host-preflight
{
  echo "===== date ====="
  now
  echo "===== hostname ====="
  hostname
  echo "===== kernel ====="
  uname -a
  echo "===== cpu / ram ====="
  echo "cores: $(nproc)"
  grep -E 'MemTotal|MemAvailable|SwapTotal' /proc/meminfo
  echo "===== disk ====="
  df -h "$WS" /
  echo "expect: data disk free >= 200GB"
  echo "===== gpu (nvidia-smi -L) ====="
  nvidia-smi -L 2>/dev/null || echo "nvidia-smi not available"
  echo "===== gpu detail ====="
  nvidia-smi --query-gpu=index,name,uuid,memory.total,compute_cap,driver_version --format=csv,noheader 2>/dev/null
  echo "===== nvidia-smi CUDA version ====="
  nvidia-smi 2>/dev/null | grep -E 'CUDA Version' | head -1
  echo "===== docker ====="
  echo "client: $(docker version --format '{{.Client.Version}}' 2>/dev/null || echo 'n/a')"
  echo "server: $(docker version --format '{{.Server.Version}} (api {{.Server.ApiVersion}})' 2>/dev/null || echo 'n/a (needs daemon access)')"
  echo "data-root: $(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo 'n/a')"
  echo "===== nvidia-container-toolkit ====="
  nvidia-container-cli --version 2>/dev/null | head -1 || echo "nvidia-container-cli not found"
  echo "===== python toolchain ====="
  python3 --version 2>/dev/null || echo "python3 not found"
  uv --version 2>/dev/null || echo "uv not found"
  git --version 2>/dev/null || echo "git not found"
} > "$OUT/host-preflight.txt"

# ---------------------------------------------------------------- gpu-topology
{
  echo "===== date ====="
  now
  echo "===== nvidia-smi topo -m ====="
  nvidia-smi topo -m
  echo "===== nvidia-smi topo -mp ====="
  nvidia-smi topo -mp
} > "$OUT/gpu-topology.txt"

# ---------------------------------------------------------------- source-versions
{
  echo "===== date ====="
  now
  echo "===== toolchain ====="
  echo "python3: $(python3 --version 2>/dev/null || echo 'n/a')"
  echo "uv: $(uv --version 2>/dev/null || echo 'n/a')"
  echo "git: $(git --version 2>/dev/null || echo 'n/a')"
  echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo 'n/a')"
  nvidia-smi 2>/dev/null | grep -E 'CUDA Version' | head -1
  echo
  echo "===== upstream source checkouts (under $SRC) ====="
  for d in "$SRC"/*/; do
    repo="${d%/}"
    name="$(basename "$repo")"
    if [ ! -d "$repo/.git" ]; then
      echo "--- $name: NOT A GIT REPO ---"
      continue
    fi
    echo "--- $name ---"
    head_full="$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo '<no HEAD>')"
    echo "$head_full"
    branch="$(git -C "$repo" branch --show-current 2>/dev/null || echo '')"
    desc="$(git -C "$repo" describe --tags --exact-match 2>/dev/null || git -C "$repo" describe --tags 2>/dev/null || echo 'no tag')"
    echo "ref: branch=${branch:-<detached>} describe=${desc}"
    echo "-- git status --short --"
    dirty="$(git -C "$repo" status --short 2>/dev/null)"
    if [ -n "$dirty" ]; then
      echo "$dirty"
      echo "-- dirty: local-patch-checksum (git diff HEAD | sha256sum) --"
      git -C "$repo" diff HEAD 2>/dev/null | sha256sum
    else
      echo "(clean)"
    fi
    echo "-- git remote -v --"
    git -C "$repo" remote -v 2>/dev/null || echo "(no remotes)"
    echo
  done
  echo "===== hf cache model snapshot revisions ====="
  for m in "$WS"/hf-cache/hub/models--*/snapshots/*/; do
    [ -d "$m" ] || continue
    rev="$(basename "$m")"
    model="$(basename "$(dirname "$(dirname "$m")")")"
    echo "$model  $rev"
  done
  echo
  echo "===== megatron patch details (from slime v0.3.0) ====="
  patch_file="$SRC/slime/docker/patch/v0.5.12.post1/megatron.patch"
  if [ -f "$patch_file" ]; then
    echo "slime megatron.patch: $patch_file"
    echo "  sha256: $(sha256sum "$patch_file" | cut -d' ' -f1)"
  else
    echo "slime megatron.patch: NOT FOUND at $patch_file"
  fi
} > "$OUT/source-versions.txt"

# ---------------------------------------------------------------- python-packages
{
  echo "===== date ====="
  now
  for venv in "$SRC/polar/.venv" "$SRC/slime/.venv"; do
    if [ ! -x "$venv/bin/python" ]; then
      echo "===== env ====="
      echo "venv: $venv NOT FOUND"
      continue
    fi
    echo "===== env ====="
    echo "venv: $venv"
    "$venv/bin/python" --version 2>/dev/null
    echo "uv: $(uv --version 2>/dev/null || echo 'n/a')"
    echo "-- key packages --"
    uv pip list --python "$venv/bin/python" 2>/dev/null \
      | grep -iE '^(torch|torchaudio|torchvision|sglang|flash-attn|flashinfer|transformers|triton|megatron|slime|mbridge|numpy|safetensors|sentencepiece|xgrammar|ring-flash|protobuf)\b' || echo "(grep matched nothing)"
    echo "-- torch cuda check --"
    timeout 90 "$venv/bin/python" -c "import torch; print('cuda_available=', torch.cuda.is_available(), 'device_count=', torch.cuda.device_count(), 'cap=', torch.cuda.get_device_capability() if torch.cuda.is_available() else None)" 2>/dev/null || echo "(torch import failed/timed out)"
    echo "-- full uv pip list --"
    uv pip list --python "$venv/bin/python" 2>/dev/null || echo "(uv pip list failed)"
    echo
  done
} > "$OUT/python-packages.txt"

# ---------------------------------------------------------------- container-images
{
  echo "===== date ====="
  now
  echo "===== container images (docker) ====="
  dk images --format '{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Digest}}' 2>/dev/null || echo "docker daemon not accessible"
  echo
  echo "===== notes ====="
  echo "docker.io (registry-1.docker.io) 不可达；base 镜像经代理 docker.1panel.live 拉取后 tag 为本地名。"
  echo "本地构建镜像无 RepoDigest，用 image ID (sha256) 作为内容 digest。"
} > "$OUT/container-images.txt"

# ---------------------------------------------------------------- paths
{
  echo "===== date ====="
  now
  echo "Day-01 路径规划（全部落数据盘 /data）"
  echo "source root:   $SRC"
  for d in "$SRC"/*/; do
    repo="${d%/}"
    [ -d "$repo/.git" ] || continue
    echo "  $(basename "$repo"): $repo   ($(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo 'no HEAD'))"
  done
  echo "rollout env:   $SRC/polar/.venv   ($([ -x "$SRC/polar/.venv/bin/python" ] && echo present || echo MISSING))"
  echo "train env:     $SRC/slime/.venv   ($([ -x "$SRC/slime/.venv/bin/python" ] && echo present || echo MISSING))"
  echo "hf cache:      $WS/hf-cache   ($([ -d "$WS/hf-cache" ] && echo present || echo MISSING))"
  echo "model dir:     $WS/models   ($([ -d "$WS/models" ] && echo present || echo MISSING))"
  echo "docker root:   $WS/docker-root   ($([ -d "$WS/docker-root" ] && echo present || echo MISSING))"
  echo "artifacts:     $WS/artifacts"
} > "$OUT/paths.txt"

echo "captured -> $OUT"
ls -la "$OUT"
