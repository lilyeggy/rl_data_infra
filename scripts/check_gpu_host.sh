#!/usr/bin/env bash
# Day-01 host preflight: GPU, driver, disk, memory, docker.
# Usage: bash scripts/check_gpu_host.sh | tee artifacts/day-01/host-preflight.txt
set -u

echo "===== date ====="
date -Is

echo "===== os ====="
. /etc/os-release && echo "$NAME $VERSION"
uname -r

echo "===== gpu (nvidia-smi) ====="
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version,persistence_mode --format=csv

echo "===== gpu count check ====="
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "gpu_count=$GPU_COUNT (expect 2)"

echo "===== xid errors (dmesg) ====="
if dmesg 2>/dev/null | grep -i xid; then
  echo "WARNING: Xid errors found"
else
  echo "no Xid errors (or dmesg not readable)"
fi

echo "===== ecc errors ====="
nvidia-smi --query-gpu=index,ecc.errors.uncorrected.volatile.total --format=csv 2>/dev/null || true

echo "===== disk ====="
df -h / /data
echo "expect: data disk free >= 200GB"

echo "===== memory ====="
free -h
echo "expect: total >= 128GB"

echo "===== cpu ====="
nproc

echo "===== docker ====="
if command -v docker >/dev/null; then
  docker --version
  docker info --format 'server={{.ServerVersion}} root={{.DockerRootDir}}' 2>/dev/null \
    || sudo -n docker info --format 'server={{.ServerVersion}} root={{.DockerRootDir}}' 2>/dev/null \
    || echo "docker info needs sudo/group re-login"
else
  echo "docker NOT installed"
fi

echo "===== toolchain ====="
python3 --version 2>&1
uv --version 2>&1
git --version
git lfs version 2>&1

echo "===== preflight done ====="
