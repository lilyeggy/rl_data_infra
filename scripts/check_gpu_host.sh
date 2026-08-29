#!/usr/bin/env bash

set -u

section() {
  printf '\n[%s]\n' "$1"
}

run_if_available() {
  local command_name="$1"
  shift
  if command -v "$command_name" >/dev/null 2>&1; then
    "$command_name" "$@" || true
  else
    printf '%s: not found\n' "$command_name"
  fi
}

section "Operating system"
run_if_available uname -a
if [[ -r /etc/os-release ]]; then
  sed -n '1,20p' /etc/os-release
fi

section "CPU and memory"
run_if_available lscpu
run_if_available free -h

section "Disk"
run_if_available df -h

section "NVIDIA driver and GPUs"
run_if_available nvidia-smi

section "GPU inventory"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id,compute_cap --format=csv,noheader || true
fi

section "GPU topology"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi topo -m || true
fi

section "CUDA compiler"
run_if_available nvcc --version

section "Docker"
run_if_available docker --version
if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    docker info --format 'Server={{.ServerVersion}} Driver={{.Driver}} Root={{.DockerRootDir}}' || true
    docker system df || true
  else
    printf '%s\n' 'Docker CLI is installed, but the daemon is unavailable.'
  fi
fi

section "Python"
run_if_available python3 --version
if command -v python3 >/dev/null 2>&1; then
  python3 -c 'import platform; print(platform.platform())' || true
  if python3 -c 'import torch' >/dev/null 2>&1; then
    python3 -c 'import torch; print("torch", torch.__version__); print("cuda", torch.version.cuda); print("cuda_available", torch.cuda.is_available()); print("gpu_count", torch.cuda.device_count())' || true
  else
    printf '%s\n' 'PyTorch is not installed in the active Python environment.'
  fi
fi

section "Environment preflight complete"
printf '%s\n' 'Save this output with the machine configuration. Missing optional tools are reported but do not make the script fail.'
