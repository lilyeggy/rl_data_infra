#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/capture_day01_evidence.sh [workspace]

Arguments:
  workspace  Upstream workspace containing src/, models/, and artifacts/.
             Default: /data/day-01-workspace

This script records read-only environment and source facts. It does not install,
start, stop, or modify Polar, SGLang, Slime, Megatron, or model weights.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

workspace="${1:-/data/day-01-workspace}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
output_dir="${workspace}/artifacts/day-01"
source_dir="${workspace}/src"

if [[ ! -d "${workspace}" ]]; then
  printf 'Workspace does not exist: %s\n' "${workspace}" >&2
  exit 1
fi

mkdir -p "${output_dir}"

bash "${script_dir}/check_gpu_host.sh" >"${output_dir}/host-preflight.txt" 2>&1

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi topo -m >"${output_dir}/gpu-topology.txt" 2>&1 || true
else
  printf '%s\n' 'nvidia-smi: not found' >"${output_dir}/gpu-topology.txt"
fi

{
  printf 'captured_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'workspace=%s\n' "${workspace}"
  printf 'project_dir=%s\n' "${project_dir}"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'user=%s\n' "$(id -un)"
  printf 'pwd=%s\n' "$(pwd)"
  printf 'source_dir=%s\n' "${source_dir}"
  printf 'models_dir=%s\n' "${workspace}/models"
} >"${output_dir}/paths.txt"

repos=(
  polar
  slime
  Megatron-LM
  sglang
  mbridge-iseekyan
  mbridge
)

{
  printf 'captured_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for repo_name in "${repos[@]}"; do
    repo_path="${source_dir}/${repo_name}"
    if [[ -d "${repo_path}/.git" ]]; then
      printf '%s.path=%s\n' "${repo_name}" "${repo_path}"
      printf '%s.commit=%s\n' "${repo_name}" "$(git -C "${repo_path}" rev-parse HEAD)"
      printf '%s.branch=%s\n' "${repo_name}" "$(git -C "${repo_path}" branch --show-current || true)"
      printf '%s.describe=%s\n' "${repo_name}" "$(git -C "${repo_path}" describe --tags --always --dirty)"
      if [[ -n "$(git -C "${repo_path}" status --porcelain)" ]]; then
        printf '%s.dirty=true\n' "${repo_name}"
      else
        printf '%s.dirty=false\n' "${repo_name}"
      fi
    else
      printf '%s.error=missing_git_repository:%s\n' "${repo_name}" "${repo_path}"
    fi
  done
} >"${output_dir}/source-versions.txt"

{
  printf 'captured_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  environments=(
    "rollout:${source_dir}/polar/.venv/bin/python"
    "training:${source_dir}/slime/.venv/bin/python"
  )
  for environment in "${environments[@]}"; do
    environment_name="${environment%%:*}"
    python_path="${environment#*:}"
    printf '\n[%s]\n' "${environment_name}"
    if [[ -x "${python_path}" ]]; then
      "${python_path}" --version
      "${python_path}" -m pip freeze || true
    else
      printf 'python_missing=%s\n' "${python_path}"
    fi
  done
} >"${output_dir}/python-packages.txt" 2>&1

{
  printf 'captured_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if command -v docker >/dev/null 2>&1; then
    docker images --digests --no-trunc \
      --format '{{.Repository}}:{{.Tag}} digest={{.Digest}} id={{.ID}}' || true
  else
    printf '%s\n' 'docker: not found'
  fi
} >"${output_dir}/container-images.txt" 2>&1

printf 'Day 1 evidence captured in %s\n' "${output_dir}"
printf '%s\n' 'Next: review source-versions.txt for missing/dirty repositories before creating configs/upstream-lock.yaml.'
