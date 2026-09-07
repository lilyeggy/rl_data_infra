#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-agentic-rl-data-plane:local}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed; container smoke test skipped" >&2
  exit 2
fi
if ! docker info >/dev/null 2>&1; then
  echo "docker CLI is installed but the daemon is unavailable; smoke test skipped" >&2
  exit 2
fi

docker build --tag "${IMAGE_NAME}" .
docker run --rm "${IMAGE_NAME}" python -m unittest discover -s tests -v
printf '%s\n' 'container smoke passed: production package and complete tests loaded'
