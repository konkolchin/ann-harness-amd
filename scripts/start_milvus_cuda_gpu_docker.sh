#!/usr/bin/env bash
# Start Milvus v2.5.4 GPU standalone via Docker Compose (CUDA / RTX 4080).
#
# Prerequisites: nvidia-smi, docker, NVIDIA Container Toolkit
# Usage:
#   export WORKDIR=~/milvus_cuda_4080
#   bash scripts/start_milvus_cuda_gpu_docker.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${HOME}/milvus_cuda_4080}"
COMPOSE_DIR="${COMPOSE_DIR:-${WORKDIR}/docker-gpu}"
MILVUS_IMAGE="${MILVUS_IMAGE:-milvusdb/milvus:v2.5.4-gpu}"
COMPOSE_URL="${COMPOSE_URL:-https://github.com/milvus-io/milvus/releases/download/v2.5.4/milvus-standalone-docker-compose-gpu.yml}"
FALLBACK_COMPOSE="${REPO_ROOT}/docker/milvus-standalone-docker-compose-gpu-v2.5.4.yml"

mkdir -p "${COMPOSE_DIR}" "${WORKDIR}/logs"
cd "${COMPOSE_DIR}"

if [ ! -f docker-compose.yml ]; then
  echo "==> downloading GPU compose from ${COMPOSE_URL}"
  if ! wget -q -O docker-compose.yml "${COMPOSE_URL}" \
    && ! curl -fsSL -o docker-compose.yml "${COMPOSE_URL}"; then
    echo "==> download failed; using repo fallback ${FALLBACK_COMPOSE}"
    cp "${FALLBACK_COMPOSE}" docker-compose.yml
  fi
fi

# Pin image tag for fair compare with AMD HIP v2.5.4 line
if grep -q 'milvusdb/milvus:' docker-compose.yml; then
  sed -i.bak -E "s|milvusdb/milvus:[^\"[:space:]]+|${MILVUS_IMAGE}|g" docker-compose.yml
fi

echo "==> image=${MILVUS_IMAGE}"
echo "==> compose dir=${COMPOSE_DIR}"
# Prefer Compose v2 GPU reservation; also set NVIDIA_VISIBLE_DEVICES for older runtimes
export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-0}"
docker compose pull || docker-compose pull || true
docker compose up -d || docker-compose up -d

echo "==> waiting for healthz on :9091"
for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:9091/healthz >/dev/null 2>&1; then
    echo "OK healthy (${i}s)"
    docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | head -20
    exit 0
  fi
  sleep 2
done

echo "ERROR: milvus not healthy; docker logs:" >&2
docker logs milvus-standalone 2>&1 | tail -80 >&2 || true
exit 1
