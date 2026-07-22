#!/usr/bin/env bash
# Stop CUDA Milvus GPU docker stack started by start_milvus_cuda_gpu_docker.sh
set -euo pipefail

WORKDIR="${WORKDIR:-${HOME}/milvus_cuda_4080}"
COMPOSE_DIR="${COMPOSE_DIR:-${WORKDIR}/docker-gpu}"

if [ ! -f "${COMPOSE_DIR}/docker-compose.yml" ]; then
  echo "ERROR: no compose file at ${COMPOSE_DIR}/docker-compose.yml" >&2
  exit 1
fi

cd "${COMPOSE_DIR}"
docker compose down || docker-compose down
echo "Stopped stack in ${COMPOSE_DIR}"
