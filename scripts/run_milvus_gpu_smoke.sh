#!/usr/bin/env bash
# Run Layer 3 Milvus GPU smoke: start deps + HIP milvus, then GPU_IVF_FLAT client.
#
# Usage:
#   bash scripts/run_milvus_gpu_smoke.sh
#   bash scripts/run_milvus_gpu_smoke.sh --skip-client   # only start server
#   MILVUS_BIN=/path/to/milvus bash scripts/run_milvus_gpu_smoke.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
INSTALL_PREFIX="${INSTALL_PREFIX:-${WORKDIR}/install}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
MILVUS_DIR="${MILVUS_DIR:-${WORKDIR}/milvus}"
SHIM_DIR="${SHIM_DIR:-${WORKDIR}/libshims}"
COMPOSE_DIR="${COMPOSE_DIR:-${REPO_ROOT}/milvus-docker}"
URI="${MILVUS_URI:-http://127.0.0.1:19530}"
SKIP_CLIENT=0
for a in "$@"; do
  case "$a" in
    --skip-client) SKIP_CLIENT=1 ;;
  esac
done

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export INSTALL_PREFIX ROCM_PATH
export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}:/usr/lib/x86_64-linux-gnu"

# Reuse Layer-2 gflags shim if present (Conan glog / gflags namespace).
_gflags_preload="${SHIM_DIR}/libgflags_gflagsns.so"
if [ -e "${_gflags_preload}" ]; then
  export LD_PRELOAD="${_gflags_preload}${LD_PRELOAD:+:${LD_PRELOAD}}"
  echo "Using gflags preload: ${_gflags_preload}"
fi

find_milvus_bin() {
  if [ -n "${MILVUS_BIN:-}" ] && [ -x "${MILVUS_BIN}" ]; then
    echo "${MILVUS_BIN}"
    return 0
  fi
  local c
  for c in \
    "${MILVUS_DIR}/bin/milvus" \
    "${MILVUS_DIR}/cmake_build/milvus" \
    "${MILVUS_DIR}/internal/core/output/bin/milvus" \
    "${MILVUS_DIR}/out/bin/milvus"
  do
    if [ -x "$c" ]; then echo "$c"; return 0; fi
  done
  # last resort
  find "${MILVUS_DIR}" -name milvus -type f -executable 2>/dev/null | head -1
}

MILVUS_BIN="$(find_milvus_bin || true)"
if [ -z "${MILVUS_BIN}" ]; then
  echo "ERROR: milvus binary not found under ${MILVUS_DIR}" >&2
  echo "  Build first: bash scripts/build_milvus_layer3.sh" >&2
  exit 1
fi
echo "milvus binary: ${MILVUS_BIN}"

# Start etcd + minio from harness compose; leave stock milvus-standalone stopped.
if [ -f "${COMPOSE_DIR}/docker-compose.yml" ]; then
  echo "==> ensure etcd/minio (stop stock milvus-standalone if running)"
  cd "${COMPOSE_DIR}"
  docker-compose stop standalone 2>/dev/null || true
  docker-compose up -d etcd minio
  sleep 5
  docker-compose ps
else
  echo "WARNING: ${COMPOSE_DIR}/docker-compose.yml missing; assume etcd/minio already up" >&2
fi

# Minimal milvus.yaml / env for standalone (use milvus defaults + user.yaml simd if present)
_cfg_dir="${WORKDIR}/milvus_gpu_config"
mkdir -p "${_cfg_dir}"
if [ -f "${COMPOSE_DIR}/user.yaml" ]; then
  cp -f "${COMPOSE_DIR}/user.yaml" "${_cfg_dir}/user.yaml"
fi

_log="${WORKDIR}/milvus_gpu_standalone.log"
echo "==> start HIP milvus standalone (log: ${_log})"
# Common Milvus env for docker-compose parity
export ETCD_ENDPOINTS="${ETCD_ENDPOINTS:-127.0.0.1:2379}"
export MINIO_ADDRESS="${MINIO_ADDRESS:-127.0.0.1:9000}"
export MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minioadmin}"
export MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-minioadmin}"

# Kill previous HIP milvus if we started one
if [ -f "${WORKDIR}/milvus_gpu.pid" ]; then
  _old="$(cat "${WORKDIR}/milvus_gpu.pid" || true)"
  if [ -n "${_old}" ] && kill -0 "${_old}" 2>/dev/null; then
    echo "Stopping previous milvus pid ${_old}"
    kill "${_old}" || true
    sleep 2
  fi
fi

# Docker etcd/minio publish to host; milvus binary must reach them on localhost.
# If compose only exposes inside network, map ports — stock compose publishes 9000.
nohup "${MILVUS_BIN}" run standalone >"${_log}" 2>&1 &
echo $! >"${WORKDIR}/milvus_gpu.pid"
echo "milvus pid $(cat "${WORKDIR}/milvus_gpu.pid")"

echo "==> wait for healthz"
_ok=0
for _i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:9091/healthz >/dev/null 2>&1; then
    echo "healthz OK"
    _ok=1
    break
  fi
  sleep 3
done
if [ "${_ok}" -ne 1 ]; then
  echo "ERROR: milvus healthz not up; last log lines:" >&2
  tail -80 "${_log}" >&2 || true
  exit 1
fi

if [ "${SKIP_CLIENT}" -eq 1 ]; then
  echo "Server up; skipping client (--skip-client)."
  echo "  python ${REPO_ROOT}/scripts/run_milvus_hdf5.py --uri ${URI} --index-type GPU_IVF_FLAT --nlist 128 --nprobes 8,16"
  exit 0
fi

echo "==> GPU_IVF_FLAT smoke via run_milvus_hdf5.py"
_data="${REPO_ROOT}/data/sift-128-euclidean.hdf5"
if [ ! -f "${_data}" ]; then
  _data="${WORKDIR}/sift-128-euclidean.hdf5"
fi
if [ ! -f "${_data}" ]; then
  echo "WARNING: SIFT hdf5 not found; creating tiny random smoke is not supported here." >&2
  echo "  Place sift-128-euclidean.hdf5 under ${REPO_ROOT}/data/ or set --data" >&2
  exit 1
fi

python3 "${REPO_ROOT}/scripts/run_milvus_hdf5.py" \
  --uri "${URI}" \
  --collection sift_gpu_smoke \
  --data "${_data}" \
  --index-type GPU_IVF_FLAT \
  --nlist 128 \
  --nprobes 8,16 \
  --insert-batch 10000 \
  --max-train-rows 50000 \
  --max-query-rows 500 \
  --k 10

echo ""
echo "Layer 3 smoke finished. Check ${_log} for HIP/Knowhere GPU lines."
echo "Stop server: kill \$(cat ${WORKDIR}/milvus_gpu.pid)"
