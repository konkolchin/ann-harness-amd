#!/usr/bin/env bash
# Layer-3 smoke: start HIP Milvus Standalone and run GPU_IVF_FLAT on a SIFT slice.
#
# Prerequisites:
#   - bin/milvus from scripts/build_milvus_layer3.sh
#   - Layer 1.5 libs under $INSTALL_PREFIX
#   - etcd + minio reachable (default: start via docker compose in milvus/)
#   - data/sift-128-euclidean.hdf5 (or DATA_PATH)
#
# Usage:
#   bash scripts/run_milvus_gpu_smoke.sh
#   SKIP_START=1 bash scripts/run_milvus_gpu_smoke.sh   # milvus already running
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
INSTALL_PREFIX="${INSTALL_PREFIX:-${WORKDIR}/install}"
MILVUS_DIR="${MILVUS_DIR:-${WORKDIR}/milvus}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
SHIM_DIR="${SHIM_DIR:-${WORKDIR}/libshims}"
URI="${MILVUS_URI:-http://127.0.0.1:19530}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/sift-128-euclidean.hdf5}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
MILVUS_LOG="${LOG_DIR}/milvus_gpu_standalone.log"
PID_FILE="${LOG_DIR}/milvus_gpu_standalone.pid"

export MILVUS_HIP_INSTALL_PREFIX="${MILVUS_HIP_INSTALL_PREFIX:-${INSTALL_PREFIX}}"
export ROCM_PATH
export PATH="${ROCM_PATH}/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${MILVUS_HIP_INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:${MILVUS_DIR}/internal/core/output/lib:${LD_LIBRARY_PATH:-}"
if [ -d "${SHIM_DIR}" ]; then
  export LD_LIBRARY_PATH="${SHIM_DIR}:${LD_LIBRARY_PATH}"
fi

# Duplicate gflags (Milvus Conan + Knowhere Conan) aborts before main with:
#   flag 'help' was defined more than once
# Prepare a single threaded libgflags LD_PRELOAD target (helps dynamic case).
if [ -x "${REPO_ROOT}/scripts/prepare_milvus_gflags.sh" ]; then
  bash "${REPO_ROOT}/scripts/prepare_milvus_gflags.sh" || true
fi
# Do NOT preload apt spdlog by default — it can mask/conflict with Knowhere.
# Only preload gflags shim unless the caller already set LD_PRELOAD.
if [ -z "${LD_PRELOAD:-}" ] && [ -e "${SHIM_DIR}/libgflags_gflagsns.so" ]; then
  export LD_PRELOAD="${SHIM_DIR}/libgflags_gflagsns.so"
  echo "==> LD_PRELOAD=${LD_PRELOAD}"
fi

mkdir -p "${LOG_DIR}"

_milvus_bin=""
for c in \
  "${MILVUS_DIR}/bin/milvus" \
  "${MILVUS_DIR}/internal/core/output/bin/milvus"
do
  if [ -x "$c" ]; then _milvus_bin="$c"; break; fi
done
if [ -z "${_milvus_bin}" ]; then
  echo "ERROR: milvus binary not found under ${MILVUS_DIR}" >&2
  echo "  Build first: bash scripts/build_milvus_layer3.sh" >&2
  exit 1
fi
echo "==> milvus: ${_milvus_bin}"

if [ ! -f "${DATA_PATH}" ]; then
  echo "ERROR: SIFT hdf5 not found: ${DATA_PATH}" >&2
  echo "  wget --no-proxy https://ann-benchmarks.com/sift-128-euclidean.hdf5 -O ${DATA_PATH}" >&2
  exit 1
fi

# Stop stock Docker standalone if it holds :19530
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'milvus-standalone'; then
  echo "==> stopping docker milvus-standalone (frees :19530)"
  docker stop milvus-standalone >/dev/null || true
fi

# Start etcd/minio via milvus docker-compose if present
_compose="${MILVUS_DIR}/deployments/docker/dev/docker-compose.yml"
if [ ! -f "${_compose}" ]; then
  _compose="${MILVUS_DIR}/docker-compose.yml"
fi
if [ "${SKIP_DEPS:-0}" != "1" ] && [ -f "${_compose}" ] && command -v docker >/dev/null; then
  echo "==> ensuring etcd/minio from ${_compose}"
  (cd "$(dirname "${_compose}")" && docker compose -f "$(basename "${_compose}")" up -d etcd minio 2>/dev/null) \
    || (cd "$(dirname "${_compose}")" && docker-compose -f "$(basename "${_compose}")" up -d etcd minio 2>/dev/null) \
    || echo "NOTE: could not start etcd/minio via compose; ensure they are already running" >&2
fi

wait_ready() {
  local i
  for i in $(seq 1 90); do
    if curl -sf "${URI%/}/v1/vector/collections" >/dev/null 2>&1 \
      || curl -sf "http://127.0.0.1:9091/healthz" >/dev/null 2>&1; then
      echo "==> milvus healthy (${i}s)"
      return 0
    fi
    # pymilvus ping alternative: TCP
    if (echo >/dev/tcp/127.0.0.1/19530) >/dev/null 2>&1; then
      # port open; give it a few more seconds for ready
      sleep 2
      echo "==> port 19530 open (${i}s)"
      return 0
    fi
    sleep 2
  done
  echo "ERROR: milvus not ready; see ${MILVUS_LOG}" >&2
  if grep -q "defined more than once" "${MILVUS_LOG}" 2>/dev/null; then
    echo "" >&2
    echo "GFLAGS DUPLICATE: two Conan gflags copies registered DEFINE_string(help)." >&2
    echo "  Run: bash ${REPO_ROOT}/scripts/prepare_milvus_gflags.sh" >&2
    echo "  Then try: unset LD_PRELOAD; export LD_PRELOAD=${SHIM_DIR}/libgflags_gflagsns.so" >&2
    echo "  If still failing, both copies are static — rebuild Knowhere with shared gflags" >&2
    echo "  matching Milvus Conan package_id (see prepare script output)." >&2
  fi
  tail -80 "${MILVUS_LOG}" >&2 || true
  return 1
}

if [ "${SKIP_START:-0}" != "1" ]; then
  if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "==> milvus already running pid=$(cat "${PID_FILE}")"
  else
    echo "==> starting milvus standalone (log: ${MILVUS_LOG})"
    cd "${MILVUS_DIR}"
    set +e
    nohup env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" LD_PRELOAD="${LD_PRELOAD:-}" \
      "${_milvus_bin}" run standalone >"${MILVUS_LOG}" 2>&1 &
    echo $! >"${PID_FILE}"
    _pid=$(cat "${PID_FILE}")
    echo "    pid=${_pid}"
    sleep 2
    if ! kill -0 "${_pid}" 2>/dev/null; then
      echo "ERROR: milvus exited immediately; log:" >&2
      cat "${MILVUS_LOG}" >&2 || true
      if grep -q "defined more than once" "${MILVUS_LOG}" 2>/dev/null; then
        echo "" >&2
        echo "This is the duplicate-gflags abort. Diagnose:" >&2
        echo "  bash ${REPO_ROOT}/scripts/prepare_milvus_gflags.sh" >&2
      fi
      set -e
      exit 1
    fi
    set -e
  fi
  wait_ready
else
  echo "==> SKIP_START=1; assuming milvus at ${URI}"
fi

echo "==> GPU_IVF_FLAT smoke (SIFT slice)"
cd "${REPO_ROOT}"
python3 scripts/run_milvus_hdf5.py \
  --uri "${URI}" \
  --index-type GPU_IVF_FLAT \
  --nlist 128 \
  --nprobes 8,16 \
  --max-train-rows 50000 \
  --max-query-rows 500 \
  --data "${DATA_PATH}" \
  --collection sift_gpu_ivf_smoke

echo ""
echo "SMOKE OK"
echo "  log: ${MILVUS_LOG}"
echo "  Confirm HIP/Knowhere activity: grep -iE 'hip|cuvs|gpu_ivf|knowhere' ${MILVUS_LOG} | tail"
echo "  Stop: kill \$(cat ${PID_FILE})"
