#!/usr/bin/env bash
# Layer-4: full SIFT-1M sealed GPU_CAGRA itopk sweep on HIP Milvus (Phase B).
#
# Prerequisites: HIP milvus on :19530; Phase A Catch2/lib CAGRA not recall-0.
#
# Usage:
#   SKIP_START=1 bash scripts/run_milvus_layer4_cagra.sh
#   DATA_PATH=data/gist-960-euclidean.hdf5 L4_COLLECTION=gist_cagra bash scripts/run_milvus_layer4_cagra.sh
#   SEARCH_WARMUP=5 SEARCH_RUNS=10 SKIP_START=1 bash scripts/run_milvus_layer4_cagra.sh
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
MILVUS_LOG="${MILVUS_LOG:-${LOG_DIR}/milvus_gpu_standalone.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/milvus_gpu_standalone.pid}"
TS="$(date +%Y%m%d_%H%M%S)"
COLLECTION="${L4_COLLECTION:-sift_gpu_cagra_l4_${TS}}"
RESULTS_JSON="${RESULTS_JSON:-${LOG_DIR}/layer4_gpu_cagra_${TS}.json}"
GRAPH_DEGREE="${GRAPH_DEGREE:-32}"
INTERMEDIATE_GRAPH_DEGREE="${INTERMEDIATE_GRAPH_DEGREE:-64}"
# gfx1100: hipVS IVF_PQ CAGRA graph build fails; use NN_DESCENT (unfiltered only).
BUILD_ALGO="${BUILD_ALGO:-NN_DESCENT}"
ITOPK_SIZES="${ITOPK_SIZES:-32,64,128,256}"
SEARCH_WIDTH="${SEARCH_WIDTH:-1}"
INDEX_WAIT_S="${INDEX_WAIT_S:-7200}"
# Full-batch search: discard warm-up, report median of timed runs (default = 1-shot).
SEARCH_WARMUP="${SEARCH_WARMUP:-0}"
SEARCH_RUNS="${SEARCH_RUNS:-1}"

export MILVUS_HIP_INSTALL_PREFIX="${MILVUS_HIP_INSTALL_PREFIX:-${INSTALL_PREFIX}}"
export ROCM_PATH
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export PATH="${ROCM_PATH}/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${MILVUS_HIP_INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:${MILVUS_DIR}/internal/core/output/lib:${LD_LIBRARY_PATH:-}"
if [ -d "${SHIM_DIR}" ]; then
  export LD_LIBRARY_PATH="${SHIM_DIR}:${LD_LIBRARY_PATH}"
fi

mkdir -p "${LOG_DIR}"

if [ ! -f "${DATA_PATH}" ]; then
  echo "ERROR: dataset not found: ${DATA_PATH}" >&2
  exit 1
fi

_milvus_bin=""
for c in \
  "${MILVUS_DIR}/bin/milvus" \
  "${MILVUS_DIR}/internal/core/output/bin/milvus"
do
  if [ -x "$c" ]; then _milvus_bin="$c"; break; fi
done
if [ -z "${_milvus_bin}" ]; then
  echo "ERROR: milvus binary not found under ${MILVUS_DIR}" >&2
  exit 1
fi
echo "==> milvus: ${_milvus_bin}"
echo "==> collection=${COLLECTION}"
echo "==> results_json=${RESULTS_JSON}"

if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'milvus-standalone'; then
  echo "==> stopping docker milvus-standalone (frees :19530)"
  docker stop milvus-standalone >/dev/null || true
fi

_compose="${MILVUS_DIR}/deployments/docker/dev/docker-compose.yml"
if [ ! -f "${_compose}" ]; then
  _compose="${MILVUS_DIR}/docker-compose.yml"
fi
if [ "${SKIP_DEPS:-0}" != "1" ] && [ -f "${_compose}" ] && command -v docker >/dev/null; then
  echo "==> ensuring etcd/minio from ${_compose}"
  (cd "$(dirname "${_compose}")" && docker compose -f "$(basename "${_compose}")" up -d etcd minio 2>/dev/null) \
    || (cd "$(dirname "${_compose}")" && docker-compose -f "$(basename "${_compose}")" up -d etcd minio 2>/dev/null) \
    || echo "NOTE: could not start etcd/minio via compose" >&2
fi

wait_ready() {
  local i
  for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:9091/healthz" >/dev/null 2>&1; then
      echo "==> milvus healthy (${i}s)"
      return 0
    fi
    if (echo >/dev/tcp/127.0.0.1/19530) >/dev/null 2>&1; then
      sleep 2
      echo "==> port 19530 open (${i}s)"
      return 0
    fi
    sleep 2
  done
  echo "ERROR: milvus not ready; see ${MILVUS_LOG}" >&2
  tail -80 "${MILVUS_LOG}" >&2 || true
  return 1
}

if [ "${SKIP_START:-1}" != "1" ]; then
  if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "==> milvus already running pid=$(cat "${PID_FILE}")"
  else
    echo "==> starting milvus standalone (log: ${MILVUS_LOG})"
    cd "${MILVUS_DIR}"
    set +e
    nohup env \
      ROCM_PATH="${ROCM_PATH}" \
      ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}" \
      HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}" \
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
      LD_PRELOAD="${LD_PRELOAD:-}" \
      "${_milvus_bin}" run standalone >"${MILVUS_LOG}" 2>&1 &
    echo $! >"${PID_FILE}"
    sleep 2
    if ! kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
      echo "ERROR: milvus exited immediately; log:" >&2
      cat "${MILVUS_LOG}" >&2 || true
      set -e
      exit 1
    fi
    set -e
  fi
  wait_ready
else
  echo "==> SKIP_START=1 (default); assuming milvus at ${URI}"
  if ! curl -sf "http://127.0.0.1:9091/healthz" >/dev/null 2>&1 \
    && ! (echo >/dev/tcp/127.0.0.1/19530) >/dev/null 2>&1; then
    echo "ERROR: nothing listening on :19530 / :9091" >&2
    exit 1
  fi
fi

echo "==> Layer-4 GPU_CAGRA (unfiltered)"
echo "    graph_degree=${GRAPH_DEGREE} intermediate=${INTERMEDIATE_GRAPH_DEGREE}"
echo "    build_algo=${BUILD_ALGO} itopk=${ITOPK_SIZES} search_width=${SEARCH_WIDTH}"
echo "    search_warmup=${SEARCH_WARMUP} search_runs=${SEARCH_RUNS} (median QPS)"
echo "    data=${DATA_PATH}"
cd "${REPO_ROOT}"
python3 scripts/run_milvus_hdf5.py \
  --uri "${URI}" \
  --index-type GPU_CAGRA \
  --flush \
  --index-wait-s "${INDEX_WAIT_S}" \
  --graph-degree "${GRAPH_DEGREE}" \
  --intermediate-graph-degree "${INTERMEDIATE_GRAPH_DEGREE}" \
  --build-algo "${BUILD_ALGO}" \
  --itopk-sizes "${ITOPK_SIZES}" \
  --search-width "${SEARCH_WIDTH}" \
  --search-warmup "${SEARCH_WARMUP}" \
  --search-runs "${SEARCH_RUNS}" \
  --data "${DATA_PATH}" \
  --collection "${COLLECTION}" \
  --results-json "${RESULTS_JSON}"

echo ""
echo "==> HIP sealed-path check"
if [ -f "${MILVUS_LOG}" ]; then
  grep -a -iE 'InvalidDeviceFunction|GPU_CUVS_CAGRA|CAGRA|DeserializeFromStream|origin_index' \
    "${MILVUS_LOG}" | tail -40 || true
  if grep -a -qi 'GPU_CUVS_CAGRA\|CAGRA' "${MILVUS_LOG}"; then
    echo "OK: CAGRA-related activity present in log"
  else
    echo "WARNING: no CAGRA lines in ${MILVUS_LOG}" >&2
  fi
fi

echo ""
echo "LAYER4 CAGRA RUN OK"
echo "  results: ${RESULTS_JSON}"
echo "  Fair peer: same recipe on RTX 4080 CUDA Milvus; compare recall@10 + QPS."
