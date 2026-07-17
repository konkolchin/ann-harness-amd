#!/usr/bin/env bash
# Layer-4: full SIFT-1M sealed GPU_IVF_FLAT nprobe sweep on HIP Milvus.
#
# Prerequisites:
#   - HIP milvus on :19530 (or omit SKIP_START=1 and let this script start it)
#   - etcd + minio reachable
#   - data/sift-128-euclidean.hdf5
#   - ROCR_VISIBLE_DEVICES=0 recommended (hide iGPU)
#
# Usage:
#   bash scripts/run_milvus_layer4.sh
#   SKIP_START=1 bash scripts/run_milvus_layer4.sh
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
COLLECTION="${L4_COLLECTION:-sift_gpu_l4_${TS}}"
RESULTS_JSON="${RESULTS_JSON:-${LOG_DIR}/layer4_gpu_ivf_${TS}.json}"
NLIST="${NLIST:-1024}"
NPROBES="${NPROBES:-1,4,8,16,32}"
INDEX_WAIT_S="${INDEX_WAIT_S:-3600}"

export MILVUS_HIP_INSTALL_PREFIX="${MILVUS_HIP_INSTALL_PREFIX:-${INSTALL_PREFIX}}"
export ROCM_PATH
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="${ROCM_PATH}/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${MILVUS_HIP_INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:${MILVUS_DIR}/internal/core/output/lib:${LD_LIBRARY_PATH:-}"
if [ -d "${SHIM_DIR}" ]; then
  export LD_LIBRARY_PATH="${SHIM_DIR}:${LD_LIBRARY_PATH}"
fi

# Prefer no gflags shim for Layer-4 if both embeds are static (shim can break malloc).
# Caller may still set LD_PRELOAD explicitly.
if [ -z "${LD_PRELOAD:-}" ] && [ "${USE_GFLAGS_SHIM:-0}" = "1" ] \
  && [ -e "${SHIM_DIR}/libgflags_gflagsns.so" ]; then
  export LD_PRELOAD="${SHIM_DIR}/libgflags_gflagsns.so"
  echo "==> LD_PRELOAD=${LD_PRELOAD}"
fi

mkdir -p "${LOG_DIR}"

if [ ! -f "${DATA_PATH}" ]; then
  echo "ERROR: SIFT hdf5 not found: ${DATA_PATH}" >&2
  echo "  wget --no-proxy https://ann-benchmarks.com/sift-128-euclidean.hdf5 -O ${DATA_PATH}" >&2
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
  echo "  Build first: bash scripts/build_milvus_layer3.sh" >&2
  exit 1
fi
echo "==> milvus: ${_milvus_bin}"
echo "==> ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
echo "==> collection=${COLLECTION}"
echo "==> results_json=${RESULTS_JSON}"

# Stop stock Docker standalone if it holds :19530
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
    || echo "NOTE: could not start etcd/minio via compose; ensure they are already running" >&2
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
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
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
    echo "ERROR: nothing listening on :19530 / :9091 — start HIP milvus first" >&2
    echo "  or: SKIP_START=0 bash scripts/run_milvus_layer4.sh" >&2
    exit 1
  fi
fi

echo "==> Layer-4 full SIFT-1M GPU_IVF_FLAT (nlist=${NLIST}, nprobes=${NPROBES})"
echo "    Expect long insert/index; FAISS Table-5 recall ladder ~0.37→0.98 as nprobe rises."
cd "${REPO_ROOT}"
python3 scripts/run_milvus_hdf5.py \
  --uri "${URI}" \
  --index-type GPU_IVF_FLAT \
  --flush \
  --index-wait-s "${INDEX_WAIT_S}" \
  --nlist "${NLIST}" \
  --nprobes "${NPROBES}" \
  --data "${DATA_PATH}" \
  --collection "${COLLECTION}" \
  --results-json "${RESULTS_JSON}"

echo ""
echo "==> HIP sealed-path check (recent log lines)"
if [ -f "${MILVUS_LOG}" ]; then
  grep -a -iE 'InvalidDeviceFunction|IVF_FLAT_CC|GPU_CUVS_IVF_FLAT|DeserializeFromStream|origin_index' \
    "${MILVUS_LOG}" | tail -40 || true
  if grep -a -q 'InvalidDeviceFunction' "${MILVUS_LOG}"; then
    echo "WARNING: InvalidDeviceFunction seen in ${MILVUS_LOG}" >&2
  fi
  if grep -a -q 'GPU_CUVS_IVF_FLAT' "${MILVUS_LOG}"; then
    echo "OK: GPU_CUVS_IVF_FLAT activity present in log"
  else
    echo "WARNING: no GPU_CUVS_IVF_FLAT lines in ${MILVUS_LOG} (check log path / CGO output)" >&2
  fi
else
  echo "NOTE: milvus log not found at ${MILVUS_LOG}; check the process stdout/TTY"
fi

echo ""
echo "LAYER4 RUN OK"
echo "  results: ${RESULTS_JSON}"
echo "  log:     ${MILVUS_LOG}"
echo "  Compare recall/QPS to runbook Tables 5 (FAISS) and 6–7 (CPU Milvus) on amd-rx7900xtx."
echo "  Pass: recall@10 rises with nprobe; sealed GPU_CUVS load without InvalidDeviceFunction."
