#!/usr/bin/env bash
# Full SIFT-1M sealed GPU_IVF_PQ nprobe sweep (same protocol as Layer-4 FLAT).
#
# Works on AMD HIP or CUDA GPU Milvus — keep nlist/m/nbits identical both sides.
# Default: nlist=1024, m=16, nbits=8, nprobe=1,4,8,16,32
# For 4080 primary compare: M=32
#
# Usage (Milvus already on :19530):
#   bash scripts/run_milvus_layer4_pq.sh
#   M=32 WORKDIR=~/milvus_cuda_4080 bash scripts/run_milvus_layer4_pq.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
M="${M:-16}"
NBITS="${NBITS:-8}"
INDEX_WAIT_S="${INDEX_WAIT_S:-3600}"
NLIST="${NLIST:-1024}"
NPROBES="${NPROBES:-1,4,8,16,32}"

if [ -z "${WORKDIR:-}" ]; then
  if [ -d "${HOME}/milvus_cuda_4080" ]; then
    WORKDIR="${HOME}/milvus_cuda_4080"
  else
    WORKDIR="${HOME}/rocmds_check_gfx1100"
  fi
fi

URI="${MILVUS_URI:-http://127.0.0.1:19530}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/sift-128-euclidean.hdf5}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
MILVUS_LOG="${MILVUS_LOG:-${LOG_DIR}/milvus_gpu_standalone.log}"
TS="$(date +%Y%m%d_%H%M%S)"
COLLECTION="${L4_COLLECTION:-sift_gpu_l4_pq_${TS}}"
RESULTS_JSON="${RESULTS_JSON:-${LOG_DIR}/layer4_gpu_ivf_pq_${TS}.json}"

export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if ! curl -sf "http://127.0.0.1:9091/healthz" >/dev/null 2>&1 \
  && ! (echo >/dev/tcp/127.0.0.1/19530) >/dev/null 2>&1; then
  echo "ERROR: Milvus not reachable on :19530 / :9091" >&2
  echo "  HIP: start standalone; CUDA: bash scripts/start_milvus_cuda_gpu_docker.sh" >&2
  exit 1
fi

if [ ! -f "${DATA_PATH}" ]; then
  echo "ERROR: missing ${DATA_PATH}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
echo "==> Layer-4 GPU_IVF_PQ full SIFT-1M"
echo "    nlist=${NLIST} m=${M} nbits=${NBITS} nprobes=${NPROBES}"
echo "    collection=${COLLECTION}"
echo "    results=${RESULTS_JSON}"

cd "${REPO_ROOT}"
python3 scripts/run_milvus_hdf5.py \
  --uri "${URI}" \
  --index-type GPU_IVF_PQ \
  --flush \
  --index-wait-s "${INDEX_WAIT_S}" \
  --nlist "${NLIST}" \
  --m "${M}" \
  --nbits "${NBITS}" \
  --nprobes "${NPROBES}" \
  --data "${DATA_PATH}" \
  --collection "${COLLECTION}" \
  --results-json "${RESULTS_JSON}"

echo ""
if [ -f "${MILVUS_LOG}" ]; then
  grep -a -iE 'InvalidDeviceFunction|GPU_CUVS_IVF_PQ|DeserializeFromStream' \
    "${MILVUS_LOG}" | tail -40 || true
elif docker ps --format '{{.Names}}' 2>/dev/null | grep -qx milvus-standalone; then
  echo "==> docker log check (GPU_CUVS_IVF_PQ):"
  docker logs milvus-standalone 2>&1 | grep -aE 'GPU_CUVS_IVF_PQ' | tail -20 || true
fi

echo ""
echo "LAYER4 PQ RUN OK"
echo "  results: ${RESULTS_JSON}"
