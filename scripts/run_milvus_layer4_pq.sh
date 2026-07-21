#!/usr/bin/env bash
# Full SIFT-1M sealed GPU_IVF_PQ nprobe sweep (same protocol as Layer-4 FLAT).
#
# Keep nlist/m/nbits identical on AMD HIP today and NVIDIA CUDA tomorrow.
# Default: nlist=1024, m=16, nbits=8, nprobe=1,4,8,16,32
#
# Usage (HIP already on :19530):
#   bash scripts/run_milvus_layer4_pq.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
URI="${MILVUS_URI:-http://127.0.0.1:19530}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/sift-128-euclidean.hdf5}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
MILVUS_LOG="${MILVUS_LOG:-${LOG_DIR}/milvus_gpu_standalone.log}"
TS="$(date +%Y%m%d_%H%M%S)"
COLLECTION="${L4_COLLECTION:-sift_gpu_l4_pq_${TS}}"
RESULTS_JSON="${RESULTS_JSON:-${LOG_DIR}/layer4_gpu_ivf_pq_${TS}.json}"
NLIST="${NLIST:-1024}"
NPROBES="${NPROBES:-1,4,8,16,32}"
M="${M:-16}"
NBITS="${NBITS:-8}"
INDEX_WAIT_S="${INDEX_WAIT_S:-3600}"

export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

if ! curl -sf "http://127.0.0.1:9091/healthz" >/dev/null 2>&1 \
  && ! (echo >/dev/tcp/127.0.0.1/19530) >/dev/null 2>&1; then
  echo "ERROR: Milvus not reachable — start HIP milvus first" >&2
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
  if grep -a -q 'InvalidDeviceFunction' "${MILVUS_LOG}"; then
    echo "WARNING: InvalidDeviceFunction in ${MILVUS_LOG}" >&2
  fi
  if grep -a -q 'GPU_CUVS_IVF_PQ' "${MILVUS_LOG}"; then
    echo "OK: GPU_CUVS_IVF_PQ activity present in log"
  else
    echo "WARNING: no GPU_CUVS_IVF_PQ lines — check log path / sealed path" >&2
  fi
fi

echo ""
echo "LAYER4 PQ RUN OK"
echo "  results: ${RESULTS_JSON}"
echo "  Tomorrow on 4080: same NLIST/M/NBITS/NPROBES with CUDA Milvus GPU_IVF_PQ."
