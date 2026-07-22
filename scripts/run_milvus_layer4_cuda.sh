#!/usr/bin/env bash
# Full SIFT-1M sealed GPU_IVF_FLAT on CUDA Milvus (RTX 4080 peer to AMD Layer-4).
#
# Prerequisite: CUDA GPU Milvus healthy on :19530
#   bash scripts/start_milvus_cuda_gpu_docker.sh
#
# Usage:
#   export WORKDIR=~/milvus_cuda_4080
#   bash scripts/run_milvus_layer4_cuda.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${HOME}/milvus_cuda_4080}"
URI="${MILVUS_URI:-http://127.0.0.1:19530}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/sift-128-euclidean.hdf5}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
TS="$(date +%Y%m%d_%H%M%S)"
COLLECTION="${L4_COLLECTION:-sift_cuda_l4_flat_${TS}}"
RESULTS_JSON="${RESULTS_JSON:-${LOG_DIR}/layer4_cuda_gpu_ivf_flat_${TS}.json}"
NLIST="${NLIST:-1024}"
NPROBES="${NPROBES:-1,4,8,16,32}"
INDEX_WAIT_S="${INDEX_WAIT_S:-3600}"

if ! curl -sf "http://127.0.0.1:9091/healthz" >/dev/null 2>&1 \
  && ! (echo >/dev/tcp/127.0.0.1/19530) >/dev/null 2>&1; then
  echo "ERROR: Milvus not reachable — start CUDA GPU stack first:" >&2
  echo "  bash scripts/start_milvus_cuda_gpu_docker.sh" >&2
  exit 1
fi

if [ ! -f "${DATA_PATH}" ]; then
  echo "ERROR: missing ${DATA_PATH}" >&2
  echo "  wget -c https://ann-benchmarks.com/sift-128-euclidean.hdf5 -O ${DATA_PATH}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
echo "==> Layer-4 CUDA GPU_IVF_FLAT full SIFT-1M"
echo "    nlist=${NLIST} nprobes=${NPROBES}"
echo "    collection=${COLLECTION}"
echo "    results=${RESULTS_JSON}"

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
echo "LAYER4 CUDA FLAT OK"
echo "  results: ${RESULTS_JSON}"
echo "  log hint: docker logs milvus-standalone 2>&1 | grep -aE 'GPU_CUVS_IVF_FLAT' | tail -20"
echo "  Compare to AMD HIP FLAT Layer-4 JSON / slides table."
