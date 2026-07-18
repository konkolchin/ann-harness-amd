#!/usr/bin/env bash
# Fair *batched* CPU baseline for management Table B.
# Same client as Layer-4 GPU: scripts/run_milvus_hdf5.py
#   → one search() call with all 10k SIFT queries (not VectorDBBench serial).
#
# Prerequisites:
#   - Docker CPU Milvus healthy on :19530 (stop HIP first)
#   - data/sift-128-euclidean.hdf5 under REPO or DATA=
#
# Usage:
#   bash scripts/run_milvus_batch_cpu_compare.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URI="${MILVUS_URI:-http://127.0.0.1:19530}"
DATA="${DATA:-${REPO_ROOT}/data/sift-128-euclidean.hdf5}"
NLIST="${NLIST:-1024}"
NPROBES="${NPROBES:-1,4,8,16,32}"
K="${K:-10}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
TS="$(date -u +%Y%m%d_%H%M%S)"
RESULTS_JSON="${RESULTS_JSON:-${LOG_DIR}/batch_cpu_ivf_${TS}.json}"
COLLECTION="${COLLECTION:-sift_batch_cpu_${TS}}"

mkdir -p "${LOG_DIR}"

if [ ! -f "${DATA}" ]; then
  echo "ERROR: missing ${DATA}" >&2
  exit 1
fi

if ! curl -sf "http://127.0.0.1:9091/healthz" >/dev/null 2>&1 \
  && ! (echo >/dev/tcp/127.0.0.1/19530) >/dev/null 2>&1; then
  echo "ERROR: Milvus not on :19530 — start Docker CPU stack first" >&2
  exit 1
fi

echo "==> Batched CPU IVF_FLAT (same client as Layer-4 GPU harness)"
echo "==> URI=${URI} nlist=${NLIST} nprobes=${NPROBES}"
echo "==> results: ${RESULTS_JSON}"

cd "${REPO_ROOT}"
python3 scripts/run_milvus_hdf5.py \
  --uri "${URI}" \
  --collection "${COLLECTION}" \
  --data "${DATA}" \
  --index-type IVF_FLAT \
  --nlist "${NLIST}" \
  --nprobes "${NPROBES}" \
  --k "${K}" \
  --flush \
  --results-json "${RESULTS_JSON}"

echo ""
echo "BATCH CPU OK"
echo "  json: ${RESULTS_JSON}"
echo "Compare to Layer-4 GPU JSON under \$WORKDIR/logs/layer4_gpu_ivf_*.json"
echo "  Speed-up = GPU_QPS / CPU_QPS at same nprobe (both batched 10k/search)."
