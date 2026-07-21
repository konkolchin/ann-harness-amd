#!/usr/bin/env bash
# Sealed GPU_IVF_PQ smoke on HIP Milvus (AMD today; keep params for CUDA 4080 tomorrow).
#
# Prerequisite: HIP Milvus already healthy on :19530 (same as Layer-4).
#   export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
#   # start milvus as usual, or: SKIP_START=0 bash scripts/run_milvus_gpu_smoke.sh
#
# Usage:
#   bash scripts/run_milvus_gpu_ivf_pq_smoke.sh
#
# Recipe (SIFT-128): nlist=128, m=16, nbits=8  (m must divide 128)
# Expect recall@10 below FLAT at the same nprobe — that is normal for PQ.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URI="${MILVUS_URI:-http://127.0.0.1:19530}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/sift-128-euclidean.hdf5}"
WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
MILVUS_LOG="${MILVUS_LOG:-${LOG_DIR}/milvus_gpu_standalone.log}"
COLLECTION="${SMOKE_COLLECTION:-sift_gpu_ivf_pq_smoke}"
NLIST="${NLIST:-128}"
NPROBES="${NPROBES:-8,16}"
M="${M:-16}"
NBITS="${NBITS:-8}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-50000}"
MAX_QUERY_ROWS="${MAX_QUERY_ROWS:-500}"
INDEX_WAIT_S="${INDEX_WAIT_S:-300}"

if ! curl -sf "http://127.0.0.1:9091/healthz" >/dev/null 2>&1 \
  && ! (echo >/dev/tcp/127.0.0.1/19530) >/dev/null 2>&1; then
  echo "ERROR: Milvus not reachable on :19530 / :9091" >&2
  echo "  Start HIP milvus first, then re-run this script." >&2
  exit 1
fi

echo "==> GPU_IVF_PQ smoke nlist=${NLIST} m=${M} nbits=${NBITS} nprobes=${NPROBES}"
echo "    collection=${COLLECTION}"
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
  --max-train-rows "${MAX_TRAIN_ROWS}" \
  --max-query-rows "${MAX_QUERY_ROWS}" \
  --data "${DATA_PATH}" \
  --collection "${COLLECTION}"

echo ""
echo "SMOKE PQ client OK"
echo "  Log check (want GPU_CUVS_IVF_PQ; watch InvalidDeviceFunction):"
echo "    grep -a -iE 'InvalidDeviceFunction|GPU_CUVS_IVF_PQ|IVF_PQ' ${MILVUS_LOG} | tail -40"
