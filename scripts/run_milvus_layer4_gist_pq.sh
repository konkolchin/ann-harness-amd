#!/usr/bin/env bash
# Fair GPU-heavy Milvus HIP↔CUDA compare: GIST-1M + sealed GPU_IVF_PQ.
#
# Why GIST (not SIFT): 960-d raises GPU kernel fraction so a cuVS vs hipVS
# library gap can appear in *Milvus* QPS (SIFT Layer-4 is stack-dominated ~1×).
#
# Freeze (both GPUs identical):
#   dataset  gist-960-euclidean.hdf5
#   index    GPU_IVF_PQ  m=32  nbits=8  nlist=1024  --flush
#   search   k=10  nprobe=8,16,32,64,128
#
# Usage (Milvus already on :19530):
#   # AMD HIP
#   bash scripts/run_milvus_layer4_gist_pq.sh
#   # CUDA 4080
#   WORKDIR=~/milvus_cuda_4080 bash scripts/run_milvus_layer4_gist_pq.sh
#
# Smoke first (optional):
#   SMOKE=1 bash scripts/run_milvus_layer4_gist_pq.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
M="${M:-32}"
NBITS="${NBITS:-8}"
INDEX_WAIT_S="${INDEX_WAIT_S:-7200}"
NLIST="${NLIST:-1024}"
NPROBES="${NPROBES:-8,16,32,64,128}"
SMOKE="${SMOKE:-0}"
# GIST-960: 50k * 960 * 4B ≈ 192MB > gRPC 64MB default → RESOURCE_EXHAUSTED
INSERT_BATCH="${INSERT_BATCH:-8000}"

if [ -z "${WORKDIR:-}" ]; then
  if [ -d "${HOME}/milvus_cuda_4080" ]; then
    WORKDIR="${HOME}/milvus_cuda_4080"
  else
    WORKDIR="${HOME}/rocmds_check_gfx1100"
  fi
fi

URI="${MILVUS_URI:-http://127.0.0.1:19530}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/gist-960-euclidean.hdf5}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
MILVUS_LOG="${MILVUS_LOG:-${LOG_DIR}/milvus_gpu_standalone.log}"
TS="$(date +%Y%m%d_%H%M%S)"

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
  echo "  mkdir -p ${REPO_ROOT}/data && wget -c \\" >&2
  echo "    https://ann-benchmarks.com/gist-960-euclidean.hdf5 \\" >&2
  echo "    -O ${DATA_PATH}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

EXTRA=()
if [ "${SMOKE}" = "1" ]; then
  NLIST="${NLIST_SMOKE:-128}"
  NPROBES="${NPROBES_SMOKE:-8,16}"
  MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-50000}"
  MAX_QUERY_ROWS="${MAX_QUERY_ROWS:-500}"
  COLLECTION="${L4_COLLECTION:-gist_gpu_pq_smoke_${TS}}"
  RESULTS_JSON="${RESULTS_JSON:-${LOG_DIR}/smoke_gist_gpu_ivf_pq_${TS}.json}"
  EXTRA+=(--max-train-rows "${MAX_TRAIN_ROWS}" --max-query-rows "${MAX_QUERY_ROWS}")
  INDEX_WAIT_S="${INDEX_WAIT_S_SMOKE:-600}"
  echo "==> SMOKE GIST GPU_IVF_PQ (slice; recall vs full GT meaningless)"
else
  COLLECTION="${L4_COLLECTION:-gist_gpu_l4_pq_${TS}}"
  RESULTS_JSON="${RESULTS_JSON:-${LOG_DIR}/layer4_gist_gpu_ivf_pq_${TS}.json}"
  echo "==> Layer-4 GIST-1M GPU_IVF_PQ (GPU-heavy fair HIP↔CUDA)"
fi

echo "    data=${DATA_PATH}"
echo "    nlist=${NLIST} m=${M} nbits=${NBITS} nprobes=${NPROBES}"
echo "    insert_batch=${INSERT_BATCH}"
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
  --insert-batch "${INSERT_BATCH}" \
  --data "${DATA_PATH}" \
  --collection "${COLLECTION}" \
  --results-json "${RESULTS_JSON}" \
  "${EXTRA[@]}"

echo ""
if [ -f "${MILVUS_LOG}" ]; then
  grep -a -iE 'InvalidDeviceFunction|GPU_CUVS_IVF_PQ|DeserializeFromStream' \
    "${MILVUS_LOG}" | tail -40 || true
elif docker ps --format '{{.Names}}' 2>/dev/null | grep -qx milvus-standalone; then
  echo "==> docker log check (GPU_CUVS_IVF_PQ):"
  docker logs milvus-standalone 2>&1 | grep -aE 'GPU_CUVS_IVF_PQ' | tail -20 || true
fi

echo ""
echo "GIST PQ RUN OK"
echo "  results: ${RESULTS_JSON}"
echo "  Compare both GPUs:"
echo "    python3 scripts/compare_milvus_layer4_json.py --amd AMD.json --cuda CUDA.json"
