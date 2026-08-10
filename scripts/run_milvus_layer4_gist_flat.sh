#!/usr/bin/env bash
# GIST-1M + sealed GPU_IVF_FLAT (same client protocol as GIST PQ).
#
# Heavier than PQ (full float scan in probed lists). Expect higher recall and
# lower QPS than GIST PQ. Library FLAT-heavy favored AMD ~1.3x --- Milvus ratio
# may not show NVIDIA ahead; still a fair GPU-heavy product compare.
#
# Memory: ~1M x 960 x 4B ≈ 3.8 GB vectors (+ index). Fine on 7900 24GB;
# watch OOM on 4080 16GB if other GPU users share the card.
#
# Usage (Milvus on :19530):
#   bash scripts/run_milvus_layer4_gist_flat.sh
#   WORKDIR=~/milvus_cuda_4080 bash scripts/run_milvus_layer4_gist_flat.sh
#   SEARCH_WARMUP=5 SEARCH_RUNS=10 bash scripts/run_milvus_layer4_gist_flat.sh
#   SMOKE=1 bash scripts/run_milvus_layer4_gist_flat.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INDEX_WAIT_S="${INDEX_WAIT_S:-7200}"
NLIST="${NLIST:-1024}"
NPROBES="${NPROBES:-8,16,32,64,128}"
SMOKE="${SMOKE:-0}"
SEARCH_WARMUP="${SEARCH_WARMUP:-0}"
SEARCH_RUNS="${SEARCH_RUNS:-1}"
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
  echo "  wget -c https://ann-benchmarks.com/gist-960-euclidean.hdf5 -O ${DATA_PATH}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

EXTRA=()
if [ "${SMOKE}" = "1" ]; then
  NLIST="${NLIST_SMOKE:-128}"
  NPROBES="${NPROBES_SMOKE:-8,16}"
  MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-50000}"
  MAX_QUERY_ROWS="${MAX_QUERY_ROWS:-500}"
  COLLECTION="${L4_COLLECTION:-gist_gpu_flat_smoke_${TS}}"
  RESULTS_JSON="${RESULTS_JSON:-${LOG_DIR}/smoke_gist_gpu_ivf_flat_${TS}.json}"
  EXTRA+=(--max-train-rows "${MAX_TRAIN_ROWS}" --max-query-rows "${MAX_QUERY_ROWS}")
  INDEX_WAIT_S="${INDEX_WAIT_S_SMOKE:-600}"
  echo "==> SMOKE GIST GPU_IVF_FLAT (slice; recall vs full GT meaningless)"
else
  COLLECTION="${L4_COLLECTION:-gist_gpu_l4_flat_${TS}}"
  RESULTS_JSON="${RESULTS_JSON:-${LOG_DIR}/layer4_gist_gpu_ivf_flat_${TS}.json}"
  echo "==> Layer-4 GIST-1M GPU_IVF_FLAT (GPU-heavy HIP↔CUDA)"
fi

echo "    data=${DATA_PATH}"
echo "    nlist=${NLIST} nprobes=${NPROBES} insert_batch=${INSERT_BATCH}"
echo "    search_warmup=${SEARCH_WARMUP} search_runs=${SEARCH_RUNS} (median QPS)"
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
  --insert-batch "${INSERT_BATCH}" \
  --search-warmup "${SEARCH_WARMUP}" \
  --search-runs "${SEARCH_RUNS}" \
  --data "${DATA_PATH}" \
  --collection "${COLLECTION}" \
  --results-json "${RESULTS_JSON}" \
  "${EXTRA[@]}"

echo ""
if [ -f "${MILVUS_LOG}" ]; then
  grep -a -iE 'InvalidDeviceFunction|GPU_CUVS_IVF_FLAT|DeserializeFromStream' \
    "${MILVUS_LOG}" | tail -40 || true
elif docker ps --format '{{.Names}}' 2>/dev/null | grep -qx milvus-standalone; then
  echo "==> docker log check (GPU_CUVS_IVF_FLAT):"
  docker logs milvus-standalone 2>&1 | grep -aE 'GPU_CUVS_IVF_FLAT' | tail -20 || true
fi

echo ""
echo "GIST FLAT RUN OK"
echo "  results: ${RESULTS_JSON}"
echo "  Compare: python3 scripts/compare_milvus_layer4_json.py --amd AMD.json --cuda CUDA.json"
