#!/usr/bin/env bash
# Sealed GPU_CAGRA smoke on HIP Milvus (Phase B — after Phase A Catch2/lib green).
#
# Prerequisite: HIP Milvus healthy on :19530 (same as IVF PQ smoke).
#
# Usage:
#   bash scripts/run_milvus_gpu_cagra_smoke.sh
#   MAX_TRAIN_ROWS=20000 ITOPK_SIZES=64,128 bash scripts/run_milvus_gpu_cagra_smoke.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URI="${MILVUS_URI:-http://127.0.0.1:19530}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/sift-128-euclidean.hdf5}"
WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
MILVUS_LOG="${MILVUS_LOG:-${LOG_DIR}/milvus_gpu_standalone.log}"
COLLECTION="${SMOKE_COLLECTION:-sift_gpu_cagra_smoke}"
GRAPH_DEGREE="${GRAPH_DEGREE:-32}"
INTERMEDIATE_GRAPH_DEGREE="${INTERMEDIATE_GRAPH_DEGREE:-64}"
ITOPK_SIZES="${ITOPK_SIZES:-64,128}"
SEARCH_WIDTH="${SEARCH_WIDTH:-1}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-20000}"
MAX_QUERY_ROWS="${MAX_QUERY_ROWS:-200}"
INDEX_WAIT_S="${INDEX_WAIT_S:-600}"

if ! curl -sf "http://127.0.0.1:9091/healthz" >/dev/null 2>&1 \
  && ! (echo >/dev/tcp/127.0.0.1/19530) >/dev/null 2>&1; then
  echo "ERROR: Milvus not reachable on :19530 / :9091" >&2
  echo "  Start HIP milvus first (e.g. SKIP_START=0 bash scripts/run_milvus_gpu_smoke.sh)." >&2
  exit 1
fi

echo "==> GPU_CAGRA smoke graph_degree=${GRAPH_DEGREE} itopk=${ITOPK_SIZES}"
echo "    collection=${COLLECTION} train=${MAX_TRAIN_ROWS} query=${MAX_QUERY_ROWS}"
cd "${REPO_ROOT}"
python3 scripts/run_milvus_hdf5.py \
  --uri "${URI}" \
  --index-type GPU_CAGRA \
  --flush \
  --index-wait-s "${INDEX_WAIT_S}" \
  --graph-degree "${GRAPH_DEGREE}" \
  --intermediate-graph-degree "${INTERMEDIATE_GRAPH_DEGREE}" \
  --itopk-sizes "${ITOPK_SIZES}" \
  --search-width "${SEARCH_WIDTH}" \
  --max-train-rows "${MAX_TRAIN_ROWS}" \
  --max-query-rows "${MAX_QUERY_ROWS}" \
  --data "${DATA_PATH}" \
  --collection "${COLLECTION}"

echo ""
echo "CAGRA SMOKE OK (client finished)"
echo "  Confirm sealed HIP path (want GPU_CUVS_CAGRA / CAGRA, not CPU fallback):"
echo "    grep -a -iE 'InvalidDeviceFunction|GPU_CUVS_CAGRA|CAGRA|origin_index|DeserializeFromStream' ${MILVUS_LOG} | tail -40"
