#!/usr/bin/env bash
# Library-level NVIDIA cuVS CAGRA microbench (CUDA peer baseline for gfx1100).
#
# Usage:
#   bash scripts/run_cuvs_cagra_bench.sh
#   WORKDIR=~/milvus_cuda_4080 bash scripts/run_cuvs_cagra_bench.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${HOME}/milvus_cuda_4080}"
GRAPH_DEGREE="${GRAPH_DEGREE:-32}"
INTERMEDIATE_GRAPH_DEGREE="${INTERMEDIATE_GRAPH_DEGREE:-64}"
ITOPK_SIZES="${ITOPK_SIZES:-32,64,128,256}"
SEARCH_WIDTH="${SEARCH_WIDTH:-1}"
GRAPH_BUILD_ALGO="${GRAPH_BUILD_ALGO:-nn_descent}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/sift-128-euclidean.hdf5}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_JSON="${RESULTS_JSON:-${LOG_DIR}/lib_cuvs_cagra_${TS}.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/cuvs_pip_ld_path.sh"

if [ ! -f "${DATA_PATH}" ]; then
  echo "ERROR: missing ${DATA_PATH}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
echo "==> cuVS CAGRA library bench"
echo "    graph_build_algo=${GRAPH_BUILD_ALGO} results=${RESULTS_JSON}"

cd "${REPO_ROOT}"
EXTRA=()
if [ -n "${MAX_TRAIN_ROWS:-}" ] && [ "${MAX_TRAIN_ROWS}" != "0" ]; then
  EXTRA+=(--max-train-rows "${MAX_TRAIN_ROWS}")
fi
if [ -n "${MAX_QUERY_ROWS:-}" ] && [ "${MAX_QUERY_ROWS}" != "0" ]; then
  EXTRA+=(--max-query-rows "${MAX_QUERY_ROWS}")
fi
if [ -n "${GRAPH_BUILD_ALGO}" ]; then
  EXTRA+=(--graph-build-algo "${GRAPH_BUILD_ALGO}")
fi

python3 scripts/bench_cuvs_cagra.py \
  --backend cuvs \
  --graph-degree "${GRAPH_DEGREE}" \
  --intermediate-graph-degree "${INTERMEDIATE_GRAPH_DEGREE}" \
  --itopk-sizes "${ITOPK_SIZES}" \
  --search-width "${SEARCH_WIDTH}" \
  --data "${DATA_PATH}" \
  --results-json "${RESULTS_JSON}" \
  "${EXTRA[@]}"

echo ""
echo "CUVS CAGRA LIB BENCH OK"
echo "  results: ${RESULTS_JSON}"
