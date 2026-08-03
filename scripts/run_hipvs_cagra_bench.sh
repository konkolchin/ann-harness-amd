#!/usr/bin/env bash
# Library-level hipVS CAGRA microbench on AMD ROCm (gfx1100 consumer challenge).
#
# Prerequisites: hipVS Python (import cuvs.neighbors.cagra), CuPy ROCm, SIFT HDF5.
# See docs/cagra_consumer_followon.md
#
# Usage:
#   source ~/hipvs-bench-venv/bin/activate
#   bash scripts/run_hipvs_cagra_bench.sh
#   MAX_TRAIN_ROWS=100000 ITOPK_SIZES=64,128 bash scripts/run_hipvs_cagra_bench.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
GRAPH_DEGREE="${GRAPH_DEGREE:-32}"
INTERMEDIATE_GRAPH_DEGREE="${INTERMEDIATE_GRAPH_DEGREE:-64}"
ITOPK_SIZES="${ITOPK_SIZES:-32,64,128,256}"
SEARCH_WIDTH="${SEARCH_WIDTH:-1}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/sift-128-euclidean.hdf5}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_JSON="${RESULTS_JSON:-${LOG_DIR}/lib_hipvs_cagra_${TS}.json}"
P99_SAMPLE="${P99_SAMPLE:-0}"

export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export ROCM_HOME
export LD_LIBRARY_PATH="${WORKDIR}/install/lib:${ROCM_HOME}/lib:${LD_LIBRARY_PATH:-}"

if ! python3 -c "from cuvs.neighbors import cagra" 2>/dev/null; then
  echo "ERROR: Python cannot import cuvs.neighbors.cagra (hipVS Python)." >&2
  echo "  See docs/cagra_consumer_followon.md §Phase A lib bench" >&2
  exit 1
fi

if [ ! -f "${DATA_PATH}" ]; then
  echo "ERROR: missing ${DATA_PATH}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
echo "==> hipVS CAGRA library bench"
echo "    graph_degree=${GRAPH_DEGREE} intermediate=${INTERMEDIATE_GRAPH_DEGREE}"
echo "    itopk=${ITOPK_SIZES} results=${RESULTS_JSON}"

cd "${REPO_ROOT}"
EXTRA=()
if [ -n "${MAX_TRAIN_ROWS:-}" ] && [ "${MAX_TRAIN_ROWS}" != "0" ]; then
  EXTRA+=(--max-train-rows "${MAX_TRAIN_ROWS}")
fi
if [ -n "${MAX_QUERY_ROWS:-}" ] && [ "${MAX_QUERY_ROWS}" != "0" ]; then
  EXTRA+=(--max-query-rows "${MAX_QUERY_ROWS}")
fi

python3 scripts/bench_cuvs_cagra.py \
  --backend hipvs \
  --graph-degree "${GRAPH_DEGREE}" \
  --intermediate-graph-degree "${INTERMEDIATE_GRAPH_DEGREE}" \
  --itopk-sizes "${ITOPK_SIZES}" \
  --search-width "${SEARCH_WIDTH}" \
  --p99-sample "${P99_SAMPLE}" \
  --data "${DATA_PATH}" \
  --results-json "${RESULTS_JSON}" \
  "${EXTRA[@]}"

echo ""
echo "HIPVS CAGRA LIB BENCH OK"
echo "  results: ${RESULTS_JSON}"
echo "  Compare: python3 scripts/compare_cuvs_lib_json.py --hipvs ${RESULTS_JSON} --cuvs <cuvs_json>"
