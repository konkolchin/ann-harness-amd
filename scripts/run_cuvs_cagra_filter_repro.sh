#!/usr/bin/env bash
# CUDA peer for scripts/run_hipvs_cagra_filter_repro.sh
#
# Usage on kvkol CUDA box:
#   source ~/cuvs-bench-venv/bin/activate
#   export WORKDIR=~/milvus_cuda_4080
#   bash scripts/run_cuvs_cagra_filter_repro.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${HOME}/milvus_cuda_4080}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_JSON="${RESULTS_JSON:-${LOG_DIR}/lib_cuvs_cagra_filter_${TS}.json}"
LOG="${LOG_DIR}/lib_cuvs_cagra_filter_${TS}.log"
GRAPH_BUILD_ALGO="${GRAPH_BUILD_ALGO:-nn_descent}"
ITOPK_SIZE="${ITOPK_SIZE:-128}"

mkdir -p "${LOG_DIR}"

if ! python3 -c "from cuvs.neighbors import cagra" 2>/dev/null; then
  echo "ERROR: cannot import cuvs.neighbors.cagra — activate cuvs-bench-venv" >&2
  exit 1
fi

echo "==> cuVS CAGRA filter repro (CUDA peer)"
cd "${REPO_ROOT}"
set +e
python3 scripts/probe_cuvs_cagra_filter.py \
  --build-algo "${GRAPH_BUILD_ALGO}" \
  --itopk-size "${ITOPK_SIZE}" \
  --n-train "${N_TRAIN:-10000}" \
  --n-query "${N_QUERY:-200}" \
  --k "${K:-1}" \
  --results-json "${RESULTS_JSON}" \
  2>&1 | tee "${LOG}"
rc=${PIPESTATUS[0]}
set -e
grep -n 'OWNER\|^\[' "${LOG}" | head -40 || true
echo "DONE: ${LOG}"
exit "${rc}"
