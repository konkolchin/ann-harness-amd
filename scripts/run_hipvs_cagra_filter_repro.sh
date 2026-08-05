#!/usr/bin/env bash
# Minimal hipVS CAGRA filtered-search repro on AMD (no Knowhere).
#
# Decides ownership for Catch2 bitset recall 0.0:
#   unfiltered OK + filter dead  => hipVS CAGRA filter
#   filter OK on hipVS            => Knowhere bitset wiring
#
# Usage on amd-rx7900xtx:
#   source ~/hipvs-bench-venv/bin/activate
#   cd ~/ann-harness-amd && git pull --ff-only
#   export WORKDIR=~/rocmds_check_gfx1100
#   bash scripts/run_hipvs_cagra_filter_repro.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_JSON="${RESULTS_JSON:-${LOG_DIR}/lib_hipvs_cagra_filter_${TS}.json}"
LOG="${LOG_DIR}/lib_hipvs_cagra_filter_${TS}.log"
GRAPH_BUILD_ALGO="${GRAPH_BUILD_ALGO:-nn_descent}"
ITOPK_SIZE="${ITOPK_SIZE:-128}"

export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export ROCM_HOME
export LD_LIBRARY_PATH="${WORKDIR}/install/lib:${ROCM_HOME}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${LOG_DIR}"

if ! python3 -c "from cuvs.neighbors import cagra" 2>/dev/null; then
  echo "ERROR: cannot import cuvs.neighbors.cagra — activate hipvs-bench-venv" >&2
  exit 1
fi

echo "==> hipVS CAGRA filter repro"
echo "    build_algo=${GRAPH_BUILD_ALGO} itopk=${ITOPK_SIZE}"
echo "    results=${RESULTS_JSON}"
echo "    log=${LOG}"

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

echo ""
echo "exit=${rc}"
echo "OWNERSHIP lines:"
grep -n 'OWNER\|^\[' "${LOG}" | head -40 || true
echo ""
echo "DONE: ${LOG}"
echo "JSON: ${RESULTS_JSON}"
exit "${rc}"
