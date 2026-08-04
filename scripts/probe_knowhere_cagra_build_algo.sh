#!/usr/bin/env bash
# Probe Knowhere sources + Catch2 for CAGRA build_algo (ivf_pq vs nn_descent).
#
# Usage on amd-rx7900xtx:
#   export WORKDIR=~/rocmds_check_gfx1100
#   bash scripts/probe_knowhere_cagra_build_algo.sh
#   RUN_CATCH2=1 bash scripts/probe_knowhere_cagra_build_algo.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
KH="${KNOWHERE_DIR:-${WORKDIR}/knowhere}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
TS="$(date +%Y%m%d_%H%M%S)"
PROBE_LOG="${LOG_DIR}/knowhere_cagra_probe_${TS}.log"

export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="${WORKDIR}/knowhere/build:${WORKDIR}/install/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${LOG_DIR}"
{
  echo "# Knowhere CAGRA build_algo probe"
  echo "date=$(date -Iseconds) host=$(hostname)"
  echo "KNOWHERE_DIR=${KH}"
  echo ""
} | tee "${PROBE_LOG}"

if [ ! -d "${KH}" ]; then
  echo "ERROR: knowhere tree not found at ${KH}" | tee -a "${PROBE_LOG}"
  exit 1
fi

echo "==> grep Knowhere for CAGRA / build_algo / nn_descent" | tee -a "${PROBE_LOG}"
set +e
rg -n -i \
  'build_algo|nn_descent|nn-descent|graph_build|IVF_PQ|cagra|CAGRA|build_algo' \
  "${KH}/src" "${KH}/include" "${KH}/tests" \
  --glob '*.{cc,cpp,h,hpp,cuh,cu}' 2>/dev/null \
  | tee -a "${PROBE_LOG}" | head -120
# fallback if rg missing
if ! command -v rg >/dev/null 2>&1; then
  grep -RInE 'build_algo|nn_descent|graph_build|CAGRA|cagra' \
    "${KH}/src" "${KH}/include" "${KH}/tests" 2>/dev/null \
    | tee -a "${PROBE_LOG}" | head -120
fi
set -e

echo "" | tee -a "${PROBE_LOG}"
echo "==> test_gpu_search.cc CAGRA cfg snippets" | tee -a "${PROBE_LOG}"
TG="${KH}/tests/ut/test_gpu_search.cc"
if [ -f "${TG}" ]; then
  grep -n -iE 'CAGRA|cagra|graph_degree|build_algo|nn_descent|cfg_json|Serialize' "${TG}" \
    | tee -a "${PROBE_LOG}" | head -80
else
  echo "missing ${TG}" | tee -a "${PROBE_LOG}"
fi

echo "" | tee -a "${PROBE_LOG}"
echo "==> Interpretation hints" | tee -a "${PROBE_LOG}"
cat <<'EOF' | tee -a "${PROBE_LOG}"
If Knowhere defaults CAGRA graph build to ivf_pq, Catch2 recall 0.0 on gfx1100
matches hipVS Python ivf_pq throw / bad graph — while nn_descent lib smoke is recall 1.0.

Look for:
  - JSON key like "build_algo" / "graph_build_algo"
  - C++ enum passed into cuvs::neighbors::cagra::index_params
  - Hard-coded IVF_PQ in knowhere cuvs/cagra wrapper

Next: patch Knowhere (or test cfg) to nn_descent, rebuild knowhere_tests, re-run:
  "$KT" "Test All GPU Index" -s
EOF

KT=""
for c in \
  "${KH}/build/tests/ut/knowhere_tests" \
  "${KH}/build/knowhere_tests"
do
  if [ -x "$c" ]; then KT="$c"; break; fi
done

if [ "${RUN_CATCH2:-0}" = "1" ]; then
  if [ -z "${KT}" ]; then
    echo "ERROR: knowhere_tests not found" | tee -a "${PROBE_LOG}"
    exit 1
  fi
  CATCH2_LOG="${LOG_DIR}/cagra_catch2_after_probe_${TS}.log"
  echo "==> Catch2 Test All GPU Index -> ${CATCH2_LOG}" | tee -a "${PROBE_LOG}"
  set +e
  "${KT}" "Test All GPU Index" -s 2>&1 | tee "${CATCH2_LOG}"
  _rc=${PIPESTATUS[0]}
  set -e
  echo "Catch2 exit=${_rc}" | tee -a "${PROBE_LOG}"
  echo "FAILED blocks:" | tee -a "${PROBE_LOG}"
  grep -n -A5 'FAILED:' "${CATCH2_LOG}" | tee -a "${PROBE_LOG}" || true
fi

echo ""
echo "PROBE DONE: ${PROBE_LOG}"
echo "If build_algo is found, draft a knowhere patch under patches/knowhere/0052-*.patch"
echo "Then: RUN_CATCH2=1 bash scripts/probe_knowhere_cagra_build_algo.sh"
