#!/usr/bin/env bash
# Phase A — reproduce CAGRA gaps on consumer gfx1100 (RX 7900 XTX).
#
# Runs:
#   1) Knowhere Catch2 CAGRA-related filters (expect recall 0.0 until fixed)
#   2) Optional hipVS-only Python CAGRA smoke (small N) to split library vs Knowhere
#
# Usage (on amd-rx7900xtx):
#   export WORKDIR=~/rocmds_check_gfx1100
#   bash ~/ann-harness-amd/scripts/reproduce_cagra_gfx1100.sh
#   SKIP_HIPVS=1 bash scripts/reproduce_cagra_gfx1100.sh   # Catch2 only
#   SKIP_CATCH2=1 bash scripts/reproduce_cagra_gfx1100.sh  # hipVS only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
TS="$(date +%Y%m%d_%H%M%S)"
CATCH2_LOG="${LOG_DIR}/cagra_catch2_${TS}.log"
HIPVS_LOG="${LOG_DIR}/cagra_hipvs_minimal_${TS}.log"
SUMMARY="${LOG_DIR}/cagra_repro_summary_${TS}.md"

export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export ROCM_PATH
export LD_LIBRARY_PATH="${WORKDIR}/knowhere/build:${WORKDIR}/install/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${LOG_DIR}"

{
  echo "# CAGRA gfx1100 repro summary"
  echo ""
  echo "- host: $(hostname)"
  echo "- date: $(date -Iseconds)"
  echo "- WORKDIR: ${WORKDIR}"
  echo "- ROCm: ${ROCM_PATH} ($(cat ${ROCM_PATH}/.info/version 2>/dev/null || echo unknown))"
  echo "- HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
  echo ""
  echo "## Baseline (documented 2026-07-26)"
  echo ""
  echo "| Section | Index | Assert | Got |"
  echo "|---------|-------|--------|-----|"
  echo "| Search TopK | GPU_CUVS_CAGRA | recall > 0.7 | 0.0 |"
  echo "| Serialize/Deserialize | GPU_CUVS_CAGRA | recall >= 0.8 | 0.0 |"
  echo ""
  echo "Layer-2 gate (Test Gpu Index Search, 108 asserts) remains PASS without CAGRA exit."
  echo ""
} >"${SUMMARY}"

KT=""
for c in \
  "${WORKDIR}/knowhere/build/tests/ut/knowhere_tests" \
  "${WORKDIR}/knowhere/build/knowhere_tests"
do
  if [ -x "$c" ]; then KT="$c"; break; fi
done

if [ "${SKIP_CATCH2:-0}" != "1" ]; then
  if [ -z "${KT}" ]; then
    echo "WARNING: knowhere_tests not found under ${WORKDIR}/knowhere/build" | tee -a "${SUMMARY}"
    echo "  Build Knowhere WITH_HIP first (Layer-2)." | tee -a "${SUMMARY}"
  else
    echo "==> Catch2 CAGRA filters via ${KT}"
    set +e
    # Catch2: filter by name substring; sections mention CAGRA / Gpu
    "${KT}" -k "CAGRA" -s 2>&1 | tee "${CATCH2_LOG}"
    _rc=${PIPESTATUS[0]}
    set -e
    {
      echo "## Catch2 (-k CAGRA)"
      echo ""
      echo "- binary: \`${KT}\`"
      echo "- log: \`${CATCH2_LOG}\`"
      echo "- exit: ${_rc}"
      echo ""
      echo '```'
      grep -a -E 'FAILED|failed|passed|All tests|recall|CAGRA|assertions' "${CATCH2_LOG}" | tail -40 || true
      echo '```'
      echo ""
    } >>"${SUMMARY}"
    echo "Catch2 exit=${_rc} (non-zero expected until CAGRA fixed)"
  fi
fi

if [ "${SKIP_HIPVS:-0}" != "1" ]; then
  echo "==> hipVS-only CAGRA minimal (10k train / 200 query)"
  set +e
  (
    cd "${REPO_ROOT}"
    DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/sift-128-euclidean.hdf5}"
    if [ ! -f "${DATA_PATH}" ]; then
      echo "missing ${DATA_PATH}"
      exit 2
    fi
    if ! python3 -c "from cuvs.neighbors import cagra" 2>/dev/null; then
      echo "cuvs.neighbors.cagra not importable — skip hipVS minimal"
      exit 3
    fi
    MAX_TRAIN_ROWS=10000 MAX_QUERY_ROWS=200 ITOPK_SIZES=64 \
      RESULTS_JSON="${LOG_DIR}/lib_hipvs_cagra_minimal_${TS}.json" \
      bash "${REPO_ROOT}/scripts/run_hipvs_cagra_bench.sh"
  ) 2>&1 | tee "${HIPVS_LOG}"
  _hrc=${PIPESTATUS[0]}
  set -e
  {
    echo "## hipVS-only minimal"
    echo ""
    echo "- log: \`${HIPVS_LOG}\`"
    echo "- exit: ${_hrc}"
    echo ""
    if grep -a -q 'recall@10=' "${HIPVS_LOG}"; then
      echo '```'
      grep -a 'recall@10=' "${HIPVS_LOG}" | tail -10
      echo '```'
      echo ""
      echo "Triage: if hipVS recall ~0 → library/kernel on gfx1100; if hipVS OK but Catch2 0 → Knowhere wiring/serialize."
    else
      echo "No recall lines (import/build failure). Escalate hipVS Python / install."
    fi
    echo ""
  } >>"${SUMMARY}"
fi

echo ""
echo "REPRO DONE — summary: ${SUMMARY}"
echo "Next: docs/cagra_consumer_followon.md §Triage ladder"
cat "${SUMMARY}"
