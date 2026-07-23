#!/usr/bin/env bash
# Time shared hipVS gtests on AMD (RDNA3 / warp_size=32 path).
# Manager ask: is hipVS competitive with cuVS?  → same unit tests + wall clock.
#
# Prerequisites: hipVS built with tests, e.g.
#   ./build.sh libcuvs tests --gpu-arch=gfx1100 '--cmake-args=-DUSE_WARPSIZE_32=ON'
#
# Usage:
#   bash scripts/run_hipvs_gtest_timing.sh
#   GTEST_BINARIES=NEIGHBORS_ANN_IVF_FLAT_TEST bash scripts/run_hipvs_gtest_timing.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
GTEST_DIR="${GTEST_DIR:-${WORKDIR}/hipVS/cpp/build/gtests}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_JSON="${RESULTS_JSON:-${LOG_DIR}/gtest_timing_hipvs_${TS}.json}"
SAVE_LOGS="${SAVE_LOGS:-${LOG_DIR}/gtest_timing_hipvs_${TS}_logs}"

export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="${WORKDIR}/install/lib:${ROCM_HOME}/lib:${LD_LIBRARY_PATH:-}"

if [ ! -d "${GTEST_DIR}" ]; then
  echo "ERROR: gtest dir not found: ${GTEST_DIR}" >&2
  echo "  Build hipVS tests, or set GTEST_DIR=..." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
EXTRA=()
if [ -n "${GTEST_BINARIES:-}" ]; then
  EXTRA+=(--binaries "${GTEST_BINARIES}")
fi
if [ "${NO_DEFAULT_FILTERS:-0}" = "1" ]; then
  EXTRA+=(--no-default-filters)
fi

echo "==> hipVS shared gtest timing"
echo "    gtest_dir=${GTEST_DIR}"
echo "    results=${RESULTS_JSON}"

cd "${REPO_ROOT}"
python3 scripts/time_cuvs_gtests.py \
  --backend hipvs \
  --gtest-dir "${GTEST_DIR}" \
  --results-json "${RESULTS_JSON}" \
  --save-logs-dir "${SAVE_LOGS}" \
  "${EXTRA[@]}"

echo ""
echo "HIPVS GTEST TIMING OK"
echo "  results: ${RESULTS_JSON}"
echo "  raw logs: ${SAVE_LOGS}"
