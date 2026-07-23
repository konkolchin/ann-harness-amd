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
HIPVS_ROOT="${HIPVS_ROOT:-${WORKDIR}/hipVS}"
GTEST_DIR="${GTEST_DIR:-}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_JSON="${RESULTS_JSON:-${LOG_DIR}/gtest_timing_hipvs_${TS}.json}"
SAVE_LOGS="${SAVE_LOGS:-${LOG_DIR}/gtest_timing_hipvs_${TS}_logs}"

export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="${WORKDIR}/install/lib:${ROCM_HOME}/lib:${LD_LIBRARY_PATH:-}"

# Resolve gtest dir (Layer-1 default, then common alternates)
if [ -z "${GTEST_DIR}" ]; then
  for cand in \
    "${HIPVS_ROOT}/cpp/build/gtests" \
    "${WORKDIR}/hipVS/cpp/build/gtests" \
    "${WORKDIR}/hipvs/cpp/build/gtests" \
    "${HOME}/hipVS/cpp/build/gtests"
  do
    if [ -x "${cand}/NEIGHBORS_ANN_IVF_FLAT_TEST" ] || [ -d "${cand}" ]; then
      GTEST_DIR="${cand}"
      break
    fi
  done
fi

if [ -z "${GTEST_DIR}" ] || [ ! -d "${GTEST_DIR}" ]; then
  echo "ERROR: hipVS gtest dir not found." >&2
  echo "  Searched under WORKDIR=${WORKDIR}" >&2
  echo "  Find existing binaries:" >&2
  echo "    find \"${WORKDIR}\" \"${HOME}\" -name NEIGHBORS_ANN_IVF_FLAT_TEST -type f 2>/dev/null | head" >&2
  echo "  Or rebuild (float IVF suite ≈ 98 cases — enough for manager compare):" >&2
  echo "    cd \"${HIPVS_ROOT}\"" >&2
  echo "    export INSTALL_PREFIX=\"\${WORKDIR}/install\"" >&2
  echo "    INSTALL_PREFIX=\$INSTALL_PREFIX ./build.sh libcuvs tests \\" >&2
  echo "      --gpu-arch=gfx1100 \\" >&2
  echo "      '--cmake-args=-DUSE_WARPSIZE_32=ON -DBUILD_CAGRA_HNSWLIB=OFF' \\" >&2
  echo "      --limit-tests=NEIGHBORS_ANN_IVF_FLAT_TEST" >&2
  echo "    GTEST_DIR=\${HIPVS_ROOT}/cpp/build/gtests \\" >&2
  echo "      GTEST_BINARIES=NEIGHBORS_ANN_IVF_FLAT_TEST \\" >&2
  echo "      bash ${REPO_ROOT}/scripts/run_hipvs_gtest_timing.sh" >&2
  exit 1
fi

if [ ! -x "${GTEST_DIR}/NEIGHBORS_ANN_IVF_FLAT_TEST" ]; then
  echo "WARN: ${GTEST_DIR}/NEIGHBORS_ANN_IVF_FLAT_TEST missing — rebuild tests (see ERROR hint above)." >&2
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
