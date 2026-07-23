#!/usr/bin/env bash
# Time shared cuVS gtests on NVIDIA (peer compare vs hipVS RDNA3).
#
# Prerequisites: cuVS built with tests under WORKDIR (or set GTEST_DIR).
# Typical:
#   WORKDIR=~/milvus_cuda_4080
#   # cuVS clone with ./build.sh libcuvs tests ...
#
# Usage:
#   WORKDIR=~/milvus_cuda_4080 bash scripts/run_cuvs_gtest_timing.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${HOME}/milvus_cuda_4080}"
GTEST_DIR="${GTEST_DIR:-${WORKDIR}/cuvs/cpp/build/gtests}"
# Fallbacks if the tree lives elsewhere
if [ ! -d "${GTEST_DIR}" ] && [ -d "${WORKDIR}/hipVS/cpp/build/gtests" ]; then
  # some labs keep the NVIDIA tree named differently
  true
fi
if [ ! -d "${GTEST_DIR}" ]; then
  for cand in \
    "${WORKDIR}/cuvs/cpp/build/gtests" \
    "${HOME}/cuvs/cpp/build/gtests" \
    "${WORKDIR}/rapids/cuvs/cpp/build/gtests"
  do
    if [ -d "${cand}" ]; then
      GTEST_DIR="${cand}"
      break
    fi
  done
fi

LOG_DIR="${LOG_DIR:-${WORKDIR}/logs}"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_JSON="${RESULTS_JSON:-${LOG_DIR}/gtest_timing_cuvs_${TS}.json}"
SAVE_LOGS="${SAVE_LOGS:-${LOG_DIR}/gtest_timing_cuvs_${TS}_logs}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [ ! -d "${GTEST_DIR}" ]; then
  echo "ERROR: gtest dir not found. Set GTEST_DIR to cuVS cpp/build/gtests" >&2
  echo "  searched under WORKDIR=${WORKDIR}" >&2
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

echo "==> cuVS shared gtest timing"
echo "    gtest_dir=${GTEST_DIR}"
echo "    results=${RESULTS_JSON}"

cd "${REPO_ROOT}"
python3 scripts/time_cuvs_gtests.py \
  --backend cuvs \
  --gtest-dir "${GTEST_DIR}" \
  --results-json "${RESULTS_JSON}" \
  --save-logs-dir "${SAVE_LOGS}" \
  "${EXTRA[@]}"

echo ""
echo "CUVS GTEST TIMING OK"
echo "  results: ${RESULTS_JSON}"
echo "  raw logs: ${SAVE_LOGS}"
