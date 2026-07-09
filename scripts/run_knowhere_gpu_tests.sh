#!/usr/bin/env bash
# Run Knowhere GPU unit tests with a safe spdlog load order.
#
# hipRAFT ships install/lib/libspdlog.so.1.14 (needed by libcuvs) but that copy
# is missing symbols Knowhere needs (e.g. set_pattern). Preload apt's
# libspdlog.so.1 so those symbols resolve, while keeping install/lib on
# LD_LIBRARY_PATH for libspdlog.so.1.14 / libcuvs / libraft.
#
# Usage:
#   bash scripts/run_knowhere_gpu_tests.sh
#   bash scripts/run_knowhere_gpu_tests.sh 'Test Gpu Index Search L2 Metric'
set -euo pipefail

WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
INSTALL_PREFIX="${INSTALL_PREFIX:-${WORKDIR}/install}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
KNOWHERE_DIR="${KNOWHERE_DIR:-${WORKDIR}/knowhere}"
TEST_BIN="${TEST_BIN:-${KNOWHERE_DIR}/build/tests/ut/knowhere_tests}"
APT_SPDLOG="${APT_SPDLOG:-/usr/lib/x86_64-linux-gnu/libspdlog.so.1}"

if [ ! -x "${TEST_BIN}" ]; then
  echo "ERROR: knowhere_tests not found: ${TEST_BIN}" >&2
  echo "  Build first: bash scripts/build_knowhere_layer2.sh" >&2
  exit 1
fi

# Restore quarantined hipRAFT spdlog if present (libcuvs needs SONAME 1.14).
shopt -s nullglob
for _f in "${INSTALL_PREFIX}/lib"/libspdlog.so*.hipraft-bak; do
  _orig="${_f%.hipraft-bak}"
  if [ ! -e "${_orig}" ]; then
    mv -v "${_f}" "${_orig}"
  fi
done
unset _f _orig
shopt -u nullglob

if [ ! -e "${APT_SPDLOG}" ]; then
  echo "ERROR: apt spdlog missing: ${APT_SPDLOG}" >&2
  echo "  sudo apt-get install -y libspdlog-dev" >&2
  exit 1
fi

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${APT_SPDLOG}${LD_PRELOAD:+:${LD_PRELOAD}}"

echo "spdlog load check:"
ldd "${KNOWHERE_DIR}/build/libknowhere.so" | grep spdlog || true
echo "LD_PRELOAD=${LD_PRELOAD}"
echo ""

if [ "$#" -eq 0 ]; then
  set -- 'Test Gpu Index Search L2 Metric'
fi

exec "${TEST_BIN}" "$@"
