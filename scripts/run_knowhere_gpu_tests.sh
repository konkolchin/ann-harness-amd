#!/usr/bin/env bash
# Run Knowhere GPU unit tests (Layer 2).
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

if [ ! -x "${TEST_BIN}" ]; then
  echo "ERROR: knowhere_tests not found: ${TEST_BIN}" >&2
  echo "  Build first: bash scripts/build_knowhere_layer2.sh" >&2
  exit 1
fi

# Restore quarantined hipRAFT spdlog if present (libcuvs may DT_NEEDED SONAME 1.14).
shopt -s nullglob
for _f in "${INSTALL_PREFIX}/lib"/libspdlog.so*.hipraft-bak; do
  _orig="${_f%.hipraft-bak}"
  if [ ! -e "${_orig}" ]; then
    mv -v "${_f}" "${_orig}"
  fi
done
unset _f _orig
shopt -u nullglob

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# Optional: if libknowhere still has undefined spdlog (pre-0047 build), try apt preload.
if ! nm -D "${KNOWHERE_DIR}/build/libknowhere.so" 2>/dev/null | grep -q 'T.*spdlog.*set_pattern'; then
  if [ -e /usr/lib/x86_64-linux-gnu/libspdlog.so.1 ]; then
    export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libspdlog.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
    echo "NOTE: set_pattern not in libknowhere.so; LD_PRELOAD apt spdlog (rebuild with patch 0047 preferred)"
  fi
fi

echo "spdlog DT_NEEDED:"
ldd "${KNOWHERE_DIR}/build/libknowhere.so" | grep spdlog || true
echo ""

if [ "$#" -eq 0 ]; then
  set -- 'Test Gpu Index Search L2 Metric'
fi

exec "${TEST_BIN}" "$@"
