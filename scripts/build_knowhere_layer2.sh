#!/usr/bin/env bash
# End-to-end Layer 2 knowhere build on the AMD lab host (apply + configure + build).
#
# Usage:
#   bash scripts/build_knowhere_layer2.sh
#   SKIP_GIT_PULL=1 bash scripts/build_knowhere_layer2.sh   # skip harness git pull
#
# Logs: $WORKDIR/knowhere_build.log
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
KNOWHERE_DIR="${KNOWHERE_DIR:-${WORKDIR}/knowhere}"
BUILD_DIR="${KNOWHERE_DIR}/build"
LOG="${WORKDIR}/knowhere_build.log"

if [ "${SKIP_GIT_PULL:-0}" != "1" ]; then
  echo "==> git pull ${REPO_ROOT}"
  git -C "${REPO_ROOT}" pull
fi

echo "==> apply Layer 2 patches"
bash "${REPO_ROOT}/scripts/apply_knowhere_layer2_patches.sh" "${KNOWHERE_DIR}"

echo "==> configure"
bash "${REPO_ROOT}/scripts/configure_knowhere_hip.sh" 2>&1 | tee "${WORKDIR}/knowhere_cmake_hip.log"

echo "==> build (log: ${LOG})"
set +e
cmake --build "${BUILD_DIR}" -j"$(nproc)" 2>&1 | tee "${LOG}"
_build_rc=${PIPESTATUS[0]}
set -e

if [ "${_build_rc}" -ne 0 ]; then
  echo ""
  echo "BUILD FAILED (exit ${_build_rc}). Last errors:" >&2
  grep -iE 'error:|fatal error:' "${LOG}" | tail -30 >&2 || tail -40 "${LOG}" >&2
  exit "${_build_rc}"
fi

echo ""
echo "BUILD OK"
echo "  lib: ${BUILD_DIR}/libknowhere.so"
echo "  tests: ${BUILD_DIR}/tests/ut/knowhere_tests"
echo ""
echo "Run GPU tests (Catch2 v2: pass test name as positional arg):"
echo "  export LD_LIBRARY_PATH=\"${WORKDIR}/install/lib:/opt/rocm/lib:\${LD_LIBRARY_PATH}\""
echo "  ${BUILD_DIR}/tests/ut/knowhere_tests 'Test All GPU Index'"
