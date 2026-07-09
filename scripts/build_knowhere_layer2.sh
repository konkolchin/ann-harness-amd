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

echo "==> conan install (with_cuvs + with_ut for knowhere_tests)"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
if ! grep -q 'WITH_UT:BOOL=ON\|WITH_UT:STRING=True' CMakeCache.txt 2>/dev/null; then
  conan install .. --build=missing -s build_type=Release \
    -o with_cuvs=True \
    -o with_ut=True
fi
cd "${REPO_ROOT}"

echo "==> configure"
bash "${REPO_ROOT}/scripts/configure_knowhere_hip.sh" 2>&1 | tee "${WORKDIR}/knowhere_cmake_hip.log"

# Fail fast if spdlog was not wired into knowhere (avoids long rebuild then link fail).
if ! grep -qE 'KNOWHERE WITH_HIP: knowhere (linked|WHOLE_ARCHIVE).*spdlog' \
     "${WORKDIR}/knowhere_cmake_hip.log" 2>/dev/null; then
  echo "WARNING: cmake log has no 'knowhere ... spdlog' link line." >&2
  echo "  Check: ls /usr/lib/x86_64-linux-gnu/libspdlog.so*" >&2
  echo "  and: grep spdlog ${WORKDIR}/knowhere_cmake_hip.log" >&2
fi

echo "==> force-relink libknowhere.so (keep host logger objects from patch 0042)"
rm -f "${BUILD_DIR}/libknowhere.so" "${BUILD_DIR}/libknowhere.so."* 2>/dev/null || true
# Drop only logger objects wrongly compiled into the HIP knowhere target (not host logger lib).
# Do NOT sed CMakeCache.txt — that corrupts the cache (cmake_check_build_system parse errors).
# configure_knowhere_hip.sh already wipes the build tree (keeps Conan Release/).
find "${BUILD_DIR}" -path '*/CMakeFiles/knowhere.dir/*' -name 'logger.cpp.o' -delete 2>/dev/null || true

# Prefer apt spdlog; hipRAFT install/lib/libspdlog.so often lacks set_pattern.
if [ ! -e /usr/lib/x86_64-linux-gnu/libspdlog.so ] && [ ! -e /usr/lib/x86_64-linux-gnu/libspdlog.a ]; then
  echo "WARNING: apt libspdlog not found. Install: sudo apt-get install -y libspdlog-dev" >&2
fi
if [ -e "${INSTALL_PREFIX:-${WORKDIR}/install}/lib/libspdlog.so" ]; then
  echo "NOTE: ${INSTALL_PREFIX:-${WORKDIR}/install}/lib/libspdlog.so exists; patch 0046 prefers /usr/lib/... instead"
fi

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
echo "  ${BUILD_DIR}/tests/ut/knowhere_tests 'Test Gpu Index Search L2 Metric'"
echo "  (Do not use 'Test All GPU Index' on gfx1100 — CAGRA/brute-force fail.)"
