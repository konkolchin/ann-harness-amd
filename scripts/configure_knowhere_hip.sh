#!/usr/bin/env bash
# Configure knowhere/build for Layer 2 (HIP / prebuilt hipVS).
set -euo pipefail

WORKDIR="${WORKDIR:-${HOME}/rocmds_check_gfx1100}"
INSTALL_PREFIX="${INSTALL_PREFIX:-${WORKDIR}/install}"
KNOWHERE_DIR="${KNOWHERE_DIR:-${WORKDIR}/knowhere}"
BUILD_DIR="${KNOWHERE_DIR}/build"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

export INSTALL_PREFIX
export ROCM_PATH HIP_PATH="${HIP_PATH:-${ROCM_PATH}}"
export PATH="${HOME}/.local/bin:${PATH}"

if ! grep -q 'Early WITH_HIP before project' "${KNOWHERE_DIR}/CMakeLists.txt" 2>/dev/null; then
  echo "ERROR: Layer 2 patches not applied under ${KNOWHERE_DIR}" >&2
  echo "  bash $(dirname "$0")/apply_knowhere_layer2_patches.sh ${KNOWHERE_DIR}" >&2
  exit 1
fi

if [ ! -f "${INSTALL_PREFIX}/lib/cmake/cuvs/cuvs-config.cmake" ]; then
  echo "ERROR: hipVS not found at ${INSTALL_PREFIX}/lib/cmake/cuvs/cuvs-config.cmake" >&2
  exit 1
fi

if [ ! -f "${BUILD_DIR}/Release/generators/conan_toolchain.cmake" ]; then
  echo "ERROR: run conan install first in ${BUILD_DIR} with -o with_cuvs=True" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
rm -f CMakeCache.txt
rm -rf CMakeFiles

cmake .. -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${BUILD_DIR}/Release/generators/conan_toolchain.cmake" \
  -DCMAKE_PREFIX_PATH="${INSTALL_PREFIX};${ROCM_PATH}" \
  -DWITH_CUVS=ON \
  -DWITH_HIP=ON \
  "$@"

echo ""
echo "--- sanity ---"
grep -E 'WITH_CUVS:|WITH_HIP:|CMAKE_CXX_COMPILER:' CMakeCache.txt | head -5
grep -E 'pre-project CXX|WITH_HIP: raft|Configuring done' "${WORKDIR}/knowhere_cmake_hip.log" 2>/dev/null || true
